# kd1-anime Architecture

本文档描述当前实现，而不是未来设计草案。

## 1. 设计目标

- 使用原生 Python、Pydantic 和显式有限状态机，不引入重型 Agent 框架。
- 将独立场景拆成可并行的规划、代码生成、审查和 Slurm 渲染单元。
- 对模型输出同时执行 LLM 语义审查与确定性 AST 校验。
- 让每次运行拥有隔离的代码、日志、媒体和输出目录。
- 在集群故障、LLM 格式错误、渲染失败和视频编码差异下提供清晰的失败边界。

## 2. 组件

```text
kd1_anime.cli / kd1_anime.tui
       │
       ▼
kd1_anime.orchestrator ───── callback events ───────────▶ TUI/Rich
       │
       ├── agents/planner.py       概要规划 + 详细分镜
       ├── agents/coder.py         ManimCE 代码生成/重写
       ├── agents/reviewer.py      结构化语义审查
       ├── agents/validator.py     AST 确定性校验
       ├── agents/auto_fixer.py    根据渲染日志修复代码
       ├── cluster/slurm.py        sbatch、批量状态查询、超时取消
       ├── media/merger.py         精确收集视频、FFmpeg 拼接
       └── run_store.py            原子清单、代码哈希、运行锁和恢复路径校验
```

`src/kd1_anime/agents/base.py` 是全部 LLM Agent 的公共层，封装 OpenAI-compatible client、重试、流式输出、JSON/代码提取和 Pydantic 校验。

## 3. 状态机

```text
INIT
  → PLANNING
  → DETAILING
  → CODING
  → REVIEWING ───────────────┐
  → DISPATCHING              │ 代码改变后重新审查
  → MONITORING               │
      ├─ 全部成功 → MERGING  │
      └─ 有失败 → FIXING ────┘
  → DONE
```

`ERROR` 是终止状态。任何未处理异常或不允许的部分输出都会触发失败；用户中断时会尝试取消仍在运行的 Slurm job。

### 3.1 PLANNING / DETAILING

Planner 使用两阶段结构化输出：

1. `plan_outline()` 生成短小的 `SceneOutline` 列表，并把 scene ID 按返回顺序规范化为 `1..N`。
2. `plan_detail()` 同时接收原始需求、全部概要和当前概要，生成包含视觉设计、镜头、动画流、关键时刻和计算说明的 `ScenePlan`。

详细分镜彼此独立，使用 `ThreadPoolExecutor` 并发调用。worker 禁用流式 stdout；失败后的交互式 retry 只在主线程执行，避免多个线程争用终端输入。

### 3.2 CODING / REVIEWING

Coder 为每个 Scene 生成一个 Python 文件，并明确禁止网络、文件读写、shell、subprocess 和动态执行。

生成结果先通过 `agents.validator.validate_manim_code()`：

- Python 必须可解析；
- 只允许 Manim、NumPy、math 和少量纯计算标准库；
- 禁止危险函数、危险属性和模块顶层执行；
- 每个文件必须且只能定义一个直接继承支持的 Scene 类；
- Scene 类必须实现 `construct()`；
- `Tex`/`MathTex` 必须显式使用已注册到 `config.tex_template` 的 XeLaTeX `.xdv` 模板，并加载 `ctex`。

若校验失败，Coder 会得到确定性反馈并重写，最多 `CODE_VALIDATION_ATTEMPTS` 次。

Reviewer 再执行数学、LaTeX、Manim API、动画生命周期、布局和安全方面的语义审查。返回结构由 Pydantic 强约束：

- `severity` 只能为 `minor` 或 `major`；
- minor 必须给出可唯一匹配的查找替换；
- major 必须给出重写反馈；
- valid 结果会被规范化，清空无意义的反馈。

Reviewer 同时接收完整 `ScenePlan`，用于核对叙事作用、数学规格、视觉流程、镜头和关键时刻，而不只检查代码能否运行。

所有未通过审查的轮次共同受 `MAX_REVIEW_ROUNDS` 限制，避免 minor 修复形成无限循环。代码一旦改变，必须重新进入 REVIEWING；通过后才可提交。

### 3.3 DISPATCHING / MONITORING

`src/kd1_anime/cluster/slurm.py` 直接生成 sbatch 字符串，不使用 Jinja 模板。每个场景对应一个 Slurm job，因此场景可以由调度器并行执行。

资源策略：

- Cairo 仅申请 CPU；即使设置 GPU 类型也不会添加 `--gres`。
- OpenGL 必须配置 `SLURM_GPU_TYPE`，并添加 GPU 资源请求。
- conda base 优先使用 `SLURM_CONDA_BASE`，否则加载 module 并执行 `conda info --base` 动态探测。
- 可选 Apptainer 执行使用 `--containall --cleanenv --no-home`，只绑定当前 run 根目录；GPU 作业增加 `--nv`。`render` 子命令会先把外部源码复制进当前 run，避免容器不可见和提交后源码变化。

监控器批量调用 `squeue`，并使用 `sacct` 查询已离开队列的最终状态。它分别追踪：

- `MONITOR_QUEUE_TIMEOUT`：尚未开始运行的排队时间；
- `MONITOR_RUN_TIMEOUT`：第一次进入运行态后的运行时间；
- `MONITOR_MAX_UNKNOWN`：连续无法确认状态的次数。

超时或连续 UNKNOWN 时调用 `scancel`，而不是只在客户端停止等待。取消失败会保留为 `CANCEL_FAILED`，禁止自动重提，以避免原作业仍运行时产生重复作业。`sbatch` 使用 `--parsable`；命令超时属于提交状态不确定，不自动重试。

### 3.4 FIXING

失败场景从精确的 stderr 路径读取最后 `LOG_TAIL_LINES` 行，并再受 `MAX_LOG_CHARS` 限制。环境、conda、Apptainer、Slurm 配置和显示服务错误不会交给 LLM 重写业务代码。其余错误交给 AutoFixer，修复结果再次经过 AST 校验；不通过时由 Coder 根据校验反馈和原始错误重写。修复后的代码会重新进入 REVIEWING，而不是直接提交。

每个场景最多自动修复 `MAX_FIX_ATTEMPTS` 次。

### 3.5 持久化与恢复

Orchestrator 在每次 FSM 转换以及每个 Slurm 提交后，使用同目录临时文件和 `os.replace()` 原子更新 `manifest.json`。清单包含原始需求、场景规划、代码 SHA-256、审查/修复次数、精确 Slurm Job ID 与媒体路径、最终输出和最后错误。

`kd1-anime resume <run-id>` 在持有 `.run.lock` 排他锁后重新读取清单：

- 代码文件必须仍位于当前 run 且 SHA-256 匹配；
- 已提交且未取消的 Job ID 会继续监控，不会重新提交；
- 中断时成功取消的作业会清除旧 Job ID，再进入 DISPATCHING；
- `ERROR` 终态和已被修改的代码拒绝自动恢复；
- 两个进程不能同时恢复同一 run。

`status` 只读取清单，不调用 LLM/Slurm。`clean` 默认只删除超过保留期的非 running 运行，并用同一把锁跳过仍活跃的进程。

### 3.6 MERGING

VideoMerger 不扫描共享目录猜测产物，而是使用 `SlurmJob.media_dir` 和 `scene_class_name` 精确定位当前 run 的最终 MP4，并排除 `partial_movie_files`。

合并顺序按 `scene_id`：

1. 首先尝试 FFmpeg concat stream copy；
2. 编码参数不兼容时回退到 H.264/AAC；
3. 回退编码使用等比例缩放 + padding，避免横竖屏被拉伸；
4. FFmpeg 写入同目录临时文件，成功后原子替换目标；失败不会留下半成品；
5. 自定义输出默认拒绝覆盖，必须显式使用 `--force`/`OVERWRITE_OUTPUT=true`。

默认 `ALLOW_PARTIAL_OUTPUT=false`。任一场景未完成时不会静默生成残缺视频。

## 4. 运行目录

`RunPaths.create()` 为每次运行生成唯一目录：

```text
workspace/runs/<YYYYMMDD-HHMMSS>-<uuid8>/
├── prompt.txt
├── manifest.json
├── .run.lock
├── scenes/
├── logs/
├── videos/
└── output_final.mp4
```

这解决了重复运行、并发运行和旧 Manim 产物被误选的问题。run 根目录权限为 `0700`，prompt、manifest、锁文件和生成代码为 `0600`。若 `OUTPUT_FILE` 被显式配置，只有最终视频写到指定位置。

## 5. 配置

`config.Settings` 使用 pydantic-settings，加载顺序为：

```text
环境变量 > 当前目录 .env > ~/.config/kd1-anime/.env
```

重要配置分组：

- LLM：`LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`、重试、token 和并发数；
- Slurm：分区、QoS、account、CPU/内存/GPU、最大在途作业数、conda、容器；
- Manim：renderer、quality、分辨率、帧率和 OpenGL platform；
- 流水线：审查/修复次数、排队/运行超时、日志截断和部分输出策略；
- 路径：`WORKSPACE_DIR`、`OUTPUT_FILE`。

`MONITOR_TIMEOUT` 仅用于兼容旧配置；新配置应使用拆分后的 queue/run timeout。

## 6. 安全边界

AST 校验用于拒绝明显危险或不符合项目结构的代码，但 Python 静态分析不是完整沙箱。高信任要求场景应开启 Apptainer，并使用只包含渲染依赖的只读镜像。`SLURM_REQUIRE_CONTAINER=true` 可防止配置遗漏时回退到宿主 conda 环境。

## 7. 打包与安装

- wheel 只包含 Python 运行时模块，不包含集群专用 `install.sh` 和运行产物。
- `install.sh` 可独立下载；远程模式通过临时 GitHub ZIP 安装，不 clone 源码。
- 本地源码模式执行 editable install。
- 安装器无 sudo，优先复用 PATH 或 `/usr/local/texlive` 中完整的 XeLaTeX，不修改完整的系统安装。
- 仅在现有环境不可用时，将最小 TeX Live 安装到用户主目录；CJK 支持只补 `ctex`、`xeCJK` 和 `fontspec` 等必需包。
- 安装器创建用户级 `.env`，但不覆盖已有配置。

## 8. 测试与 CI

测试覆盖：

- JSON/LaTeX 转义；
- Reviewer schema；
- AST 安全校验；
- Reviewer 状态机退出与轮次上限；
- Cairo/OpenGL 资源策略和 Slurm 超时取消；
- 精确视频选择和 partial 文件排除；
- run 路径唯一性；
- manifest 原子持久化、代码哈希恢复、路径逃逸拒绝和过期清理；
- Slurm 最大在途作业限制；
- 通用 LLM 配置校验。

CI 执行 Ruff、compileall、`bash -n install.sh`、pytest 和 wheel 构建。
