# kd1-anime Architecture

本文档描述当前实现，而不是未来设计草案。

## 1. 设计目标

- 使用原生 Python、Pydantic 和显式有限状态机，不引入重型 Agent 框架。
- 让场景分镜可并行生成、代码按顺序交接，随后独立完成 Slurm 渲染与修复。
- 对模型输出同时执行 LLM 语义审查与确定性 AST 校验。
- 让每次运行拥有隔离的代码、日志、媒体和输出目录。
- 用可验证的产物身份避免旧视频、错误配置和错误 Job 被误复用。
- 在集群故障、LLM 格式错误、渲染失败和视频编码差异下提供清晰失败边界。

## 2. 组件

```text
kd1_anime.cli / kd1_anime.tui
       │
       ▼
kd1_anime.orchestrator ───── callback events ───────────▶ TUI/Rich
       │
       ├── agents/planner.py       概要规划 + 详细分镜
       ├── agents/continuity.py    全片连续性审查与局部重规划
       ├── agents/coder.py         ManimCE 代码生成/重写
       ├── agents/reviewer.py      结构化语义审查
       ├── agents/validator.py     AST 确定性校验
       ├── agents/auto_fixer.py    根据渲染日志修复代码
       ├── agents/render_context.py renderer 能力与对象生命周期约束
       ├── cluster/slurm.py        sbatch、状态查询、产物验证、超时取消
       ├── rendering.py            RenderProfile、ffprobe、SceneArtifact
       ├── resources.py            跨批量项目的进程级 LLM/RAG/Slurm 配额
       ├── media/merger.py         精确输入列表、FFmpeg xfade/acrossfade 原子合并
       ├── rag/                    SQLite 索引、独立 Embedding/Reranker 检索
       ├── eval/                   代码/效率评估与独立多模态视觉质量门
       └── run_store.py            Manifest v3、原子检查点、运行锁
```

`agents/base.py` 封装 OpenAI-compatible client、重试、静默流式传输、JSON/代码提取和 Pydantic 校验。非空但 `finish_reason=length` 的响应不会被消费；系统会提高输出预算并完整重试，持续截断时抛出明确错误。

## 3. 执行模型

```text
全局：INIT → LLM 可用性探测 → PLANNING

分镜屏障：所有 Scene 并行 DETAILING → 全片连续性审查

顺序代码屏障：
Scene 1 CODING → REVIEWING → Scene 2 CODING → REVIEWING → …

渲染阶段：各 Scene 并行 DISPATCHING → MONITORING
                                      ▲              │
                                      └── FIXING ◀────┘

全部线程结束：VISUAL_EVALUATING → (低分回到 CODING) → MERGING
                                      → 成片视觉报告
                                      → (EVALUATING → 可定位场景回到 CODING) → DONE
```

FSM 枚举同时用于清单检查点和 TUI 阶段提示。分镜仍然并行，但编码/审查按场景顺序执行：Scene N 审查通过后，提取其连续性导出区，才允许 Scene N+1 编码。这样 Coder 收到的是上一场景真实生成的最终 Mobject 定义，而不是仅凭 Planner 描述猜测的状态。所有代码就绪后，Slurm 渲染继续并行；每个 worker 使用独立 Agent 实例并关闭流式终端输出，也不读取共享 stdin。

LLM 调用受 `LLM_PARALLEL_WORKERS` 信号量限制；RAG 请求受独立的 `RAG_PARALLEL_WORKERS` 信号量限制；Slurm 提交受 `SLURM_MAX_IN_FLIGHT` 限制。批量模式中的多个 Orchestrator 共享同一个 `ResourceCoordinator`，不会把每项目配额相乘。
CLI 在进入 chat、规划、生成或恢复需要 Agent 的运行前，会用短超时发送一次主 LLM 请求；探测失败直接退出，不把明显的配置/网络问题拖到 Clarifier 或 Planner 阶段才暴露。视觉评估使用完全独立的 Key、URL、模型、超时和并发配置；批处理中的多个 Orchestrator 共享进程级视觉并发配额。配置缺失会在启动前失败，网络探测暂时失败时生成流水线降级为 `unknown`；显式 `evaluate --visual` 则失败退出。`status`、`render`、`clean`、已完成运行恢复和纯代码评估不依赖这些探测。

`ERROR` 是失败检查点。任何未处理异常或不允许的部分输出都会触发失败；用户中断时会尝试取消仍在运行的 Job。

RAG 索引使用 SQLite 保存文本分块、元数据和 Embedding BLOB。Markdown 和 reStructuredText 按标题/段落切分，Python 按顶层定义切分；索引构建通过临时数据库原子替换。运行时先做本地余弦初排，再调用独立 Reranker；服务故障只记录 `degraded` 并继续原有流水线。Planner、Coder 和 AutoFixer 收到的检索内容均标记为不可信资料，且每次注入都保存查询、索引和分块哈希收据。

### 3.1 PLANNING / DETAILING

Planner 使用分层结构化输出：

1. `plan_outline()` 按最小必要粒度生成短小 `SceneOutline` 列表，并按返回顺序规范化 scene ID 为 `1..N`。同一画布中逐个出现、保留并对比的对象属于同一个场景；当用户明确要求同屏/整体展示而模型仍按对象拆分时，Planner 会将概要确定性合并为一个场景。只有用户明确要求多场景，或镜头/布局/叙事弧线确实独立时才拆分。
2. `plan_continuity_bible()` 在分镜并行前固定全片背景、调色板、字体、布局、数学符号、持续对象、镜头语言和转场规则，并写入运行清单。
3. 每个 worker 的 `plan_detail()` 接收原始需求、全部概要、相邻概要和 continuity bible，生成视觉设计、镜头、动画流、关键时刻、计算说明以及 opening/closing state、结构化 `inherited_elements` / `elements_to_remove` / `new_elements` 和转场合同。
4. 所有 Detail 完成后执行确定性检查和一次全片连续性审查；冲突只重规划未进入编码的相关场景，受 `MAX_CONTINUITY_FIX_ROUNDS` 限制。

Pydantic 模型拒绝未知字段并限制字符串、列表和场景数量。用户需求被明确标记为不可信数据，不能改变系统规则。
`GlobalVisualState` 固定全片颜色、字体、字号、线宽、布局锚点和镜头语言；每个 `ScenePlan` 都携带同一份只读配置。`VisualElementState` 为跨场景对象分配稳定的 `element_id`。

### 3.2 CODING / REVIEWING

Coder 为每个 Scene 生成一个 Python 文件，并明确禁止网络、文件读写、shell、subprocess 和动态执行。Coder、Reviewer 和 AutoFixer 都收到当前 renderer 能力说明：

- OpenGL 禁止 `self.camera.frame`、`MovingCameraScene` 运镜和自定义 Mobject 根类子类；
- Cairo 只有 `MovingCameraScene` 可使用 frame API；
- 3D Scene 使用专用相机 API；
- introducer、Transform、FadeOut 等必须遵守对象生命周期。

生成结果先通过 `validate_manim_code()`：

- Python 必须可解析；
- 只允许 Manim、NumPy、math 和少量纯计算标准库；
- 禁止危险函数、危险属性、别名逃逸和模块顶层执行；
- 每个文件必须且只能定义一个直接继承支持基类的 Scene 类；
- Scene 类必须实现 `construct()`；
- 使用 `Tex`/`MathTex` 时必须显式使用注册到 `config.tex_template` 的 XeLaTeX `.xdv` 模板并加载 `ctex`。

若校验失败，确定性反馈会交回 Coder，最多尝试 `CODE_VALIDATION_ATTEMPTS` 次。Coder 必须在代码中提供 `KD1_CONTINUITY_EXPORT_BEGIN/END` 区，区内只允许纯 Mobject 定义；Orchestrator 通过 AST 安全提取并保存为下一场景的 `[Inherited Elements Code]`。Reviewer 再检查数学、LaTeX、Manim API、动画生命周期、布局、安全、分镜符合度和元素交接。结构化输出只允许：

- valid / `info`：通过；
- `minor`：至少一条可精确唯一匹配的查找替换；
- `major`：带具体反馈并回到 Coder 重写。

审查受 `MAX_REVIEW_ROUNDS` 限制。任何代码变化都会把 `reviewed` 重置为 false。AutoFix 输出也必须重新进入 Reviewer；major 反馈仍回到 CODING，绝不直接提交。

连续性审查结果和警告也保存到 `manifest.json`；resume 会复用已保存的 continuity bible，不会因为重启而重新生成一套风格规范。如果上游代码改变，尚未提交渲染的下游场景会清除旧交接代码并按顺序重新编码；恢复旧清单时会优先重新提取导出区，提取失败不会静默复用下游状态。

### 3.3 DISPATCHING / MONITORING

`cluster/slurm.py` 直接构建 sbatch 脚本，并对所有 directive 值使用配置层单行校验和 shell quoting。

资源策略：

- Cairo 只申请 CPU；OpenGL 必须配置 GPU 类型并申请 GPU；
- Manim 命令显式传入 renderer、分辨率和帧率；OpenGL 显式启用 `--write_to_movie`；
- conda base 优先使用配置，否则加载 module 并动态探测；
- 可选 Apptainer 使用 `--containall --cleanenv --no-home`，只绑定当前 run；OpenGL 增加 `--nv` 并显式传递 `PYOPENGL_PLATFORM`。

每次成功提交都会保存数字 Job ID、提交时间、代码哈希和 RenderProfile。监控区分：

- 正常调度器状态；
- `GONE`：squeue 可达，但作业不在队列且 sacct 无记录；
- `UNKNOWN`：调度器查询失败，不能确认作业是否存在。

`UNKNOWN` 达阈值时先取消；`scancel` 失败会进入 `CANCEL_FAILED` 并禁止自动重提。`GONE` 会依据当前 Job 的最终视频和日志分类，不会把查询故障等同于作业消失。被抢占退回排队后会重置运行计时，避免误触发 run timeout。

Job 只有在最终 MP4 通过 ffprobe、目标分辨率和帧率验证后才算成功。每次提交都使用独立的 `attempt_<token>` 媒体目录；定位会递归适配 Manim 嵌套层级、排除 `partial_movie_files` 和早于本次提交的文件，不会把上一次修复的 MP4 当成当前产物。

### 3.4 FIXING

失败场景只读取精确 Job 的 stderr 尾部，并受 `LOG_TAIL_LINES` 和 `MAX_LOG_CHARS` 限制。环境、conda、容器、Slurm、显示服务和字体错误不会交给 LLM 重写业务代码。

其余错误交给 AutoFixer，结果再次通过 AST 校验；不通过时由 Coder 根据校验反馈和原始错误重写。修复后的代码强制复审。修复次数和连续相同错误次数都有上限。

### 3.5 持久化与恢复

Orchestrator 在关键阶段和每次 Slurm 提交后更新 `manifest.json`：写同目录临时文件、文件 `fsync`、`os.replace()`、目录 `fsync`。schema v3 包含单调 revision、场景 phase、代码哈希、审查/修复次数、精确 Job、RenderProfile、场景产物凭据、视觉 profile/收据/最佳候选和最终视频哈希。API Key 与端点不写入清单。恢复后的 Agent、确定性校验、Slurm 脚本和 FFmpeg 始终使用清单里捕获的 RenderProfile；视觉策略也使用清单里捕获的模型、帧数、阈值和修复上限。

每个成功场景保存 `SceneArtifact`：

- 来源 run/job、scene ID 和类名；
- 代码 SHA-256、RenderProfile SHA-256；
- run 内相对视频路径、视频 SHA-256；
- ffprobe 验证的大小、时长、分辨率和帧率。

v1 清单读取时会迁移。旧复用占位 Job、缺失 Job 或无法验证的视频会保守地重新渲染。

`resume` 在持有 `.run.lock` 后读取清单：

- 代码必须位于规范 run 路径且 SHA-256 匹配；
- 未确认终态或成功取消前不会复用/重提 Job ID；
- COMPLETED/GONE 只有产物验证成功才恢复为 rendered；
- 已完成、失败、在途场景的事件快照会补发给 TUI；
- 两个进程不能同时恢复同一 run。

`status` 只读清单；`clean` 使用同一把锁跳过活跃运行。

### 3.6 MERGING 与增量复用

VideoMerger 不扫描目录猜测输入。Orchestrator 先从每个 `SceneArtifact` 解析精确路径，再核对 scene ID、类名、代码哈希、配置哈希和视频哈希。

按 scene ID 合并：

1. 单场景直接 remux；多场景使用 FFmpeg `xfade=transition=fade` 链式交叉淡化，默认 0.5 秒；
2. 输入先统一帧率、分辨率、像素格式和 SAR；存在音频时同步使用 `acrossfade`，混合输入为无音频场景补静音；
3. 用 ffprobe 验证输入和临时输出的分辨率、帧率、音频状态与转场后时长；
4. 写临时文件，验证成功后原子替换最终输出；
5. 自定义输出默认拒绝覆盖。

默认 `ALLOW_PARTIAL_OUTPUT=false`。增量运行仍完成新代码的生成、校验和审查；只有代码哈希、RenderProfile 哈希和旧视频哈希全部一致时才复用旧 `SceneArtifact`，不会创建伪 Job ID。

### 3.7 VISUAL_EVALUATING / EVALUATING

- 代码和效率指标由确定性逻辑计算；运行对比会聚合同一指标的所有场景分数。
- 每个场景在合并前从精确 `SceneArtifact` 抽取 1–8 帧；每帧保存可信时间戳和 SHA-256，一次多模态请求联合检查数学正确性、相关性、可读性、布局与跨帧一致性。
- 响应使用关闭的 Pydantic schema；问题必须引用本次存在的帧 ID。视觉输出被当作不可信诊断，不能直接提供或执行代码。
- 低于阈值或存在 major 问题时，诊断交给主 Coder，代码重新经过 AST 校验、Reviewer 和 Slurm 渲染。每场景修复次数有界；失败时可恢复完整、哈希可验证的最佳候选。
- 抽帧、端点或结构化响应失败时记录为 `unknown`，不填充假分数，也不丢弃已经成功渲染的视频。`passed`、`warning`、`unknown` 收据都必须绑定当前视频哈希，合并前再次校验。
- 合并后再生成一份成片视觉报告，但不依据难以归因的成片问题直接改写代码。
- 自动改进只有在低分可定位到具体场景代码时才重生成；无法归因时停止循环并保留报告。

## 4. 运行目录

默认用户数据根目录是 `~/.kd1-anime/`。配置、RAG 索引和知识库源文件分别位于
`.env`、`rag/` 和 `knowledge/`；运行目录位于 `workspace/`。这些路径都可由
用户显式配置覆盖，外部配置的输出文件不会被强制搬迁。

```text
~/.kd1-anime/workspace/runs/<YYYYMMDD-HHMMSS>-<uuid8>/
├── prompt.md
├── manifest.json
├── .run.lock
├── scenes/
├── logs/
├── videos/
├── eval_frames/          # 场景与成片关键帧
├── eval_reports/         # 每轮场景报告和成片报告
├── visual_candidates/   # 可恢复候选代码
└── output_final.mp4
```

run 根目录权限为 `0700`，prompt、manifest、锁文件和生成代码为 `0600`。外部源码在 `render` 提交前复制到私有 run。若显式配置外部输出，只有最终视频写到该位置。

## 5. 配置

加载顺序：

```text
进程环境变量 > 当前目录 .env > ~/.kd1-anime/.env
```

早期版本的用户配置和默认 RAG 索引会以非破坏方式复制到新目录；旧文件保留作为
备份。大型旧 `workspace/` 不会在导入模块时自动复制，避免启动时意外消耗大量磁盘。

主要分组：LLM、独立视觉 LLM、RAG、Slurm、Manim、流水线/监控、评估和路径。RAG 默认关闭；开启后使用单独的 Embedding/Reranker URL、模型和 Key，不从其它 LLM 配置回退。`MONITOR_TIMEOUT` 只用于旧配置兼容；新配置使用 queue/run/unknown 三类 timeout，并为共享文件系统产物提供完成宽限期。

## 6. 安全边界

AST 校验是纵深防御，不是 Python 沙箱。共享或高信任要求集群应使用只读 Apptainer 镜像，并设置 `SLURM_REQUIRE_CONTAINER=true` 防止回退宿主环境。`SLURM_CONTAINER_DISABLE_NETWORK=true` 可在集群支持时增加隔离网络，但默认关闭以兼容不同 HPC 配置。未配置容器时 CLI 会明确告警。

## 7. 打包、测试与 CI

- wheel 只包含 Python 运行时模块；主机依赖由 `install.sh` 和文档管理。
- 安装器无 sudo，优先复用完整 XeLaTeX；需要时安装最小用户级 TeX Live。
- 单元测试不得调用真实 LLM、提交 Slurm 或执行生成代码。

质量门：

```bash
ruff check .
python -m compileall -q .
bash -n install.sh
pytest -q
python -m build --wheel
```

测试覆盖结构化输出、截断重试、renderer 提示词、AST 安全、AutoFix 强制复审、Slurm GONE/UNKNOWN、超时取消、ffprobe 与产物身份、Manifest 迁移/恢复、增量复用、视觉 unknown、RAG 文档切分/索引/排序/降级、批量资源配额和 FFmpeg 原子输出。
