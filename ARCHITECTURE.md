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
       ├── agents/plan_compiler.py  确定性计划编译（时间线/等式/几何/生命周期）
       ├── agents/plan_reviewer.py 计划正确性、数学与可实现性审查
       ├── agents/technical_planner.py TechnicalSpec 生成与确定性编译
       ├── agents/lifecycle.py     AST 动画生命周期校验
       ├── agents/prompt_context.py 有界 Prompt 区块构造
       ├── agents/continuity.py    全片连续性审查与局部重规划
       ├── agents/coder.py         ManimCE 代码生成/重写
       ├── agents/reviewer.py      结构化语义审查
       ├── agents/validator.py     AST 确定性校验
       ├── agents/auto_fixer.py    根据渲染日志修复代码
       ├── agents/render_context.py renderer 能力与对象生命周期约束
       ├── cluster/slurm.py        sbatch、状态查询、产物验证、超时取消
       ├── rendering.py            RenderProfile/MergeProfile、ffprobe、SceneArtifact
       ├── resources.py            跨批量项目的进程级 LLM/RAG/Slurm 配额
       ├── media/merger.py         精确输入列表、FFmpeg xfade/acrossfade 原子合并
       ├── rag/                    SQLite 索引、独立 Embedding/Reranker 检索
       ├── eval/                   代码/效率评估与独立多模态视觉质量门
       ├── llm_cache.py            SQLite LLM 响应缓存与安全限额
       ├── agents/state_ledger.py  场景边界语义账本与渲染证据
       ├── security.py             脱敏和 JSON-safe 诊断序列化
       └── run_store.py            Manifest v6、原子检查点、运行锁
```

`agents/base.py` 封装 OpenAI-compatible client、重试、静默流式传输、JSON/代码提取和 Pydantic 校验。普通文本/代码的非空 `finish_reason=length` 响应不会被消费；计划审查、连续性审查和代码审查等严格结构化响应允许先交给 JSON/Pydantic 校验，只有完整结构才会被接受，持续截断时仍抛出明确错误。

## 3. 执行模型

```text
全局：INIT → 主 LLM/RAG 可用性探测 → PLANNING

分镜屏障：全片 PlanningDraft → 所有 Scene 并行 DETAILING → 计划编译/审查 → 全片连续性审查

顺序代码屏障：
Scene 1 技术设计 → CODING → 代码 REVIEWING → Scene 2 技术设计 → CODING → 代码 REVIEWING → …

渲染阶段：各 Scene 并行 DISPATCHING → MONITORING
                                      ▲              │
                                      └── FIXING ◀────┘

任一场景渲染完成：即时 VISUAL_EVALUATING → (低分回到 CODING，失效后继交接)
全部线程结束：补评剩余场景 → MERGING
                                      → 成片视觉报告
                                      → (EVALUATING → 可定位场景回到 CODING) → DONE
```

FSM 枚举同时用于清单检查点和 TUI 阶段提示。概要阶段一次性建立 LessonSpec 和 TeachingGraph；每个 Scene 必须先完成 Detail、确定性计划编译、计划审查和全片连续性审查，确认数学关系、定义域、教学依赖、几何可行性和时间线正确，再进入编码。分镜仍然并行，但编码/代码审查按场景顺序执行：Scene N 先由 Technical Planner 生成结构化 TechnicalSpec，确定性编译通过后才允许 Coder 工作；代码通过生命周期校验和 Reviewer 后，提取其连续性导出区，才允许 Scene N+1 编码。这样 Coder 收到的是上一场景真实生成的最终 Mobject 定义，并且必须遵守明确的对象生命周期，而不是仅凭 Planner 描述猜测状态。所有代码就绪后，Slurm 渲染继续并行；每个 worker 使用独立 Agent 实例并关闭流式终端输出，也不读取共享 stdin。

LLM 调用受 `LLM_PARALLEL_WORKERS` 信号量限制；RAG 请求受独立的 `RAG_PARALLEL_WORKERS` 信号量限制；Slurm 提交受 `SLURM_MAX_IN_FLIGHT` 限制。批量模式中的多个 Orchestrator 共享同一个 `ResourceCoordinator`，不会把每项目配额相乘。
CLI 在进入 chat、规划、生成或恢复需要 Agent 的运行前，会用短超时发送一次主 LLM 请求；启用 RAG 时还会确认索引存在且未过期，并探测 Embedding 和 Reranker。探测失败直接退出，不把明显的配置/网络问题拖到 Clarifier 或 Planner 阶段才暴露。视觉评估使用完全独立的 Key、URL、模型、超时和并发配置；批处理中的多个 Orchestrator 共享进程级视觉并发配额。配置缺失会在启动前失败，网络探测暂时失败时生成流水线降级为 `unknown`；显式 `evaluate --visual` 则失败退出。`status`、`render`、`clean`、已完成运行恢复和纯代码评估不依赖这些探测。

`ERROR` 是失败检查点。任何未处理异常或不允许的部分输出都会触发失败；用户中断时会尝试取消仍在运行的 Job。顶层检查点使用显式转移表；并发 worker 或恢复入口产生的非典型回退只写入 `fsm_warnings`，不把可恢复的运行误判为失败。

RAG 索引使用 SQLite 保存文本分块、元数据和 Embedding BLOB。Markdown 和 reStructuredText 按标题/段落切分，Python 按顶层定义切分；索引构建通过临时数据库原子替换。运行时先做本地余弦初排，再调用独立 Reranker；服务故障只记录 `degraded` 并继续原有流水线。Planner、Technical Planner、Coder 和 AutoFixer 收到的检索内容均标记为不可信资料，且每次注入都保存查询、索引和分块哈希收据。

### 3.1 PLANNING / DETAILING

Planner 使用分层结构化输出：

1. `plan_draft()` 一次性生成 `PlanningDraft`：LessonSpec 固定学习目标、实体、数学断言、定义域和时长；TeachingGraph 固定断言依赖与场景分配；`SceneOutline` 按最小必要视觉单元生成。概要按返回顺序规范化 scene ID 为 `1..N`。同一画布中逐个出现、保留并对比的对象属于同一个场景；当用户明确要求同屏/整体展示而模型仍按对象拆分时，Planner 会将概要确定性合并为一个场景。只有用户明确要求多场景，或镜头/布局/叙事弧线确实独立时才拆分。
2. `plan_continuity_bible()` 在分镜并行前固定全片背景、调色板、字体、布局、数学符号、持续对象、镜头语言和转场规则，并写入运行清单。
3. 每个 worker 的 `plan_detail()` 接收原始需求、教学合同、全部概要、相邻概要和 continuity bible，生成视觉设计、镜头、动画流、关键时刻、计算说明以及 opening/closing state、结构化 `inherited_elements` / `elements_to_remove` / `new_elements` 和转场合同。
4. 所有 Detail 完成后先运行 Plan Compiler，检查场景 ID、断言覆盖/依赖、时间线覆盖、可解析等式、多边形鞋带面积、画布边界和元素生命周期。随后逐场景执行 Plan Review，检查数学正确性、几何可实现性和交接合同；问题只回到 Planner 重规划，单份计划的审查轮数受 `MAX_PLAN_REVIEW_ROUNDS` 限制，Planner 总重调用次数另受 `MAX_PLAN_REPLAN_ATTEMPTS` 限制，未通过的计划不会进入 Coder。计划/问题指纹重复时冻结计划并停止空转。
5. Plan Review 通过后执行全片连续性审查；冲突只重规划未进入编码的相关场景，受 `MAX_CONTINUITY_FIX_ROUNDS` 限制。高风险几何方案在计划审查或代码审查耗尽后，可切换为保守的面积/等式教学方案。

Pydantic 模型拒绝未知字段并限制字符串、列表和场景数量。ScenePlan 还包含 timeline、math_claims、geometry_specs 和 handoff 四类结构化合同；无法确定的数学表达式不会被编译器擅自判定为正确。用户需求被明确标记为不可信数据，不能改变系统规则。
`GlobalVisualState` 固定全片颜色、字体、字号、线宽、布局锚点和镜头语言；每个 `ScenePlan` 都携带同一份只读配置。`VisualElementState` 为跨场景对象分配稳定的 `element_id`。`LessonSpec` 是数学事实唯一来源，`TeachingGraph` 是依赖顺序唯一来源；Detail、TechnicalSpec、Coder 和 Reviewer 不得静默增加核心断言。

### 3.2 CODING / REVIEWING

TechnicalSpec 是 CODING 内部的强制前置阶段，不改变顶层 FSM 的兼容状态集合。它把
ScenePlan 的对象声明映射为变量名、构造器、动画事件、Transform 语义、LaTeX 分段、布局
约束和导出清单；编译器会模拟 active 状态，阻断对已退出对象的继续使用。它和输入哈希、
规范化 JSON 一起写入 `artifacts/scene_<id>_technical_spec.json` 与 `manifest.json`，计划或
继承代码变化后自动失效，恢复时校验哈希后才会复用。

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

若 TechnicalSpec 编译失败，先在有限次数内重新生成技术合同，不会把语义错误转嫁给 Coder。合同通过后，生成结果先经过 `validate_manim_code()` 和 AST 生命周期校验；失败反馈交回 Coder，最多尝试 `CODE_VALIDATION_ATTEMPTS` 次。Coder 必须在代码中提供 `KD1_CONTINUITY_EXPORT_BEGIN/END` 区，区内允许纯 Mobject 定义以及作用于区内对象的白名单样式/布局调用；复合 Mobject 所需的坐标数组和子 Mobject 可以作为 helper 一并放入带有 `element_id` 的分组，但不能包含动画或外部依赖。Orchestrator 通过 AST 安全提取并保存为下一场景的 `[Inherited Elements Code]`，同时更新运行级 ElementManifest。清单记录 element_id、变量名、类型、依赖、语义状态、源代码及哈希；Coder 只接收当前场景需要的最小 entries。Reviewer 再检查数学、LaTeX、Manim API、动画生命周期、布局、安全、分镜符合度和元素交接，并要求 major finding 提供代码中可验证的证据；若 evidence 协议不通过，最多重试一次，仍不明确则停止而不是接受无依据的阻断。结构化输出只允许：

- valid / `info`：通过；
- `minor`：至少一条可精确唯一匹配的查找替换；
- `major`：带具体反馈并回到 Coder 重写。

Plan Review 和 Code Review 使用独立状态与计数。Plan Review 不通过只允许 Planner 重规划，不会生成代码；Code Review 只检查已确认计划对应的 Manim 实现，受 `MAX_REVIEW_ROUNDS` 限制。任何代码变化都会把 `reviewed` 重置为 false。AutoFix 输出也必须重新进入 Code Review；major 反馈仍回到 CODING，绝不直接提交。

连续性审查结果和警告也保存到 `manifest.json`；resume 会复用已保存的 continuity bible，不会因为重启而重新生成一套风格规范。如果上游代码改变，尚未提交渲染的下游场景会清除旧交接代码并按顺序重新编码；恢复旧清单时会优先重新提取导出区，提取失败不会静默复用下游状态。

当 `LOCAL_SMOKE_RENDER_ENABLED=true` 且不是 dry-run 时，代码在进入 Reviewer 前会以低质量、
同 renderer 的本地命令运行一次；若配置了 Apptainer，则沿用 containall/cleanenv/no-home、当前
run bind 和 OpenGL 的 GPU/平台参数。成功结果只写入不含敏感信息的阶段快照，失败直接阻断该场景，
不会把本地 Smoke Render 产物当作正式视频。

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

Job 只有在最终 MP4 通过 ffprobe、目标分辨率和帧率验证后才算成功。每次提交都使用独立的 `attempt_<token>` 媒体目录；定位会递归适配 Manim 嵌套层级、排除 `partial_movie_files`、符号链接和早于本次提交的文件，不会把上一次修复的 MP4 当成当前产物。正式作业完成后还会记录计算节点的 Python/Manim/FFmpeg/XeLaTeX/renderer 指纹；不一致时标记 warning，增量复用会拒绝带 warning 的旧产物。

### 3.4 FIXING

失败场景只读取精确 Job 的 stderr 尾部，并受 `LOG_TAIL_LINES` 和 `MAX_LOG_CHARS` 限制。环境、conda、容器、Slurm、显示服务和字体错误不会交给 LLM 重写业务代码。

其余错误交给 AutoFixer，结果再次通过 AST 校验；不通过时由 Coder 根据校验反馈和原始错误重写。修复后的代码强制复审。修复次数和连续相同错误次数都有上限。

### 3.5 持久化与恢复

Orchestrator 在关键阶段和每次 Slurm 提交后更新 `manifest.json`：写同目录临时文件、文件 `fsync`、`os.replace()`、目录 `fsync`。schema v6 包含单调 revision、LessonSpec、TeachingGraph、StateLedger、场景 phase、代码哈希、审查/修复次数、精确 Job、RenderProfile、MergeProfile、场景产物凭据、视觉 profile/收据/最佳候选、ElementManifest 和最终视频哈希。计划编译、计划审查、代码审查和 Smoke 结果也以私有阶段快照保存。API Key 与端点不写入清单；`events.jsonl` 只保存脱敏后的事件轨迹。恢复后的 Agent、确定性校验、Slurm 脚本和 FFmpeg 始终使用清单里捕获的 RenderProfile/MergeProfile；视觉策略也使用清单里捕获的模型、帧数、阈值和修复上限。

每个成功场景保存 `SceneArtifact`：

- 来源 run/job、scene ID 和类名；
- 代码 SHA-256、RenderProfile SHA-256；
- run 内相对视频路径、视频 SHA-256；
- ffprobe 验证的大小、时长、分辨率和帧率。

v6 清单只接受当前教学合同、StateLedger、结构化计划、ElementManifest、阶段状态和最终合并配置；v4/v5 仍可只读查看但不能安全恢复或写回，v1-v3 不再猜测迁移，恢复旧版会明确失败并要求重新生成。LLM 非流式完整响应默认写入用户私有 SQLite 缓存；缓存键包含端点、模型、提示词、模式、代理策略和生成参数，不含 API Key，条目数受 LLM_CACHE_MAX_ENTRIES 限制。

`resume` 在持有 `.run.lock` 后读取清单：

- 代码必须位于规范 run 路径且 SHA-256 匹配；
- 未确认终态或成功取消前不会复用/重提 Job ID；
- COMPLETED/GONE 只有产物验证成功才恢复为 rendered；
- 已完成、失败、在途场景的事件快照会补发给 TUI；
- 两个进程不能同时恢复同一 run。
直接 `render --wait` 创建的运行会持久化 `direct_render` 标记；这类运行在等待和恢复时都
跳过所有 Planner、Technical Planner、Coder 和 Reviewer 调用，只执行渲染监控与合并。

`status` 只读清单；`clean` 使用同一把锁跳过活跃运行。只有显式使用
`--include-running` 时才会处理陈旧的 running 清单，并且删除前会先取消其中已知的
Slurm Job；任一 Job 取消失败都会保留运行目录。

### 3.6 MERGING 与增量复用

VideoMerger 不扫描目录猜测输入。Orchestrator 先从每个 `SceneArtifact` 解析精确路径，再核对 scene ID、类名、代码哈希、渲染配置（含 Manim/FFmpeg/XeLaTeX 版本）和视频哈希。

按 scene ID 合并：

1. 单场景直接 remux；多场景使用 FFmpeg `xfade=transition=fade` 链式交叉淡化，默认 0.5 秒；
2. 输入先统一帧率、分辨率、像素格式和 SAR；存在音频时同步使用 `acrossfade`，混合输入为无音频场景补静音；
3. 用 ffprobe 验证输入和临时输出的分辨率、帧率、音频状态与转场后时长；
4. 写临时文件，验证成功后原子替换最终输出；
5. 自定义输出默认拒绝覆盖。

默认 `ALLOW_PARTIAL_OUTPUT=false`。增量运行仍完成新代码的生成、校验和审查；只有代码哈希、RenderProfile 哈希和旧视频哈希全部一致时才复用旧 `SceneArtifact`，不会创建伪 Job ID。

### 3.7 VISUAL_EVALUATING / EVALUATING

- 代码和效率指标由确定性逻辑计算；运行对比会聚合同一指标的所有场景分数。
- 每个场景在合并前从精确 `SceneArtifact` 抽取 1–8 帧；抽样优先覆盖开场、首个数学状态、中段、结论和结束状态。多场景合并前另抽取真实相邻场景的 `boundary_end`/`boundary_start` 帧，检查交接对象是否丢失。每帧保存可信时间戳、语义 role 和 SHA-256，一次多模态请求联合检查数学正确性、相关性、可读性、布局与跨帧一致性；场景产物完成后立即启动该检查。
- 响应使用关闭的 Pydantic schema；问题必须引用本次存在的帧 ID。视觉输出被当作不可信诊断，不能直接提供或执行代码。
- 低于阈值或存在 major 问题时，按 `repair_target` 路由：数学/叙事问题回到 Planner，元素交接问题回到 Continuity，布局/可读性问题才交给主 Coder。所有代码重新经过 AST 校验、Reviewer 和 Slurm 渲染；每场景修复次数有界，失败时可恢复完整、哈希可验证的最佳候选。
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
├── artifacts/            # 教学合同、计划编译、TechnicalSpec、审查和状态账本
├── events.jsonl          # 脱敏的阶段/检查点事件轨迹
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

测试覆盖结构化输出、教学合同/依赖图、截断重试、renderer 提示词、AST 安全、辅助函数生命周期、AutoFix 强制复审、Slurm GONE/UNKNOWN、超时取消、ffprobe 与产物身份、Manifest v6 与 v4/v5 只读恢复、增量复用、视觉边界/unknown/路由、RAG 文档切分/索引/排序/降级、批量资源配额、事件脱敏和 FFmpeg 原子输出。手动或定时运行 `Integration` workflow 可在真实 Ubuntu 环境验证 Cairo、XeLaTeX、CJK、MathTex 和 FFmpeg。
