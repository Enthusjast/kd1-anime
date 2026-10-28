# 执行合同与可恢复运行

## 状态提交

运行使用 manifest v10。源码只写入 `scenes/revisions/scene_<id>/<sha256>.py`，阶段报告写入 `artifacts/objects/<sha256>.json`。清单中的路径与摘要是执行、修复、交接及恢复的依据；固定的 `scene_N.py` 与命名报告仅为展示副本。未被清单引用的对象可以保留，本轮不自动清理。

`CommitCommand` 是状态提交入口。任务先在隔离快照上产生提案；提交时校验预期修订和完整输入身份，再合并到副本。传输期间已写入的请求及阶段计数独立合并，不能被旧 worker 快照覆盖。嵌套状态按 planning、code、execution、verification、lifecycle 分组，兼容字段必须与分组投影一致。Orchestrator 在锁内应用结构化服务结果，归一化渲染凭据及计划状态，检查运行/场景合法转移，构造独立深拷贝快照，校验跨字段不变量，然后原子替换 manifest。只有持久化成功才更新清单 revision 并发布完成事件。写盘失败会停止 worker，后续检查点与模型请求均被拒绝。

运行生命周期使用 `RunPhase`；场景阶段独立记录，原有 `State` 保留为界面投影。`StaticVerification`、`ExecutionVerification` 和 `VisualVerification` 分别记录代码接纳、执行及视觉证据。视觉 `unknown` 允许表达缺少证据，不能计为已验证，也不能删除有效视频。

`events.py` 将结构化事件适配到现有 TUI 两参数回调。成功事件附带已持久化的 manifest/scene revision；进度事件用于诊断。TUI 队列容量为 4096，合并同场景进度，溢出后用最新已提交快照重同步；回调异常只记诊断。清单负责恢复；JSONL 仅诊断，即使日志缺失也不能据此重置预算或认领作业。

## 请求与修复预算

`MAX_SCENE_LLM_REQUESTS=64` 与 `MAX_RUN_LLM_REQUESTS=512` 都必须为正整数。一次实际主/视觉模型请求，在 SDK 调用之前于运行锁内预占、写入清单；探测、兼容回退、网络重试、JSON 修复分别计数。未确认结果的请求不退还。运行级请求消耗全片限额，带 scene ID 的请求同时消耗场景限额。RAG 使用独立检索配额。

可选 `MAX_RUN_LLM_TOKENS` 和 `MAX_RUN_LLM_COST_USD` 在请求发送前按输入估算和最大输出预占；预占不因失败或未知结果退还。硬成本上限要求 `LLM_MODEL_PRICES_USD` 提供对应模型的输入/输出价格，未知价格、多模态未知输入或不受限输出均 fail closed。Clarifier 在创建 run 前也使用私有 Session 持久化预算，确认后并入同一运行账本。

两种生成模式都遵守有限 `MAX_*` 配置。新运行捕获重试限额，恢复和 retry 沿用原请求计数、修复计数及限额；增量运行拥有新的请求账本。原本支持零的配置继续以零禁用，例如 `MAX_FIX_ATTEMPTS=0` 不调用 AutoFixer，也不执行自动补丁。`relaxed` 仅将主观意见降级为 warning，确定性错误仍阻断。

代码验证与技术规划按逻辑输入摘要预占并保存内部尝试次数。恢复同一输入会继续原计数；计划、交接输入或有界审查重写轮次变化才建立新的计数。写入失败时不发送请求。

达到请求上限时不再发送新请求，未完成场景按失败与 `ALLOW_PARTIAL_OUTPUT` 策略收尾。视觉请求无法完成时记录 `unknown` 并保留视频。运行初始化后才进行计入预算的网络探测，CLI 的前置检查仅校验模型配置。

每次请求记录最终消息 hash、逻辑调用 ID、实际请求 ID、重试序号、scene revision 和可用 token 用量。provider 返回的 usage 标记 `provider_reported`，仅本地可见文本估算标记 `locally_estimated`；无证据为 `unknown`，未知成本不显示为零。并发阶段耗时按场景跨度汇总，可能超过运行墙钟时间。

## 服务边界

领域模型位于 `domain/`，旧 Agent 导入路径继续兼容导出。服务接收明确输入并返回结构化结果：

- `PlanningService` 返回场景身份已校验的 `PlannedScene`。
- `GenerationService` 负责有界生成及确定性候选检查。
- `ReviewService` 返回绑定源码哈希的 `ReviewedCode`。
- `RecoveryService` 返回本地恢复决策及 `RepairedCode`，不直接写入状态或文件。
- `VisualEvaluationService` 返回绑定源码、继承上下文、视频和帧的视觉结果。
- `RenderingService` 使用独立 `RenderJob`、轮询观测与监控接口，连接本地或 Slurm 后端。本地后端不继承 Slurm 调度器。

上述阶段通过 `StageResult` 绑定修订、输入摘要和结构化结果；`services/policies.py` 提供独立的接受、停止、候选选择和重规划路由规则。编排器负责预算、调度、接受结果和状态提交。模型计算期间不持有状态锁；结果提交前重新检查 revision 与完整输入摘要。已过期的代码、计划、审查、技术规划、修复及视觉结果不能覆盖新状态。

## 有界调度、连接及截止时间

FIFO 就绪队列按教学、结构化交接和源码依赖派发阶段任务，每场景最多一个改变状态的在途任务。固定大小的 `ContextPool` 继承运行配置与控制上下文，等待远端渲染不占用模型工作线程；计划与必要的连续性审查仍保持全片屏障。批处理继续共享模型、视觉、RAG 及渲染配额。

`RunControl` 将取消贯穿队列、配额等待、请求和退避。`LLM_TIMEOUT_TOTAL` / `VISUAL_LLM_TIMEOUT_TOTAL` 默认 0；非零时裁剪传输超时至剩余时间。取消后返回的结果不得提交，已预占请求仍计数。同步在途请求依传输超时收敛。

每次运行拥有按主模型、视觉与 RAG 端点隔离的连接池。Agent 只持有自己的调用上下文；重试不会关闭其他 Agent 正在使用的客户端，运行退出时显式关闭池。每次提交只构建一次依赖节点快照，按场景修订和输入内容缓存摘要。SQLite 快照缓存向量范数，检索用稳定并列排序的 top-k。`tools/benchmark_runtime.py` 记录 10、50、100 场景的操作次数与诊断耗时；详见 `docs/benchmarks/runtime-comparison.md`。

## 外部执行与提交恢复

Slurm、本地正式渲染及 smoke 共用 `ExecutionPolicy`。环境通过固定白名单传递，模型凭据、`PYTHONPATH`、SSH agent 及容器环境覆盖变量不会传给生成代码。保留线程/资源限制与 Apptainer `--containall --cleanenv --no-home`、当前运行目录绑定、可选 `--nv`。`SLURM_REQUIRE_CONTAINER` 对所有路径生效。

`RunConfig` 保存运行的非敏感设置，凭据在执行时解析。`ExecutionSpec` 记录镜像 SHA-256、资源/隔离策略、渲染配置及可确定的工具版本；提交前及启动时验证镜像。每个 attempt 独立记录实际环境。缺少身份、环境不一致或执行摘要不匹配时拒绝增量复用，保留已有视频。

每次提交先持久化唯一 token、源码哈希及 scene revision，再调用外部后端；取得 Job ID 后立即保存，历史尝试留在清单。保存失败立即尝试取消；取消失败、提交状态不明或查询不完整时禁止自动重提。Slurm 只按唯一尝试名称查询并认领唯一 Job，随后核对当前源码身份。没有查到作业也不能推定上次提交没有发生。

恢复必须持有运行锁、校验源码与产物身份。本地进程不凭历史 PID 重新认领；只有已确认终态的旧任务可以重启，状态不明则保留身份并阻止自动重提。v4–v9 清单只读展示与 JSON 导出，保留原版本；在 API 探测、作业访问、运行目录锁定与文件修改之前拒绝恢复、retry 和增量复用。不会执行旧格式迁移。

## 依赖与失效

依赖分为教学、结构化交接和源码上下文。清单保存生产者修订号及对应摘要；结构化消费者依据导出语义合同，源码消费者依据完整源码。源码上下文改变会使相应消费者失效；导出合同不变时保留纯结构化消费者；失效沿依赖图传播。`dependencies_complete=true` 加空依赖表示独立场景，缺少声明则保守建立前序关系。结构化流水线只绑定实际消费的交接合同。

教学合同、教学图、全局连续性样式和渲染配置也参与输入摘要；变化时取消并使已捕获这些输入的消费者失效，取消失败则停止。未消费变更输入的独立场景保留。

## 精确 API 目录与可选编译

`manim_catalog_0201.py` 是从可信安装提取的 Manim 0.20.1 元数据，包含公开符号、签名与继承信息。只有目标版本精确匹配时才判定符号存在或不存在；未知/不匹配版本返回 `unknown`。未探测容器内部版本时保留未知，不使用宿主机版本冒充。RAG 继续负责语义文档检索。生成目录不会复制上游实现：

```bash
python tools/build_api_catalog.py src/kd1_anime/manim_catalog_0201.py
```

默认 `SCENE_CODEGEN_MODE=llm`。可选 `compiler_first` 仅在计划含完整 `SceneProgram` v2 时尝试确定性编译，否则进入普通 Coder。v2 使用封闭的对象和事件类型，覆盖 Text/Tex/MathTex、坐标系、函数曲线、二维几何及有限动画。

函数表达式只接受 `x/pi/e`、有限常量、算术及单参数 `sin/cos/tan/exp/log/sqrt/abs`，通过 AST 转译，不执行 `eval`/`exec`。几何尺寸必须明确。圆/矩形等的 position 表示中心；多边形顶点与线段端点按给定坐标保留，position 表示整体平移，线段不使用隐式 buff 缩短。

事件须给定 ID、时段和对象。不能静默丢弃参数；重叠时段、不支持的数学/3D/updater/镜头、缺少尺寸、无法表示的最终导出状态均明确回退。导出对象的就地变换需要完整最终边界，否则回退；替换变换可显式导出最终对象。编译结果继续经过统一静态接纳与普通 Reviewer，编译不会绕过审查。旧模板降级路径继续标注 `degraded`，与 v2 内容保真路径分开记录。

## 固定评测

24 条独立编写的任务位于 `eval/benchmark.py`，覆盖公式、函数、几何、连续性、中文/TeX、复杂能力，以及渲染失败、预算耗尽、提交确认丢失和取消失败四种恢复场景。恢复场景需要在独立集成环境注入故障；离线基线只检查 schema、编译和确定性合同，不注入运行时故障、不请求模型、不渲染或执行生成代码：

```bash
python -m kd1_anime.eval.benchmark > /tmp/compiler-baseline.json
```

仓库中的 `docs/benchmarks/compiler-baseline.json` 保存离线基线。可以单独显式运行四条可信固定场景的实际渲染（需要 Manim、FFmpeg 和 XeLaTeX；不请求模型或提交 Slurm）：

```bash
python tools/render_trusted_scenes.py
```

该工具通过统一执行策略调用本地后端，在私有临时运行目录保存公式、函数图像、圆形与中文 TeX 的视频、哈希和元数据报告。它验证固定输入的执行结果，不评估一般教学质量。

已有真实运行可通过 `summarize_runs({task_id: Path(run_directory)})` 查看证据，或用 `compare_modes({"llm": {...}, "compiler_first": {...}})` 比较成功率、请求数、修复数与耗时。静态合同接纳和任务内容保持分别统计；只有源码 AST 与独立任务程序精确匹配时自动确认内容保持，其他实现保持未知，报告同时显示测量覆盖率。恢复检查也仅在清单中存在对应故障证据时给出结论。离线编译通过率不能代表真实渲染或教学质量。普通单元测试隔离模型、Slurm 与生成代码执行；可信固定场景的真实渲染应在单独显式触发的集成任务中运行。完成真实评测前保持编译默认关闭。

真实模型比较须显式执行，以下命令会消耗所配置模型的请求预算，本轮未运行：

```bash
python tools/compare_codegen.py --run-models --task geometry_circle --backend local --output /tmp/comparison.json
```

也可用 `--bindings /tmp/comparison.bindings.json` 只读取已有证据；不带这些参数时仅运行离线静态检查。静态合同、可信实际渲染与故障注入证据分别记录，缺失结果、视觉 unknown 和未知 token 用量不能推定为通过。
