# 配置参考

本文是 kd1-anime 当前配置的完整说明。配置字段由
src/kd1_anime/config.py 校验；模板见 ../.env.example。

## 配置文件与优先级

程序按以下优先级读取配置，越靠前优先级越高：

    进程环境变量 > 当前目录 .env > ~/.kd1-anime/.env

推荐把用户配置放在 ~/.kd1-anime/.env，并限制权限：

    mkdir -p ~/.kd1-anime
    chmod 700 ~/.kd1-anime
    chmod 600 ~/.kd1-anime/.env

安装器会保留已有用户配置，不会用模板覆盖它。早期版本的
~/.config/kd1-anime/.env 会非破坏地迁移到新目录；旧文件不会自动删除。

相对路径通常按当前工作目录解析；默认路径全部位于 ~/.kd1-anime/。运行目录由
WORKSPACE_DIR 控制，单个 run 仍会使用自己的私有子目录。

## 最小配置

完整生成至少需要主模型：

    LLM_API_KEY=your-api-key
    LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
    LLM_MODEL=your-model-name

LLM_BASE_URL 必须是带 http:// 或 https:// 的 URL，不能把用户名、密码或换行写入
URL。API Key 不会写入 manifest 或事件日志。

## 主模型

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| LLM_API_KEY | 空 | 主模型 API Key；必填，除非只执行不调用模型的命令 |
| LLM_BASE_URL | https://api.openai.com/v1 | OpenAI-compatible Base URL |
| LLM_MODEL | 空 | 主模型名；必填 |
| LLM_PLANNING_MODEL / LLM_TECHNICAL_MODEL / LLM_CODE_MODEL / LLM_REVIEW_MODEL / LLM_FIX_MODEL | 空 | 可选阶段模型；为空回退到 LLM_MODEL |
| LLM_SEND_MAX_TOKENS | true | 是否向端点发送 max_tokens |
| LLM_TEMPERATURE | 0.3 | 温度，范围 0–2 |
| LLM_PLANNING_TEMPERATURE | 0.2 | Planner/Clarifier 温度 |
| LLM_TECHNICAL_TEMPERATURE | 0.0 | TechnicalSpec 温度 |
| LLM_CODE_TEMPERATURE | 0.2 | Coder 温度 |
| LLM_REVIEW_TEMPERATURE | 0.0 | Plan/Code/Continuity Review 温度 |
| LLM_FIX_TEMPERATURE | 0.1 | AutoFix 温度 |
| LLM_MAX_TOKENS | 32768 | 全局/兼容输出上限；阶段配置优先 |
| LLM_PLANNING_MAX_TOKENS | 16384 | 规划、澄清和计划审查预算 |
| LLM_TECHNICAL_MAX_TOKENS | 16384 | TechnicalSpec 输出预算 |
| LLM_CODE_MAX_TOKENS | 24576 | Coder 和代码修复预算 |
| LLM_REVIEW_MAX_TOKENS | 8192 | 结构化 Reviewer 输出预算 |
| LLM_MAX_RETRIES | 3 | 外部请求重试次数 |
| LLM_RETRY_BASE_DELAY | 2.0 | 重试退避初始秒数 |
| LLM_TIMEOUT_CONNECT | 30 | 连接超时秒数 |
| LLM_TIMEOUT_READ | 600 | 读取超时秒数 |
| LLM_HEALTHCHECK_TIMEOUT | 15 | 启动探测超时秒数 |
| LLM_SILENT_STREAM | true | 非流式业务请求是否静默收集流式响应 |
| LLM_EMPTY_RETRY_MAX_TOKENS | 16384 | 空响应重试时使用的预算 |
| LLM_JSON_REPAIR_ATTEMPTS | 2 | JSON/Pydantic 校验失败后的修复次数 |
| LLM_PARALLEL_WORKERS | 4 | 进程内主模型并发上限 |
| LLM_DEBUG | false | 是否输出调试信息 |
| LLM_TRUST_ENV | true | 是否读取 HTTP(S)_PROXY 等代理环境变量 |
| LLM_USE_JSON_MODE | true | 是否请求 response_format=json_object |
| FAILURE_CASES_PATH | ~/.kd1-anime/diagnostics/failure_cases.sqlite3 | 脱敏失败案例库路径 |
| FAILURE_CASE_MAX_PER_CATEGORY | 100 | 每类失败案例最大保存数 |
| LLM_MAX_CONTEXT_CHARS | 120000 | Agent 输入总字符预算 |
| LLM_MAX_CODE_CONTEXT_CHARS | 60000 | 代码、继承定义和修复上下文预算 |
| LLM_MAX_REVIEW_CONTEXT_CHARS | 90000 | Reviewer 输入预算 |
| LLM_MAX_TECHNICAL_SPEC_CHARS | 30000 | TechnicalSpec 注入预算 |
| MAX_TECHNICAL_SPEC_ATTEMPTS | 3 | TechnicalSpec 编译失败后的重生成次数 |
| CODEGEN_MODE | python | 普通 Python 生成；hybrid/ir 为实验性模板化路径 |

阶段预算独立设置，可以减少推理模型把输出预算耗在分析过程而导致 JSON 或代码
截断。若模型能力或输出复杂度不同，可以只调整对应阶段。

## 独立视觉模型

视觉模型只在 ENABLE_VISUAL_EVAL=true 或显式执行视觉评估时需要。它必须支持
图片输入，不会继承主模型的 URL、Key 或模型。

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| VISUAL_LLM_API_KEY | 空 | 视觉模型 API Key |
| VISUAL_LLM_BASE_URL | 空 | 独立视觉端点 |
| VISUAL_LLM_MODEL | 空 | 多模态模型名 |
| VISUAL_LLM_SEND_MAX_TOKENS | true | 是否发送输出上限 |
| VISUAL_LLM_TEMPERATURE | 0.0 | 视觉评估温度 |
| VISUAL_LLM_MAX_TOKENS | 3000 | 视觉报告输出上限 |
| VISUAL_LLM_MAX_RETRIES | 3 | 请求重试次数 |
| VISUAL_LLM_RETRY_BASE_DELAY | 2.0 | 重试退避初始秒数 |
| VISUAL_LLM_TIMEOUT_CONNECT | 30 | 连接超时秒数 |
| VISUAL_LLM_TIMEOUT_READ | 300 | 读取超时秒数 |
| VISUAL_LLM_HEALTHCHECK_TIMEOUT | 20 | 视觉端点探测超时秒数 |
| VISUAL_LLM_JSON_REPAIR_ATTEMPTS | 1 | 结构化报告修复次数 |
| VISUAL_LLM_USE_JSON_MODE | true | 是否请求 JSON 模式 |
| VISUAL_LLM_PARALLEL_WORKERS | 2 | 视觉请求并发上限 |
| VISUAL_LLM_DEBUG | false | 是否输出视觉调试信息 |
| VISUAL_LLM_TRUST_ENV | true | 是否读取代理环境变量 |

    ENABLE_VISUAL_EVAL=true
    VISUAL_LLM_API_KEY=your-visual-api-key
    VISUAL_LLM_BASE_URL=https://your-visual-endpoint/v1
    VISUAL_LLM_MODEL=your-multimodal-model

EVAL_VISUAL_MODEL 是旧版模型名兼容别名，新配置请使用 VISUAL_LLM_MODEL。普通
流水线中的视觉网络故障会将结果记为 unknown 并继续；缺少视觉配置或显式
evaluate --visual 时，程序会在视觉流程前报错。

## RAG

RAG 默认关闭。开启后必须配置独立 Embedding、Reranker 和未过期的本地索引；
任何一个服务都不会继承其它模型的配置。

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| RAG_ENABLED | false | 是否启用生成流程中的检索 |
| RAG_INDEX_PATH | ~/.kd1-anime/rag/index.sqlite3 | SQLite 索引路径 |
| RAG_DOCS_DIR | ~/.kd1-anime/knowledge/docs | 文档源目录 |
| RAG_EXAMPLES_DIR | ~/.kd1-anime/knowledge/examples | 示例源目录 |
| RAG_RECIPES_DIR | ~/.kd1-anime/knowledge/recipes | 内置及本地匿名 Recipe 目录 |
| RAG_EMBEDDING_API_KEY | 空 | Embedding API Key |
| RAG_EMBEDDING_BASE_URL | 空 | OpenAI-compatible Embedding 端点 |
| RAG_EMBEDDING_MODEL | 空 | Embedding 模型名 |
| RAG_EMBEDDING_TIMEOUT | 60 | Embedding 请求超时秒数 |
| RAG_EMBEDDING_BATCH_SIZE | 32 | 建索引时单批文本数 |
| RAG_RERANK_API_KEY | 空 | Reranker API Key |
| RAG_RERANK_BASE_URL | 空 | Cohere-compatible Rerank 端点 |
| RAG_RERANK_MODEL | 空 | Reranker 模型名 |
| RAG_RERANK_TIMEOUT | 60 | Reranker 请求超时秒数 |
| RAG_TRUST_ENV | true | 是否读取代理环境变量 |
| RAG_TOP_K | 8 | 向量初排候选数 |
| RAG_RERANK_TOP_N | 4 | 重排后注入候选数 |
| RAG_MAX_CONTEXT_CHARS | 12000 | 单次注入检索上下文上限 |
| RAG_CHUNK_SIZE | 1800 | 文本分块大小 |
| RAG_CHUNK_OVERLAP | 200 | 分块重叠大小，必须小于 chunk size |
| RAG_PARALLEL_WORKERS | 2 | 进程内 RAG 请求并发上限 |

TechnicalSpec 使用 v2 语义动作合同：`introduce`、`update`、`remove`、`camera` 和 `hold`。
Coder 在每个 `self.play` 前写 `# KD1_ANIMATION_EVENT: <event_id>`，静态检查器据此校验
对象状态；具体使用哪一种 Manim 动画由 Coder 自主选择。无法识别的动画调用只记录 warning，
但 dry-run 会对包含这类调用的场景强制执行低质量 frame+短视频 Smoke Render。

索引构建需要 Embedding 服务；完整生成还需要 Reranker。源文件、Embedding 模型、
分块参数变化后，执行：

    kd1-anime rag index
    kd1-anime rag index --rebuild
    kd1-anime rag status
    kd1-anime doctor --probe-rag

## Slurm 与容器

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| SLURM_PARTITION | 空 | Slurm 分区 |
| SLURM_ACCOUNT | 空 | Slurm account |
| SLURM_QOS | 空 | Slurm QoS |
| SLURM_CONDA_ENV | manim_env | 远端 conda 环境名 |
| SLURM_CONDA_BASE | 空 | conda 根目录；为空时动态探测 |
| SLURM_TIME_LIMIT | 01:00:00 | 作业时限，格式为 [days-]HH:MM:SS |
| SLURM_CPUS_PER_TASK | 4 | 每个作业 CPU 数 |
| SLURM_MEM_GB | 空 | 可选内存约束，如 32G |
| SLURM_GPU_TYPE | 空 | OpenGL 作业的 GPU 类型 |
| SLURM_GPU_COUNT | 1 | OpenGL GPU 数 |
| AUTO_RESOURCE_ESTIMATION | true | 是否按场景复杂度自动增加资源（只向上调整） |
| SLURM_MAX_IN_FLIGHT | 0 | 最大在途场景作业数；0 表示不限制 |
| SLURM_SUBMIT_RETRIES | 3 | 明确 sbatch 失败的重试次数 |
| SLURM_SUBMIT_RETRY_DELAY | 2.0 | sbatch 重试退避秒数 |
| SLURM_CONTAINER_IMAGE | 空 | Apptainer 镜像路径 |
| SLURM_REQUIRE_CONTAINER | false | 为 true 时没有镜像会拒绝生成 |
| SLURM_CONTAINER_DISABLE_NETWORK | false | 支持时在容器内禁用网络 |

Cairo 不申请 GPU；只有 MANIM_RENDERER=opengl 时才使用 GPU 配置。所有 Slurm
标识和时限都会经过单行格式校验。

## Manim、Smoke Render 与合并

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| MANIM_RENDERER | cairo | cairo 或 opengl |
| MANIM_QUALITY | h | l、m、h、p 或 k |
| MANIM_PIXEL_WIDTH | 1920 | 输出宽度，必须为偶数 |
| MANIM_PIXEL_HEIGHT | 1080 | 输出高度，必须为偶数 |
| MANIM_FRAME_RATE | 60 | 输出帧率 |
| MANIM_OPENGL_PLATFORM | egl | OpenGL 后端：egl 或 glx |
| RENDER_BACKEND | slurm | 正式渲染后端：slurm 或 local |
| LOCAL_RENDER_MAX_IN_FLIGHT | 1 | 本地正式渲染最大并发数 |
| LOCAL_RENDER_TIMEOUT | 3600 | 单个本地正式渲染超时秒数 |
| LOCAL_RENDER_MEMORY_MB | 16384 | 本地正式渲染地址空间上限 |
| SMOKE_RENDER_ENABLED | true | 正式 Slurm 渲染前执行轻量探针 |
| SMOKE_RENDER_MODE | both | 预检模式：frame、video 或 both |
| SMOKE_RENDER_QUALITY | l | Smoke Render 质量：l 或 m |
| SMOKE_RENDER_TIMEOUT | 180 | 远端 Smoke Render 超时秒数 |
| SMOKE_RENDER_SHORT_ANIMATIONS | 3 | 短视频预检最多执行的前几个动画事件 |
| ADAPTIVE_SMOKE_RENDER | true | 是否按场景风险选择 Smoke 强度 |
| LOCAL_SMOKE_RENDER_ENABLED | false | 是否在本地做额外运行时预检 |
| LOCAL_SMOKE_RENDER_MODE | frame | 本地预检模式：frame、video 或 both |
| LOCAL_SMOKE_RENDER_QUALITY | l | 本地预检质量 |
| LOCAL_SMOKE_RENDER_TIMEOUT | 180 | 本地预检超时秒数 |
| LOCAL_SMOKE_RENDER_MEMORY_MB | 4096 | 本地预检地址空间上限 |
| LOCAL_SMOKE_RENDER_SHORT_ANIMATIONS | 3 | 本地短视频预检最多执行的动画事件 |
| ALLOW_PARTIAL_OUTPUT | false | 是否允许缺少场景时合并 |
| OVERWRITE_OUTPUT | false | 是否允许覆盖自定义输出 |
| TRANSITION_TYPE | fade | 当前支持的场景转场 |
| TRANSITION_DURATION | 0.5 | xfade 转场秒数 |
| MERGE_VIDEO_CODEC | libx264 | 最终视频编码器，也支持 libx265 |
| MERGE_VIDEO_PRESET | medium | FFmpeg preset |
| MERGE_VIDEO_CRF | 18 | 视频质量参数，范围 0–51 |
| MERGE_AUDIO_SAMPLE_RATE | 48000 | 合并音频采样率 |
| MERGE_AUDIO_CHANNEL_LAYOUT | stereo | 合并音频声道布局 |

MANIM_RENDERER 决定 Cairo/OpenGL；MANIM_OPENGL_PLATFORM 只决定 OpenGL 上下文
后端。无显示的 headless 节点通常使用 egl，需要显示服务的环境才使用 glx。

## 流水线、审查与监控

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| MAX_REVIEW_ROUNDS | 8 | 单场景代码审查/重写轮数 |
| MAX_LOW_RISK_REVIEW_ROUNDS | 2 | 低风险场景的代码审查轮数；确定性检查不跳过 |
| MAX_PLAN_REVIEW_ROUNDS | 2 | 单场景计划审查轮数 |
| MAX_PLAN_REPLAN_ATTEMPTS | 3 | 计划反馈后的 Planner 总重调用次数 |
| MAX_CONTINUITY_FIX_ROUNDS | 2 | 连续性局部重规划次数；耗尽后 warning 放行 |
| CONTINUITY_CONTEXT_MODE | minimal | 跨场景代码上下文范围：minimal、full 或 stateless |
| SKIP_REVIEW | false | 是否跳过语义代码审查；确定性校验仍保留 |
| SAFE_FALLBACK_ENABLED | true | 高风险几何失败后是否切换保守方案 |
| MAX_IDENTICAL_REVIEW_ATTEMPTS | 2 | 相同代码/反馈重复次数上限 |
| MAX_STAGNANT_ATTEMPTS | 2 | 渲染修复无进展后切换确定性回退的次数 |
| MAX_FIX_ATTEMPTS | 8 | 渲染失败后的最大代码修复次数 |
| MAX_INFRA_RETRIES | 2 | 基础设施故障重排队次数 |
| MAX_FIX_IDENTICAL_ERRORS | 3 | 相同渲染错误指纹的放弃阈值 |
| MAX_CLARIFY_ROUNDS | 12 | Clarifier 最大轮数 |
| MAX_SCENES | 12 | 单次规划最大场景数 |
| MAX_PROMPT_CHARS | 50000 | 用户需求字符上限 |
| MAX_CLARIFY_CONTEXT_CHARS | 40000 | 多轮澄清上下文上限 |
| MAX_LOG_CHARS | 30000 | AutoFixer 接收的日志上限 |
| CODE_VALIDATION_ATTEMPTS | 3 | 代码校验失败后的重生成次数 |
| MAX_CODE_CANDIDATES_LOW/MEDIUM/HIGH | 1/2/3 | 按场景风险允许的不同代码实现策略数 |
| MONITOR_POLL_INTERVAL | 10 | Slurm 轮询间隔秒数 |
| MONITOR_QUEUE_TIMEOUT | 3600 | 排队超时秒数 |
| MONITOR_RUN_TIMEOUT | 3600 | 运行超时秒数 |
| MONITOR_MAX_UNKNOWN | 5 | 连续 UNKNOWN 次数阈值 |
| MONITOR_UNKNOWN_TIMEOUT | 300 | UNKNOWN 状态最短持续时间 |
| MONITOR_ARTIFACT_GRACE | 60 | 作业结束后等待共享文件系统的秒数 |
| LOG_TAIL_LINES | 80 | 读取日志尾部行数 |

MONITOR_TIMEOUT 是旧配置兼容项。新配置应分别设置 queue、run 和 unknown 相关
参数。UNKNOWN 表示控制面无法确认作业状态，不等于作业已经失败；达到条件后
程序会先尝试取消，取消失败时禁止自动重复提交。

## 评估与路径

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| ENABLE_AUTO_EVAL | false | 合并后启用确定性代码/效率评估循环 |
| EVAL_THRESHOLD | 3.5 | 自动评估通过阈值 |
| MAX_EVAL_ROUNDS | 2 | 自动评估-改进循环次数 |
| ENABLE_VISUAL_EVAL | false | 启用逐场景视觉质量门和成片报告 |
| VISUAL_EVAL_FRAME_COUNT | 6 | 每个场景关键帧数，范围 1–8 |
| VISUAL_EVAL_THRESHOLD | 3.5 | 视觉评分通过阈值，范围 1–5 |
| MAX_VISUAL_FIX_ATTEMPTS | 2 | 视觉诊断触发的场景修复次数 |
| WORKSPACE_DIR | ~/.kd1-anime/workspace | 持久化运行目录根路径 |
| OUTPUT_FILE | output_final.mp4 | 默认最终输出；默认值放到当前 run 目录 |
| SCENES_DIR | 派生路径 | 旧调用兼容项 |
| LOGS_DIR | 派生路径 | 旧调用兼容项 |
| VIDEOS_DIR | 派生路径 | 旧调用兼容项 |

## 推荐配置组合

### 本地/无 Slurm

    RENDER_BACKEND=local
    MANIM_RENDERER=cairo
    kd1-anime generate --file prompt.md --backend local

本地正式渲染在当前进程的前台执行，`render` 命令必须使用 `--wait`；本地任务不会把
PID 写入 manifest，恢复时会重新启动未完成任务，不会认领旧 PID。若只想验证计划和代码，
使用 `--dry-run`，它不会提交正式本地任务。

### HPC：Cairo

    MANIM_RENDERER=cairo
    SLURM_PARTITION=your-partition
    SLURM_ACCOUNT=your-account
    SLURM_QOS=your-qos

### HPC：OpenGL headless

    MANIM_RENDERER=opengl
    MANIM_OPENGL_PLATFORM=egl
    SLURM_GPU_TYPE=your-gpu
    SLURM_GPU_COUNT=1

OpenGL 代码不能使用 self.camera.frame 或 MovingCameraScene 的 Cairo 运镜 API；
切换 renderer 后建议执行 kd1-anime doctor --probe。

### 开启 RAG

    RAG_ENABLED=true
    RAG_DOCS_DIR=~/.kd1-anime/knowledge/docs
    RAG_EXAMPLES_DIR=~/.kd1-anime/knowledge/examples
    # 填写 RAG_EMBEDDING_* 和 RAG_RERANK_* 后：
    kd1-anime rag index
    kd1-anime doctor --probe-rag

### 开启视觉评估

    ENABLE_VISUAL_EVAL=true
    VISUAL_LLM_API_KEY=your-visual-api-key
    VISUAL_LLM_BASE_URL=https://your-visual-endpoint/v1
    VISUAL_LLM_MODEL=your-multimodal-model
    kd1-anime doctor --probe-visual-llm

## 如何确认配置生效

    kd1-anime doctor
    kd1-anime rag status
    kd1-anime doctor --probe-llm
    kd1-anime doctor --probe-visual-llm
    kd1-anime doctor --probe-rag
    kd1-anime version

诊断输出会显示模型名和 URL，但不会显示 API Key。若怀疑读取了错误的 .env，可
暂时使用进程环境变量覆盖并重新运行 doctor；不要在日志或 issue 中粘贴完整配置文件。
