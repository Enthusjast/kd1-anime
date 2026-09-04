# kd1-anime

`kd1-anime` 是一个面向数学教学的 Manim Community Edition 代码生成与渲染流水线。它把自然语言需求转换为分镜和 Manim Scene，先检查数学与可实现性，再生成、审查、渲染、修复并合并视频。

项目使用显式有限状态机、Pydantic 数据模型和 OpenAI-compatible API，不依赖 LangChain、AutoGen 或 LangGraph。当前 Python 包版本为 `0.4.0`，默认锁定 Manim Community Edition `0.20.1`。

确定性校验和高置信度证据负责阻断真正的核心错误；风格建议、一般节奏意见和证据不足的模型判断会记录为 warning，不会触发无意义的重写循环。

## 目录

- [kd1-anime](#kd1-anime)
  - [目录](#目录)
  - [适用场景与前提](#适用场景与前提)
  - [快速开始](#快速开始)
    - [1. 安装](#1-安装)
    - [2. 配置主模型](#2-配置主模型)
    - [3. 检查环境](#3-检查环境)
    - [4. 生成视频](#4-生成视频)
  - [生成流程](#生成流程)
    - [计划与代码审查的职责](#计划与代码审查的职责)
    - [场景粒度与并行](#场景粒度与并行)
  - [常用命令](#常用命令)
    - [需求、规划和生成](#需求规划和生成)
    - [单 Scene 渲染](#单-scene-渲染)
    - [查询、恢复和清理](#查询恢复和清理)
    - [环境和模型诊断](#环境和模型诊断)
    - [缓存](#缓存)
  - [配置](#配置)
    - [主模型、视觉模型和 RAG 服务](#主模型视觉模型和-rag-服务)
  - [RAG 知识检索](#rag-知识检索)
  - [视觉评估](#视觉评估)
  - [运行产物与恢复](#运行产物与恢复)
  - [增量渲染与批量处理](#增量渲染与批量处理)
    - [增量渲染](#增量渲染)
    - [批量处理](#批量处理)
  - [渲染器、转场与视频合并](#渲染器转场与视频合并)
  - [安全边界](#安全边界)
  - [开发与验证](#开发与验证)
  - [文档](#文档)
  - [技术栈](#技术栈)
  - [许可证](#许可证)

## 适用场景与前提

| 使用方式 | 必需条件 | 是否提交 Slurm |
| --- | --- | --- |
| `generate --dry-run` | 主模型、Python 依赖；启用 RAG 时还需 RAG 服务和索引 | 否 |
| 完整生成 | 主模型、Manim、XeLaTeX、FFmpeg、Slurm | 是 |
| `render scene.py` | Manim、XeLaTeX、FFmpeg、Slurm | 是（除非使用全局 `--dry-run`） |
| 视觉评估 | 独立的多模态视觉模型 | 不一定，取决于评估的运行 |
| RAG | 独立 Embedding、Reranker 和本地索引 | 否，索引保存在本地 |

完整渲染还需要目标集群提供可用的 `sbatch`、`squeue`、`sacct` 和 `scancel`。没有 Slurm 时，可以用 `--dry-run` 验证规划、技术合同、代码生成和代码审查流程；它不会提交作业，也不会执行生成代码。

## 快速开始

### 1. 安装

在 Ubuntu/HPC 上可以只下载并运行安装脚本。脚本默认不使用 sudo，也不会把完整源码 clone 到当前目录或主目录：

```bash
curl -fsSL https://raw.githubusercontent.com/Enthusjast/kd1-anime/main/install.sh \
  -o /tmp/kd1-anime-install.sh \
  && bash /tmp/kd1-anime-install.sh
```

安装器会创建或复用 `manim_env`，安装 Manim CE `0.20.1`、FFmpeg、CJK 字体和 Manim 所需的最小 XeLaTeX 依赖，并将 Manim 文档和示例放入 `~/.kd1-anime/knowledge/`。

交互式终端中，安装器最后会启动模型配置向导，依次配置主模型、视觉模型、Embedding 和 Reranker。非交互环境默认跳过向导：

```bash
# 显式启动向导
KD1_ANIME_CONFIGURE_MODE=interactive bash /tmp/kd1-anime-install.sh

# 显式跳过向导
KD1_ANIME_CONFIGURE_MODE=never bash /tmp/kd1-anime-install.sh
```

安装到指定版本时可以固定 tag；发布或生产环境建议同时校验 SHA-256：

```bash
export KD1_ANIME_REF=v0.4.0
export KD1_ANIME_ARCHIVE_SHA256=<github-zip-sha256>
# 可选：校验 TeX Live 安装器
export KD1_ANIME_TEXLIVE_INSTALLER_SHA256=<install-tl-sha256>
# 设置后，上面两个摘要都必须提供
# export KD1_ANIME_REQUIRE_CHECKSUM=1
bash /tmp/kd1-anime-install.sh
```

如果已经在源码目录中开发，使用：

```bash
git clone https://github.com/Enthusjast/kd1-anime.git
cd kd1-anime
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate manim_env
python -m pip install -e '.[dev]'
```

仅安装 Python 包不会自动安装 Manim、XeLaTeX、FFmpeg 或 Slurm；这些原生依赖由 `install.sh` 或系统环境负责。

### 2. 配置主模型

安装器会创建 `~/.kd1-anime/.env`。也可以复制模板后编辑：

```bash
cp .env.example ~/.kd1-anime/.env
chmod 600 ~/.kd1-anime/.env
$EDITOR ~/.kd1-anime/.env
```

最少需要配置一个主模型：

```dotenv
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
LLM_MODEL=your-model-name
```

配置优先级为：

```text
进程环境变量 > 当前目录 .env > ~/.kd1-anime/.env
```

API Key 不会写入运行清单、事件日志或缓存键。不要把 `.env` 提交到 Git。

### 3. 检查环境

先做本地依赖检查，再按需探测网络服务：

```bash
# 依赖、配置和安全策略检查；默认不发送网络请求
kd1-anime doctor

# 额外执行 FFmpeg、XeLaTeX、CJK/MathTex 和当前 renderer 的本地探针
kd1-anime doctor --probe

# 发送最小请求探测主模型
kd1-anime doctor --probe-llm
```

启动 chat、`plan` 或 `generate` 前，程序会自动探测主模型；探测失败会在进入 Agent 流程前退出。`status`、`version`、`logs`、`clean` 等诊断命令不会自动发起业务请求。

### 4. 生成视频

```bash
# 交互式澄清需求
kd1-anime
# 等价写法
kd1-anime chat

# 跳过澄清，直接使用给定需求
kd1-anime generate "解释欧拉公式的几何意义"

# 没有 Slurm 时验证完整的计划与代码生成流程
kd1-anime generate "解释特征值的几何意义" --dry-run

# 需要显式执行本地低质量 Smoke/Frame Canary 时再打开；会执行生成代码
kd1-anime generate "解释特征值的几何意义" --dry-run --smoke
```

完成后，终端会显示最终视频路径和 run ID。默认最终视频位于该 run 的私有目录；需要固定到外部路径时使用 `--output`。

## 生成流程

完整流水线如下：

```text
INIT
  → 主模型/RAG 预检
  → PLANNING（概要、教学合同、数学断言图）
  → DETAILING（各场景分镜并行生成）
  → PLAN_REVIEWING（确定性编译 + 计划审查 + 连续性审查）
  → CODING（TechnicalSpec → Coder，按场景顺序交接）
  → REVIEWING（AST/生命周期 + 代码语义审查）
  → DISPATCHING / MONITORING（场景级 Slurm 并行）
  → FIXING → REVIEWING → …
  → VISUAL_EVALUATING（可选）
  → MERGING
  → EVALUATING（可选的代码/效率评估循环）
  → DONE
```

### 计划与代码审查的职责

- **Plan Review** 检查数学断言、等式关系、定义域、几何方案、时间线和元素交接是否正确。失败只回到 Planner，不会让 Coder 反复修补错误计划。
- **Technical Planner** 把分镜编译为对象、动画事件、布局、LaTeX 和最终导出清单。确定性编译失败时只有限重试。
- **Code Review** 检查已确认计划的 Manim 实现、数学展示、API、生命周期、布局、安全和场景交接。代码变化后必须重新审查。
- **审查分级**：确定性校验或带源码/合同证据的高置信度核心错误才是 hard blocker；可唯一匹配的局部替换先自动修复；风格建议、一般节奏和不确定的“可能问题”作为 warning 放行。
- **Render Fix** 只处理渲染日志暴露的代码问题；环境、Slurm、字体和显示服务错误不会盲目交给模型重写。
- **Continuity Review** 只处理跨场景边界。达到 `MAX_CONTINUITY_FIX_ROUNDS` 后会记录 warning 并沿用当时的可验证计划继续，不会因为连续性审查耗尽而阻断整条流水线。
- **可靠性回退**：确定性 Scene IR、稳定场景模板、精确 traceback 证据和修复停滞检测共同限制重复生成；复杂场景可按风险尝试有限的备选实现。

### 场景粒度与并行

场景不是清单条目的机械切分单位。若用户要求在同一画布中同时展示一组对象，且这些对象需要共同变化或最终对比，Planner 应将其合并为一个场景；只有镜头、布局或叙事弧线确实独立时才拆分。

分镜生成可以并行；代码生成按 Scene ID 顺序执行，以便把上一场景实际导出的 Mobject 定义交给下一场景。所有代码通过审查后，场景渲染可以并行提交到 Slurm。`SLURM_MAX_IN_FLIGHT` 可限制同时排队/运行的场景数量。

## 常用命令

### 需求、规划和生成

```bash
# 交互模式；回车提交，Shift+Enter/Ctrl+Enter 换行
kd1-anime chat

# 从文件读取长 prompt，避免 shell 转义和多行粘贴问题
kd1-anime generate --file prompt.md --dry-run

# 只生成规划；默认执行计划审查和连续性审查
kd1-anime plan "解释傅里叶级数"

# 只预览未经审查的规划
kd1-anime plan "解释傅里叶级数" --no-review

# 导出结构化计划；计划文件可交给 generate --plan
kd1-anime plan "解释傅里叶级数" --output fourier-plan.json
kd1-anime generate --plan fourier-plan.json --dry-run

# 计划审查后暂停人工确认；非交互环境视为批准
kd1-anime generate "解释勾股定理" --approve-plan
```

`plan --no-review` 只适合查看模型草案；从计划文件继续生成时，仍会重新执行确定性编译、计划审查和连续性审查。

### 单 Scene 渲染

```bash
# 校验用户提供的 Scene 后提交 Slurm，并等待结果
kd1-anime render scene.py --class MyScene --wait

# 只提交，立即返回 run ID；稍后使用 resume
kd1-anime render scene.py --class MyScene
kd1-anime resume <run-id>

# 全局 dry-run 只校验代码，不提交作业
kd1-anime --dry-run render scene.py --class MyScene
```

`render` 是直接渲染模式，不调用 Planner、Technical Planner、Coder 或 Reviewer；它只执行确定性代码校验、渲染监控和合并。

### 查询、恢复和清理

```bash
# 最近运行
kd1-anime status

# 某次运行的详细状态；--json 便于脚本处理
kd1-anime status <run-id>
kd1-anime status <run-id> --json

# 查看渲染日志尾部
kd1-anime logs <run-id> --scene-id 2 --lines 120
kd1-anime logs <run-id> --scene-id 2 --stderr

# 恢复中断或失败运行；不会自动扫描历史运行
kd1-anime resume <run-id>

# 只重试某个失败场景
kd1-anime retry <run-id> --scene-id 2

# 查看离线成功率、审查/修复次数和失败分类；不会调用 LLM 或 Slurm
kd1-anime stats
kd1-anime stats <run-id> --json

# 清理 30 天前的已结束运行
kd1-anime clean --older-than 30d --yes
```

启动程序不会自动弹出历史可恢复运行。请先执行 `status` 找到 run ID，再显式执行 `resume`。恢复要求当前可写的 manifest schema 为 v7；v4–v6 可以只读查看，但不能安全继续修改。

### 环境和模型诊断

```bash
kd1-anime version
kd1-anime doctor
kd1-anime doctor --deep
kd1-anime doctor --probe
kd1-anime doctor --probe-llm
kd1-anime doctor --probe-visual-llm
kd1-anime doctor --probe-rag
kd1-anime doctor --security-strict

# 检查 JSON 模式和基本请求
kd1-anime test-llm
kd1-anime test-llm --no-json-mode --verbose
```

`doctor` 默认只做本地配置检查；`--probe-*` 才会发送相应服务请求。视觉探针会发送一张最小图片消息，RAG 探针会分别请求 Embedding 和 Reranker。

### 缓存

```bash
kd1-anime cache status
kd1-anime cache clear --yes
```

缓存只保存完整的非流式业务响应和脱敏调用统计，不缓存交互式流式响应，也不保存 API Key。调试 prompt 变化或怀疑复用了旧响应时，可以先查看或清理缓存。

## 配置

完整配置参考见 [`docs/configuration.md`](docs/configuration.md)，模板见 [`.env.example`](.env.example)。常用配置如下：

| 配置项 | 默认值 | 作用 |
| --- | ---: | --- |
| `LLM_BASE_URL` / `LLM_MODEL` | API 地址 / 空 | 主模型端点和模型名；必须配置 |
| `LLM_HEALTHCHECK_TIMEOUT` | `15` | 启动前主模型探测超时（秒） |
| `LLM_PLANNING_MAX_TOKENS` | `16384` | 计划和澄清阶段输出预算 |
| `LLM_CODE_MAX_TOKENS` | `24576` | 代码生成阶段输出预算 |
| `LLM_REVIEW_MAX_TOKENS` | `8192` | 结构化审查输出预算 |
| `LLM_*_MODEL` | 空 | 可选阶段模型路由；为空回退到 `LLM_MODEL` |
| `LLM_TRUST_ENV` | `true` | 是否读取 `HTTP(S)_PROXY` 等代理环境变量 |
| `LLM_CACHE_ENABLED` | `true` | 是否启用本地非流式响应缓存 |
| `LLM_MAX_CONTEXT_CHARS` | `120000` | Agent 输入总预算；低优先级区块会先裁剪 |
| `MANIM_RENDERER` | `cairo` | `cairo` 使用 CPU；`opengl` 需要 GPU/图形上下文 |
| `MANIM_QUALITY` | `h` | Manim 质量级别：`l/m/h/p/k` |
| `MANIM_PIXEL_WIDTH` / `HEIGHT` | `1920/1080` | 输出分辨率 |
| `MANIM_FRAME_RATE` | `60` | 输出帧率 |
| `MANIM_OPENGL_PLATFORM` | `egl` | OpenGL 上下文后端；无显示的 HPC 通常使用 `egl` |
| `SMOKE_RENDER_ENABLED` | `true` | 正式 Slurm 渲染前执行同 renderer 的轻量探针 |
| `SMOKE_RENDER_MODE` | `both` | 预检模式：frame、短视频或两者 |
| `LOCAL_SMOKE_RENDER_ENABLED` | `false` | 是否在本地编码后执行额外运行时预检 |
| `LOCAL_SMOKE_RENDER_MODE` | `frame` | 本地预检模式：最后一帧、MP4 或两者 |
| `CODEGEN_MODE` | `python` | 普通 Python 生成；`hybrid/ir` 为实验性模板化路径 |
| `MAX_CODE_CANDIDATES_LOW/MEDIUM/HIGH` | `1/2/3` | 按场景风险允许的备选实现策略数 |
| `MAX_SCENES` | `12` | 单次规划的最大场景数 |
| `MAX_PLAN_REVIEW_ROUNDS` | `2` | 单场景计划审查/重规划轮数 |
| `MAX_PLAN_REPLAN_ATTEMPTS` | `3` | 计划反馈后的 Planner 总重调用次数 |
| `MAX_CONTINUITY_FIX_ROUNDS` | `2` | 连续性局部重规划次数；耗尽后 warning 放行 |
| `MAX_REVIEW_ROUNDS` | `5` | 单场景代码审查/重写轮数 |
| `MAX_LOW_RISK_REVIEW_ROUNDS` | `2` | 低风险场景的审查轮数；确定性检查始终执行 |
| `MAX_STAGNANT_ATTEMPTS` | `2` | 渲染修复无进展后切换 IR/安全模板的次数 |
| `MAX_FIX_ATTEMPTS` | `5` | 渲染失败后的代码修复次数 |
| `SAFE_FALLBACK_ENABLED` | `true` | 高风险几何方案失败后是否切换保守方案 |
| `SLURM_MAX_IN_FLIGHT` | `0` | 最大在途场景作业数；`0` 表示不额外限制 |
| `AUTO_RESOURCE_ESTIMATION` | `true` | 是否按场景复杂度只向上增加 Slurm 资源 |
| `MONITOR_QUEUE_TIMEOUT` / `RUN_TIMEOUT` | `3600/3600` | 排队/运行超时（秒） |
| `MONITOR_UNKNOWN_TIMEOUT` | `300` | 控制面不可查询时的最短等待时间 |
| `MONITOR_ARTIFACT_GRACE` | `60` | Slurm 完成后等待共享文件系统同步产物的时间 |
| `ALLOW_PARTIAL_OUTPUT` | `false` | 是否允许缺少场景时合并部分视频 |
| `TRANSITION_DURATION` | `0.5` | 相邻场景 `xfade` 转场秒数 |
| `ENABLE_VISUAL_EVAL` | `false` | 是否启用独立视觉质量门 |
| `VISUAL_EVAL_THRESHOLD` | `3.5` | 视觉评分通过阈值（1–5） |
| `MAX_VISUAL_FIX_ATTEMPTS` | `2` | 视觉诊断触发的最大修复次数 |
| `RAG_ENABLED` | `false` | 是否启用本地知识检索 |
| `WORKSPACE_DIR` | `~/.kd1-anime/workspace` | 运行目录根路径 |

### 主模型、视觉模型和 RAG 服务

视觉评估必须使用独立的多模态端点，不会继承主模型的 URL、Key 或模型：

```dotenv
ENABLE_VISUAL_EVAL=true
VISUAL_LLM_API_KEY=your-visual-api-key
VISUAL_LLM_BASE_URL=https://your-visual-endpoint/v1
VISUAL_LLM_MODEL=your-multimodal-model
```

RAG 的 Embedding 和 Reranker 同样完全独立。Embedding 使用 OpenAI-compatible `/embeddings`，Reranker 使用 Cohere-compatible `/rerank`：

```dotenv
RAG_ENABLED=true
RAG_INDEX_PATH=~/.kd1-anime/rag/index.sqlite3
RAG_DOCS_DIR=~/.kd1-anime/knowledge/docs
RAG_EXAMPLES_DIR=~/.kd1-anime/knowledge/examples
RAG_RECIPES_DIR=~/.kd1-anime/knowledge/recipes
RAG_EMBEDDING_API_KEY=your-embedding-key
RAG_EMBEDDING_BASE_URL=https://your-embedding-endpoint/v1
RAG_EMBEDDING_MODEL=your-embedding-model
RAG_RERANK_API_KEY=your-rerank-key
RAG_RERANK_BASE_URL=https://your-reranker-endpoint/v1
RAG_RERANK_MODEL=your-reranker-model
```

启用 RAG 后，生成入口会检查索引存在且未过期，并在开始 Agent 调用前探测两个服务；缺配置、索引过期或启动探测失败会直接退出。运行中的单次检索异常则记录为 `degraded`，并尽可能继续使用无 RAG 的流程。

## RAG 知识检索

默认知识库由安装器放入：

```text
~/.kd1-anime/knowledge/
├── docs/manim-0.20.1/       # Markdown/reStructuredText 文档
├── examples/manim-0.20.1/   # Python 示例
└── recipes/manim-0.20.1/    # 带 renderer/风险标签的可信 API 配方
```

索引只读取 `.md`、`.rst` 和 `.py`，并将源目录、源文件哈希、分块参数和 Embedding 模型写入 SQLite 索引。Recipe 会额外带有 ManimCE、版本、renderer、主题和风险标签，供 Coder 选择相关 API 配方。修改知识库文件、分块参数或 Embedding 模型后，旧索引会被标记为过期：

```bash
# 使用配置中的默认目录建立或复用索引
kd1-anime rag index

# 强制重新计算所有 Embedding
kd1-anime rag index --rebuild

# 查看索引和服务配置（不联网）
kd1-anime rag status

# 手动检索并查看片段
kd1-anime rag search "TransformMatchingTex usage" --top-k 8

# 真实探测两个服务
kd1-anime doctor --probe-rag
```

索引构建本身需要 Embedding 服务；完整生成还需要 Reranker。检索结果会以不可信参考资料注入 Planner、Technical Planner、Coder 和 AutoFixer，并在运行清单中保存查询、索引和分块哈希收据，不会直接执行检索内容。

## 视觉评估

视觉评估是可选质量门，使用独立的多模态模型检查关键帧中的数学正确性、相关性、可读性、布局和跨帧/跨场景一致性：

```bash
# 配置 ENABLE_VISUAL_EVAL、VISUAL_LLM_* 后运行完整流水线
kd1-anime generate "解释矩阵变换的几何意义"

# 对已经完成的运行重新执行视觉评估
kd1-anime evaluate <run-id> --visual

# 只评估一个场景
kd1-anime evaluate <run-id> --scene-id 2 --visual

# 输出 JSON 或保存报告
kd1-anime evaluate <run-id> --visual --json --output visual-report.json
```

每个场景会从精确的已验证视频中抽取 1–8 帧，并在多场景运行中额外抽取真实的相邻结尾/开头帧。视觉报告、关键帧和修复候选均绑定当前代码、继承上下文、视频和帧哈希。

- 数学/叙事问题回到 Planner；元素交接问题回到 Continuity；布局、可读性和遮挡问题交给 Coder。
- 视觉修复仍会重新经过代码校验、代码审查和渲染，且次数有上限。
- 视觉服务或抽帧失败记为 `unknown`，不伪造低分，也不会删除已经成功的渲染产物。
- 普通流水线中的视觉端点网络故障可以安全降级；显式 `evaluate --visual` 要求端点可用。

不带 `--visual` 时，`evaluate` 仍可执行确定性的代码/效率评估；支持 `--code`、`--code-file`、`--image` 和 `--compare`，具体选项可执行 `kd1-anime evaluate --help` 查看。

## 运行产物与恢复

默认所有持久化数据位于 `~/.kd1-anime/`：

```text
~/.kd1-anime/
├── .env                         # 私有配置（0600）
├── .env.example                 # 配置模板
├── knowledge/                   # Manim 文档和示例
├── rag/index.sqlite3            # 本地知识索引
├── cache/llm.sqlite3            # LLM 完整响应缓存
├── diagnostics/failure_cases.sqlite3 # 脱敏渲染失败案例
└── workspace/
    ├── eval_results/            # 独立 evaluate 命令的报告
    └── runs/<run-id>/
        ├── prompt.md            # 需求文件；不是 prompt.txt
        ├── manifest.json        # 当前为 schema v7
        ├── events.jsonl         # 脱敏事件轨迹
        ├── scenes/              # Python Scene 与 sbatch 脚本
        ├── logs/                # stdout/stderr
        ├── videos/              # 当前 run 的媒体目录
        ├── artifacts/            # 计划、合同、审查和账本快照
        ├── run_report.json       # 结构化运行统计与失败诊断
        ├── eval_frames/          # 视觉评估关键帧
        ├── eval_reports/         # 场景/成片视觉报告
        ├── visual_candidates/    # 视觉修复候选
        └── output_final.mp4      # 默认最终视频
```

每个 run 使用独立目录和运行锁。manifest 会原子写入，并保存阶段、代码 SHA-256、精确 Slurm Job、Render/Merge Profile、视频哈希和 ffprobe 元数据。恢复时不会用共享目录扫描猜测视频，也不会复用不匹配的旧产物。

运行 ID 可通过 `status` 获取；中断后显式恢复：

```bash
kd1-anime status
kd1-anime resume 20260831-120000-1234abcd
```

旧版本的 `~/.config/kd1-anime/.env` 会非破坏地迁移到 `~/.kd1-anime/.env`；旧文件不会删除。旧项目目录中的相对 `workspace/` 不会自动搬迁，以避免启动时复制大型视频。

## 增量渲染与批量处理

### 增量渲染

增量渲染仍会执行新运行的规划、代码生成和审查，只在以下身份全部一致时复用旧场景视频：代码哈希、Render Profile 哈希、旧视频哈希、场景 ID/类名以及环境验证结果。

```bash
kd1-anime generate \
  "解释欧拉公式的几何意义" \
  --incremental 20260801-120000-1234abcd
```

复用的视频会复制到新 run 的私有目录，不会创建伪造的 Job ID；合并阶段还会再次验证所有身份。

### 批量处理

输入文件可以是每行一个 prompt 的文本文件，也可以是包含 `prompts` 数组的 JSON：

```bash
cat > prompts.txt <<'EOF'
解释欧拉公式的几何意义
展示傅里叶级数的几何意义
可视化特征值和特征向量
EOF

kd1-anime batch prompts.txt --max-parallel 3

cat > prompts.json <<'EOF'
{
  "prompts": [
    "解释欧拉公式的几何意义",
    "展示傅里叶级数的几何意义"
  ]
}
EOF

kd1-anime batch prompts.json --dry-run
```

`--max-parallel` 限制项目级并行数；所有项目还共享进程级 `LLM_PARALLEL_WORKERS`、`RAG_PARALLEL_WORKERS`、`VISUAL_LLM_PARALLEL_WORKERS` 和 `SLURM_MAX_IN_FLIGHT` 配额。使用 `--output-dir` 时，输出文件会按任务编号写入该目录；重复目标和不允许覆盖的文件会在执行前被拒绝。

## 渲染器、转场与视频合并

- `MANIM_RENDERER=cairo` 是默认 CPU 渲染器；`MANIM_RENDERER=opengl` 需要有效 GPU。只有 OpenGL 模式才会申请 `SLURM_GPU_TYPE`。
- `MANIM_OPENGL_PLATFORM` 只决定 PyOpenGL 上下文后端：`egl` 适合无显示的 headless 节点，`glx` 需要可用显示服务。它不等同于选择 Cairo/OpenGL 渲染器。
- OpenGL 不支持 `self.camera.frame`/`MovingCameraScene` 这类 Cairo 运镜 API；3D 场景应使用专用相机 API。遇到 `OpenGLCamera ... frame` 错误，应修改代码或切换 Cairo，而不是只重复提交任务。
- 正式渲染前默认执行 import-only、frame 和风险自适应的短视频 Smoke Render；本地预检需要显式设置 `LOCAL_SMOKE_RENDER_ENABLED=true` 或使用 `--smoke`，且 dry-run 默认永不执行生成代码。
- 复杂场景默认只向上调整 Slurm CPU、内存和时间资源；可通过 `AUTO_RESOURCE_ESTIMATION=false` 恢复固定资源。
- AutoFix 会优先使用唯一可匹配补丁，并保存最多 3 个经过验证的代码候选；连续无进展时优先回滚到可信版本。
- 多场景默认使用 FFmpeg `xfade=transition=fade`，转场时长为 `TRANSITION_DURATION=0.5` 秒；有音频时同步使用 `acrossfade`。
- 合并写入临时文件，ffprobe 验证通过后才原子替换最终输出。默认拒绝覆盖自定义输出，使用 `--force` 或配置 `OVERWRITE_OUTPUT=true` 才允许覆盖。

## 安全边界

LLM 生成的 Python 是不可信输入。AST 校验会限制导入、动态执行、文件/网络访问、危险属性、Scene 结构、Manim 生命周期和 renderer API，但 AST 校验不是 Python 沙箱。

共享集群或处理不可信需求时，建议使用只读 Apptainer 镜像：

```dotenv
SLURM_CONTAINER_IMAGE=/path/to/manim.sif
SLURM_REQUIRE_CONTAINER=true
# 集群支持无特权网络命名空间且验证通过后再开启
SLURM_CONTAINER_DISABLE_NETWORK=true
```

容器作业使用 `--containall --cleanenv --no-home`，只绑定当前 run；OpenGL 会额外使用 `--nv`。使用 `kd1-anime doctor --security-strict` 检查是否启用 fail-closed 策略。

## 开发与验证

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate manim_env
python -m pip install -e '.[dev]'

ruff check .
ruff format --check .
python -m compileall -q .
bash -n install.sh
pytest -q
python -m build --sdist --wheel
```

测试不得调用真实 LLM、提交 Slurm 或执行生成代码。CI 会在 Python 3.10、3.11 和 3.12 上运行静态检查、编译检查、Shell 语法检查、测试、sdist/wheel 构建和安装后 CLI 检查；`Integration` workflow 额外验证真实 Manim、XeLaTeX、CJK、FFmpeg 联动。

## 文档

- [`ARCHITECTURE.md`](ARCHITECTURE.md)：FSM、Agent 分工、合同、恢复、渲染和安全边界。
- [`docs/configuration.md`](docs/configuration.md)：完整配置项参考和推荐配置组合。
- [`docs/troubleshooting.md`](docs/troubleshooting.md)：LLM、RAG、Slurm、OpenGL、XeLaTeX 和恢复问题排查。
- [`CONTRIBUTING.md`](CONTRIBUTING.md)：开发环境、测试、提交和 Pull Request 约定。
- [`CHANGELOG.md`](CHANGELOG.md)：版本变更记录。

## 技术栈

Python 3.10+ · OpenAI-compatible API · Pydantic · Rich · Typer · prompt_toolkit · Manim CE 0.20.1 · FFmpeg · XeLaTeX · Slurm · 可选 Apptainer

## 许可证

MIT
