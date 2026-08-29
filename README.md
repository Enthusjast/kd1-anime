# kd1-anime

`kd1-anime` 是一个 AI Agent 驱动的 Manim Community Edition 数学动画生成器。用户用自然语言描述目标，程序会澄清需求、规划场景、生成并审查代码、提交 Slurm 并行渲染、自动修复失败场景，并用 FFmpeg 合并最终视频。

## 主要特性

- **对话式终端交互**：先追问受众、时长、内容重点和视觉风格，再开始生成。
- **分层规划与计划审查**：先生成全片概要，再为每个场景生成详细导演分镜，并在写代码前审查数学正确性与可实现性。
- **全片教学合同**：概要阶段同时固定 LessonSpec、数学断言依赖图和最小视觉单元；后续分镜、技术计划与代码只能实现已声明的断言。
- **确定性计划编译**：对场景编号、时间线覆盖、可解析等式、多边形面积、画布边界和元素生命周期先做本地检查，再调用计划审查模型。
- **最小场景粒度**：同一画布中的逐步绘制、叠加和对比默认合并为一个场景，避免按函数或清单条目机械拆分。
- **全片视觉状态**：Planner 固定全局颜色、字体、字号、线宽与布局，并为每个场景声明继承、移除和新增元素。
- **代码级场景交接**：计划审查通过后才进入编码；编码/代码审查按顺序执行，上一场景的最终 Mobject 定义会安全注入下一场景。
- **最小连续性清单**：运行中维护带元素身份、变量名、依赖、语义状态、源代码和哈希的 ElementManifest，只把当前场景需要的交接定义注入 Coder。
- **状态账本**：额外记录每个场景的开场/收场元素、数学状态、代码哈希、视频哈希和相邻边界帧，恢复与视觉审查都绑定到同一份状态证据。
- **技术实现合同**：每个新场景在 Coder 前先生成结构化 TechnicalSpec，明确对象、生命周期、动画源/目标、布局、LaTeX 和最终导出清单；确定性编译失败会阻断编码。
- **生命周期校验**：不执行生成代码即可用 AST 检查 Create/FadeOut/Transform、`self.add/remove/clear`、OpenGL 相机 API 和最终交接对象，修复后仍必须复审。
- **多 Agent 流水线**：Planner → 计划审查 → Technical Planner → Coder → 代码审查 → AutoFixer，不依赖 LangChain 等重型框架。
- **分阶段并行**：场景分镜并行生成；计划审查和代码交接按场景顺序执行；所有已通过代码审查的场景仍可并行提交 Slurm 渲染。
- **确定性安全校验**：在 LLM 审查之外，使用 Python AST 检查语法、Scene 结构、导入和危险调用。
- **运行隔离**：每次运行写入 `~/.kd1-anime/workspace/runs/<run-id>/` 下的独立目录，避免并发运行和旧产物互相污染。
- **可验证产物**：每个 MP4 都绑定代码哈希、渲染配置哈希、视频哈希和 ffprobe 元数据，避免把旧文件误判为本次结果。
- **中断恢复**：版本化、原子 `manifest.json` 保存阶段、代码哈希、Slurm Job ID 和产物凭据，可查询并恢复中断运行。
- **可选容器隔离**：可用 Apptainer 执行 LLM 生成的 Manim 代码。
- **可恢复渲染**：监控 Slurm 状态、区分排队/运行超时、失败后读取日志并自动修复。
- **Smoke Render**：正式 Slurm 渲染前在同一 renderer 和节点资源中执行轻量运行时检查；也可显式开启本地 Smoke Render，在编码后提前发现 OpenGL、XeLaTeX、Manim API 和运行时错误。
- **有界 Prompt**：结构化合同和代码区不会被静默截断，低优先级的 RAG、历史说明和重复上下文会按预算裁剪，避免模型因上下文过长产生不完整输出。
- **平滑转场**：多场景使用 FFmpeg `xfade` 淡入淡出，默认 0.5 秒；有音频时同步 `acrossfade`。
- **通用 LLM 接口**：通过 `.env` 配置任意 OpenAI-compatible API，不绑定 DeepSeek 或其他特定厂商。
- **启动前 API 探测**：进入会话或 LLM 流水线前探测主 LLM；启用 RAG 时同时探测 Embedding 和 Reranker。配置、网络或模型不可用时立即退出。视觉端点单独探测，暂时不可用时安全降级为 `unknown`。
- **独立视觉质量门**：可为每个已渲染场景抽取带时间戳和哈希的关键帧，用单独的多模态 LLM 检查数学正确性、相关性、可读性、布局和跨帧一致性；主 Planner/Coder 端点不会被替换。
- **有界视觉修复**：低分场景把纯诊断反馈交回 Coder，重新经过校验、审查和渲染；达到上限时保留更好的可验证版本，视觉端点故障则记为 `unknown` 并继续。
- **即时视觉门**：场景完成渲染后立即评估，不必等待整批场景结束；低分上游场景会停止并重建后继交接。关键帧包含开场、首个数学状态、转场边界、中段、结论和结束状态。
- **本地 LLM 缓存**：非流式完整响应默认缓存到用户目录，缓存键包含模型、端点、提示词和生成参数但不含 API Key；可关闭或限制条目数。
- **可选知识检索**：使用本地 SQLite 索引、独立 Embedding 和 Reranker 服务，为 Planner、Coder 和 AutoFixer 提供受限、可审计的 Manim 文档与示例上下文。

## 一行安装（Ubuntu / HPC，无 sudo）

只下载并运行安装脚本，不会在当前目录或主目录 `git clone` 完整源码：

```bash
curl -fsSL https://raw.githubusercontent.com/Enthusjast/kd1-anime/main/install.sh \
  -o /tmp/kd1-anime-install.sh \
  && bash /tmp/kd1-anime-install.sh
```

远程运行时，脚本会从 GitHub ZIP 源码归档临时构建并安装 Python 包，临时目录会在退出时清理。默认安装 `main`；发布版本可在运行前设置 `KD1_ANIME_REF=vX.Y.Z` 固定到指定 tag。

发布时建议同时公布该 tag 源码 ZIP 的 SHA-256，用户可执行强校验安装：

```bash
export KD1_ANIME_REF=v0.4.0
export KD1_ANIME_ARCHIVE_SHA256=<release-zip-sha256>
# 可选：同时固定 TeX Live 安装器摘要；设置为 1 后两项摘要都必须提供
export KD1_ANIME_TEXLIVE_INSTALLER_SHA256=<install-tl-unx-tar-gz-sha256>
# export KD1_ANIME_REQUIRE_CHECKSUM=1
bash /tmp/kd1-anime-install.sh
```

摘要不匹配或 ref 含路径遍历字符时，安装器会在调用 pip 前终止。设置
`KD1_ANIME_REQUIRE_CHECKSUM=1` 可让远程源码归档和 TeX Live 安装器都强制要求 SHA-256。

安装器会全自动完成：

1. 加载 `python3.12/3.12` 和 `miniconda/py312` module（若系统提供 module）。
2. 创建或复用 `manim_env` conda 环境。
3. 安装 Manim Community Edition 0.20.1、FFmpeg 和 Noto CJK 字体。
4. 依次检查 PATH、`/usr/local/texlive` 和 `~/texlive` 中已有的 XeLaTeX；完整环境直接复用且不调用 `tlmgr`。
5. 仅当现有 TeX Live 缺失或无法无 sudo 补齐依赖时，才从 USTC CTAN 镜像安装最小用户目录版到 `~/texlive/<release>/`。
6. 只安装 Manim/XeLaTeX 所需包及 `ctex`、`xeCJK`、`fontspec`，不安装完整 TeX Live scheme/collection。
7. 安装 `kd1-anime` 命令，不保留远程源码目录。
8. 将 Manim Community Edition 0.20.1 文档和示例程序解压到 `~/.kd1-anime/knowledge/`。
9. 在 `~/.local/bin` 安装 `kd1-anime` / `manim-env` 包装器，并写入 conda 激活钩子、shell 函数和用户级配置模板。

在交互式终端中，安装结束前会自动启动模型配置向导，依次配置主模型、视觉模型、
Embedding 和 Reranker；非交互安装会自动跳过向导。需要显式开启向导时可执行：

```bash
KD1_ANIME_CONFIGURE_MODE=interactive bash install.sh
```

向导会在 Embedding 配置完成后询问是否立即建立 RAG 索引；也可以稍后手动执行
`kd1-anime rag index`。选择关闭视觉、Embedding 或 Reranker 时会保留已有凭据，
但不会启用对应功能。

Coder 生成的 `Tex`/`MathTex` 统一使用 `xelatex` 和 `.xdv`，并加载 `ctex`；普通中文文字使用 Noto CJK 字体和 Manim `Text`（Pango）。

安装完成后无需手动 `source` RC 文件或激活 conda，即可直接运行：

```bash
# 启动程序
kd1-anime

# 可选：进入已激活 manim_env 的交互 shell
manim-env

# 配置 OpenAI-compatible API
$EDITOR ~/.kd1-anime/.env
```

### 用户数据目录

除非通过配置显式覆盖，程序生成的持久化文件统一位于 `~/.kd1-anime/`：

```text
~/.kd1-anime/
├── .env                         # 用户配置
├── .env.example                 # 配置模板
├── knowledge/
│   ├── docs/                    # 默认 Manim 文档源文件
│   └── examples/                # 默认 Manim 示例源文件
├── rag/index.sqlite3            # 本地 RAG 索引
├── cache/llm.sqlite3             # 非流式 LLM 响应缓存（可关闭）
└── workspace/runs/<run-id>/     # 场景代码、日志、视频和评估报告
```

旧版本的 `~/.config/kd1-anime/.env` 会在首次加载配置时复制到新位置并更新旧的
默认路径；旧文件不会被删除。旧的 RAG 索引也会在首次使用 RAG 时复制到新位置。
为避免意外复制大型视频，旧项目目录中的相对 `workspace/` 不会自动搬迁；如需保留
旧运行，请先将它移动到 `~/.kd1-anime/workspace/`，或显式设置 `WORKSPACE_DIR`。

> `install.sh` 针对具有 Environment Modules、Miniconda 和 Slurm 的 Ubuntu/HPC 环境。可通过 `KD1_ANIME_CONDA_BASE` 和 `KD1_ANIME_ENV_NAME` 覆盖 conda 路径与环境名。

## 开发安装

```bash
git clone https://github.com/Enthusjast/kd1-anime.git
cd kd1-anime
bash install.sh
```

在源码目录运行时，安装器会执行 editable install。若系统已经具备 Manim/TeX/FFmpeg，也可仅安装 Python 包：

```bash
python -m pip install -e '.[dev]'
```

## LLM 配置

配置加载优先级为：

```text
系统环境变量 > 当前目录 .env > ~/.kd1-anime/.env
```

最少需要填写：

```dotenv
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
LLM_MODEL=your-model-name
```

若启用 `ENABLE_VISUAL_EVAL=true`，还必须单独配置支持 `image_url` 的多模态端点；它不会继承主 LLM 的 Key、URL 或模型：

```dotenv
VISUAL_LLM_API_KEY=your-visual-api-key
VISUAL_LLM_BASE_URL=https://your-visual-endpoint/v1
VISUAL_LLM_MODEL=your-multimodal-model
```

如需使用知识检索，可单独配置 Embedding 和 Reranker 服务。RAG 不会继承主 LLM
或视觉 LLM 的任何凭据；Embedding 使用 OpenAI-compatible `/embeddings` 接口，
Reranker 使用 Cohere-compatible `/rerank` 接口：

```dotenv
RAG_ENABLED=true
RAG_INDEX_PATH=~/.kd1-anime/rag/index.sqlite3
# 也可以把文档和示例复制到这两个默认目录，或改为其它绝对路径。
RAG_DOCS_DIR=~/.kd1-anime/knowledge/docs
RAG_EXAMPLES_DIR=~/.kd1-anime/knowledge/examples
RAG_EMBEDDING_API_KEY=your-embedding-key
RAG_EMBEDDING_BASE_URL=https://your-embedding-endpoint/v1
RAG_EMBEDDING_MODEL=your-embedding-model
RAG_RERANK_API_KEY=your-rerank-key
RAG_RERANK_BASE_URL=https://your-rerank-endpoint/v1
RAG_RERANK_MODEL=your-rerank-model
```

建立索引并检查服务：

```bash
kd1-anime rag index
kd1-anime rag status
kd1-anime rag search "TransformMatchingTex usage"
kd1-anime doctor --probe-rag
```

将可索引的 `.md`/`.rst`/`.py` 文档复制到上面的两个默认目录后执行 `rag index`；也可以
通过 `--docs-dir`、`--examples-dir` 或配置项指定其它源目录。索引文件始终默认写入
`~/.kd1-anime/rag/index.sqlite3`。

RAG 运行时服务暂时不可用时会降级继续：Embedding 失败则跳过检索，Reranker
失败则使用 Embedding 初排结果。索引只读取 `.md`/`.rst`/`.py`，并排除运行目录和疑似
密钥行；知识库源文件发生变化后，旧索引会被标记为过期，需重新执行
`kd1-anime rag index`。

如果 `RAG_ENABLED=true`，启动生成前还会要求索引存在且未过期；索引缺失或过期时先执行
`kd1-anime rag index`。源文件和 Embedding 模型未变化时，普通 `rag index` 会复用已有索引；
需要强制重新计算 Embedding 时使用 `kd1-anime rag index --rebuild`。

完整示例见 `.env.example`。安装脚本也会生成：

```text
~/.kd1-anime/.env
~/.kd1-anime/.env.example
```

## 使用

```bash
# 默认：交互式需求澄清 + 完整流水线
kd1-anime
kd1-anime chat

# 跳过澄清，直接生成
kd1-anime generate "展示欧拉公式 e^{iπ}+1=0 的直观推导"

# 只生成规划和代码，不提交 Slurm
kd1-anime generate "解释特征值的几何意义" --dry-run

# 仅查看场景规划
kd1-anime plan "解释傅里叶级数"
# 仅查看未经审查的模型原始规划
kd1-anime plan "解释傅里叶级数" --no-review
# 导出结构化计划（可交给 generate --plan）
kd1-anime plan "解释傅里叶级数" --output fourier-plan.json
# 从结构化计划继续生成；仍会执行计划/连续性审查
kd1-anime generate --plan fourier-plan.json --dry-run

# 对已有的单 Scene Manim 文件做安全检查并提交渲染
kd1-anime render scene.py --class MyScene --wait
# 也可仅提交后立即返回；输出会给出 run ID 和恢复命令
kd1-anime render scene.py --class MyScene

# 覆盖显式指定且已存在的输出文件
kd1-anime generate "..." --output final.mp4 --force

# 查看最近运行或某次运行的逐场景状态
kd1-anime status
kd1-anime status 20260728-120000-1234abcd
kd1-anime status 20260728-120000-1234abcd --json

# 查看某次运行的日志尾部；不触发模型或 Slurm 请求
kd1-anime logs 20260728-120000-1234abcd --scene-id 2 --lines 120

# 查看/清理本地 LLM 缓存
kd1-anime cache status
kd1-anime cache clear --yes

# 只重试一个失败场景，保留其它场景
kd1-anime retry 20260728-120000-1234abcd --scene-id 2

# 从清单恢复中断运行；不会重复提交仍有 Job ID 的场景
kd1-anime resume 20260728-120000-1234abcd

# 计划审查后人工确认再开始编码（默认不暂停）
kd1-anime generate "解释勾股定理" --approve-plan

# 检查依赖；--probe 会额外运行本地最小 FFmpeg/XeLaTeX/Manim 探测，不提交 Slurm
kd1-anime doctor --probe
# 分别探测主 LLM 与独立视觉 LLM（视觉探测会发送一张 1×1 图片）
kd1-anime doctor --probe-llm --probe-visual-llm

# 对已有运行执行独立视觉评估
kd1-anime evaluate 20260728-120000-1234abcd --visual
# 只评估清单中 Scene 2 的精确视频产物
kd1-anime evaluate 20260728-120000-1234abcd --scene-id 2

# 清理 30 天前的已结束运行目录
kd1-anime clean --older-than 30d --yes
```

启动交互会话、规划或生成流水线前，程序会先发送一次最小请求检查主 LLM
的配置、网络、鉴权和模型路由；启用 RAG 时还会检查索引是否存在/过期以及
Embedding 与 Reranker 模型。检查失败会立即退出，不进入后续 Agent 流程。
`status`、`version`、`doctor`、`clean` 等只读或诊断命令不会自动发送业务请求，
需要真实探测时可使用 `kd1-anime doctor --probe-llm`。
交互模式启动时不会扫描或弹出历史可恢复运行；需要恢复时请先用 `status` 找到
run ID，再显式执行 `kd1-anime resume <run-id>`。
`render` 不带 `--wait` 时只负责提交并立即返回，终端会显示 run ID。之后使用

`kd1-anime status <run-id>` 查看清单，或用 `kd1-anime resume <run-id>` 继续监控并合并。
`render --wait` 以及之后的 `resume` 只监控用户提供的 Scene，不会调用 Planner、Technical
Planner、Coder 或 Reviewer。

### 每次运行的产物

默认输出位于：

```text
~/.kd1-anime/workspace/runs/<timestamp>-<uuid>/
├── prompt.md
├── manifest.json         # schema v5：教学合同、状态账本、阶段、Job 与产物凭据
├── scenes/              # 生成的 Python 和 sbatch 脚本
├── logs/                # Slurm stdout/stderr
├── videos/              # 当前 run 的 Manim 媒体目录
├── eval_frames/         # 场景/成片关键帧（启用视觉评估时）
├── eval_reports/        # 严格结构化的场景与成片视觉报告
├── visual_candidates/   # 视觉修复失败时可恢复的候选代码
├── artifacts/           # 教学合同、计划编译、TechnicalSpec、审查、状态账本等阶段快照
└── output_final.mp4
```

如设置 `OUTPUT_FILE=/path/to/final.mp4`，最终视频写到该路径；其余中间产物仍保留在独立 run 目录中。
`manifest.json` 和 `.run.lock` 的权限为 `0600`。`resume` 会校验代码 SHA-256，持有运行级排他锁，并只复用与当前代码及渲染配置匹配的已验证视频。当前清单为 v5，包含全片教学合同和 StateLedger；v4 仅可查看，不能猜测迁移或继续修改，恢复时会给出明确错误，请重新生成。`clean` 只删除 run 目录，不会删除目录外的自定义输出。
增量运行复用的场景视频会复制到新 run 的私有目录，因此清理基准 run 不会破坏新 run 的恢复或重新拼接。

## 关键配置

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| `LLM_BASE_URL` | OpenAI API 地址 | 任意 OpenAI-compatible 端点 |
| `LLM_MODEL` | 空 | 必须设置为实际模型名 |
| `LLM_HEALTHCHECK_TIMEOUT` | `15` | 进入会话/流水线前的最小 API 探测超时（秒）；探测失败立即退出 |
| `LLM_PARALLEL_WORKERS` | `4` | 分镜/连续性审查等可并行 LLM 请求上限；代码交接阶段按场景顺序执行 |
| `LLM_MAX_TOKENS` | `32768` | 默认输出上限；端点拒绝该参数时会自动降级 |
| `LLM_CACHE_ENABLED` | `true` | 是否缓存完整的非流式 LLM 响应；流式交互永不缓存 |
| `LLM_CACHE_PATH` | 用户目录 cache/llm.sqlite3 | LLM 缓存 SQLite 路径 |
| `LLM_CACHE_MAX_ENTRIES` | `512` | 缓存最大条目数，设为 `0` 等同关闭写入 |
| `LLM_MAX_CONTEXT_CHARS` | `120000` | 单次 Agent 输入的总字符预算；超出时先裁剪低优先级区块 |
| `LLM_MAX_CODE_CONTEXT_CHARS` | `60000` | 代码、继承定义和修复代码区的字符预算；必需代码不会静默截断 |
| `LLM_MAX_REVIEW_CONTEXT_CHARS` | `90000` | Reviewer 输入总字符预算 |
| `LLM_MAX_TECHNICAL_SPEC_CHARS` | `30000` | TechnicalSpec 注入 Coder/Reviewer 的字符预算 |
| `MAX_TECHNICAL_SPEC_ATTEMPTS` | `3` | TechnicalSpec 确定性编译失败后的最大重生成次数 |
| `VISUAL_LLM_BASE_URL` | 空 | 独立多模态 OpenAI-compatible 端点；不回退主端点 |
| `VISUAL_LLM_MODEL` | 空 | 支持 `image_url` 输入的视觉模型；启用视觉评估时必须设置 |
| `VISUAL_LLM_PARALLEL_WORKERS` | `2` | 进程级并行视觉请求上限，与主 LLM 并发池分离；批处理任务共享此配额 |
| `RAG_ENABLED` | `false` | 是否启用本地知识检索；启用后启动前检查 Embedding/Reranker |
| `RAG_INDEX_PATH` | `~/.kd1-anime/rag/index.sqlite3` | SQLite RAG 索引路径 |
| `RAG_EMBEDDING_BASE_URL` / `RAG_EMBEDDING_MODEL` | 空 | 独立 Embedding 服务和模型 |
| `RAG_RERANK_BASE_URL` / `RAG_RERANK_MODEL` | 空 | 独立 Reranker 服务和模型 |
| `RAG_TOP_K` / `RAG_RERANK_TOP_N` | `8` / `4` | 向量初排和重排数量 |
| `RAG_MAX_CONTEXT_CHARS` | `12000` | 注入单次 Agent 请求的最大检索上下文 |
| `RAG_PARALLEL_WORKERS` | `2` | 跨批量任务共享的 RAG 请求并发上限 |
| `RAG_DOCS_DIR` / `RAG_EXAMPLES_DIR` | `~/.kd1-anime/knowledge/{docs,examples}` | 默认知识库源目录 |
| `WORKSPACE_DIR` | `~/.kd1-anime/workspace` | 运行、场景、日志、视频和评估产物根目录 |
| `MAX_SCENES` | `12` | 单次规划允许的最大场景数 |
| `MAX_PLAN_REVIEW_ROUNDS` | `2` | 单场景计划审查/重规划轮数 |
| `MAX_CONTINUITY_FIX_ROUNDS` | `2` | 全片连续性审查发现冲突后的局部分镜重规划轮数 |
| `SAFE_FALLBACK_ENABLED` | `true` | 复杂几何方案审查耗尽后是否自动切换为保守教学方案 |
| `MAX_IDENTICAL_REVIEW_ATTEMPTS` | `2` | 相同代码与审查反馈重复出现后的提前终止次数 |
| `MAX_PROMPT_CHARS` | `50000` | 用户需求最大字符数 |
| `MAX_CLARIFY_CONTEXT_CHARS` | `40000` | 澄清多轮对话发送给模型的最大字符数，超出时保留初始需求和最近回答 |
| `MAX_LOG_CHARS` | `30000` | 发送给 AutoFixer 的错误日志字符上限 |
| `MANIM_RENDERER` | `cairo` | `cairo` 使用 CPU；`opengl` 可使用 GPU |
| `MANIM_QUALITY` | `h` | Manim 质量级别 `l/m/h/p/k` |
| `MANIM_PIXEL_WIDTH` / `MANIM_PIXEL_HEIGHT` | `1920` / `1080` | 显式输出分辨率，也是产物身份的一部分 |
| `MANIM_FRAME_RATE` | `60` | 显式输出帧率，也是产物身份的一部分 |
| `SMOKE_RENDER_ENABLED` | `true` | 正式渲染前是否执行同 renderer 的轻量运行时检查 |
| `SMOKE_RENDER_QUALITY` | `l` | Smoke Render 质量级别（`l`/`m`） |
| `SMOKE_RENDER_TIMEOUT` | `180` | 单个 Smoke Render 的最长秒数 |
| `LOCAL_SMOKE_RENDER_ENABLED` | `false` | 是否在本地编码后执行额外的运行时预检；默认关闭，dry-run 不执行 |
| `LOCAL_SMOKE_RENDER_QUALITY` | `l` | 本地 Smoke Render 质量级别（`l`/`m`） |
| `LOCAL_SMOKE_RENDER_TIMEOUT` | `180` | 本地 Smoke Render 最长秒数 |
| `SLURM_CPUS_PER_TASK` | `4` | 每个场景作业的 CPU 数 |
| `SLURM_GPU_TYPE` | 空 | OpenGL 模式必须设置；Cairo 模式不会申请 GPU |
| `SLURM_MAX_IN_FLIGHT` | `0` | 最大在途场景作业数；`0` 表示不额外限制 |
| `SLURM_SUBMIT_RETRIES` | `3` | 明确失败时的 sbatch 重试次数；命令超时不会自动重提 |
| `MONITOR_QUEUE_TIMEOUT` | `3600` | 排队超时秒数，超时自动 `scancel` |
| `MONITOR_RUN_TIMEOUT` | `3600` | 运行超时秒数，超时自动 `scancel` |
| `MONITOR_UNKNOWN_TIMEOUT` | `300` | 集群状态连续不可查询的最短持续时间，避免短暂控制面故障误取消作业 |
| `MONITOR_ARTIFACT_GRACE` | `60` | Slurm 完成后等待共享文件系统同步最终 MP4 的秒数 |
| `MAX_INFRA_RETRIES` | `2` | 节点故障、抢占等基础设施终态的自动重新排队次数 |
| `ALLOW_PARTIAL_OUTPUT` | `false` | 是否允许缺失场景时合并部分视频 |
| `OVERWRITE_OUTPUT` | `false` | 是否允许覆盖已存在的自定义输出文件 |
| `TRANSITION_TYPE` | `fade` | 相邻场景的视频转场类型 |
| `TRANSITION_DURATION` | `0.5` | 相邻场景转场秒数，短视频会自动缩短 |
| `MERGE_VIDEO_CODEC` | `libx264` | 最终视频编码器 |
| `MERGE_VIDEO_PRESET` | `medium` | FFmpeg 编码速度/压缩 preset |
| `MERGE_VIDEO_CRF` | `18` | 最终视频质量参数（0–51） |
| `MERGE_AUDIO_SAMPLE_RATE` | `48000` | 跨场景音频统一采样率 |
| `MERGE_AUDIO_CHANNEL_LAYOUT` | `stereo` | 跨场景音频统一声道布局 |
| `SLURM_CONTAINER_IMAGE` | 空 | 可选 Apptainer 镜像路径 |
| `SLURM_REQUIRE_CONTAINER` | `false` | 为 `true` 时未配置镜像即拒绝执行 |
| `SLURM_CONTAINER_DISABLE_NETWORK` | `false` | 容器支持时通过独立网络命名空间禁用网络；需先在目标集群验证 |
| `ENABLE_AUTO_EVAL` | `false` | 合并后执行确定性代码/效率评估改进循环 |
| `ENABLE_VISUAL_EVAL` | `false` | 合并前执行逐场景视觉质量门，并在合并后生成成片视觉报告 |
| `VISUAL_EVAL_FRAME_COUNT` | `6` | 每个场景/成片抽取的语义关键帧数（1–8，成片预算优先保留真实相邻场景边界） |
| `VISUAL_EVAL_THRESHOLD` | `3.5` | 场景视觉通过阈值（1–5）；重大问题无论均分都会触发修复 |
| `MAX_VISUAL_FIX_ATTEMPTS` | `2` | 每个场景最多由视觉诊断触发的 Coder 修复次数 |

### 并行与 GPU 说明

- **场景级并行**：编码阶段按 Scene ID 顺序传递连续性上下文；编码完成后，每个 Scene 是一个独立 Slurm job，调度器可将多个场景分配到不同节点或 CPU 核心并行渲染。这通常是最有效、最稳定的 Manim 并行方式。
- **计划级屏障**：所有 Detail 完成后先经过确定性计划编译、Plan Review 和全片连续性 Review；数学断言、定义域、教学依赖或视觉单元冲突会回到 Planner，不会让 Coder 反复修补错误计划。
- **边界视觉审查**：启用视觉评估且场景数不少于 2 时，合并前会抽取选定相邻场景的真实结尾帧和下一场景开头帧。数学/故事问题回到 Planner，交接问题回到 Continuity，布局问题才回到 Coder。
- **并发限流**：共享集群可设置 `SLURM_MAX_IN_FLIGHT=4` 等值，程序会分批提交并在完成后继续下一批。
- **单场景内部**：本项目不把单个 Scene 的动画帧拆成多个进程；ManimCE 本身也没有通用的单 Scene 多进程渲染开关。
- **GPU**：只有 `MANIM_RENDERER=opengl` 时才申请 GPU。Cairo 是 CPU 渲染，配置了 `SLURM_GPU_TYPE` 也不会浪费 GPU 配额。

## 安全模型

LLM 生成代码在提交前会经过 AST 校验，包括顶层动态执行、装饰器、NumPy 文件 API 和本地图片/SVG 加载检查，但静态校验仍不能等价于完整沙箱。处理不可信输入或多人共享集群时，建议：

1. 构建包含 Manim、TeX Live、FFmpeg 和字体的只读 Apptainer 镜像；
2. 设置 `SLURM_CONTAINER_IMAGE=/path/to/image.sif`；
3. 设置 `SLURM_REQUIRE_CONTAINER=true`；
4. 若集群允许无特权网络命名空间，测试通过后设置 `SLURM_CONTAINER_DISABLE_NETWORK=true`。

容器作业使用 `--containall --cleanenv --no-home`，仅绑定当前 run 目录；OpenGL 模式额外使用 `--nv` 并显式传递 `PYOPENGL_PLATFORM`（例如 `egl`）。未配置容器时程序保持兼容运行，但 `doctor` 和生成流程会给出醒目的安全提示。

可使用 `kd1-anime doctor --probe-llm` 验证主 LLM endpoint，使用
`kd1-anime doctor --probe-visual-llm` 验证视觉端点确实接受图片消息；使用
`kd1-anime doctor --security-strict` 检查是否启用了容器 fail-closed 策略。

## 开发与验证

```bash
ruff check .
python -m compileall -q .
bash -n install.sh
pytest -q
python -m build --wheel
```

CI 会执行静态检查、编译检查、Shell 语法检查、测试和 wheel 构建。

## 技术栈

Python 3.10+ · OpenAI-compatible API · Pydantic · Rich · Typer · prompt_toolkit · Manim CE · FFmpeg · TeX Live · Slurm · 可选 Apptainer

## 增量渲染

增量渲染允许你基于上一次运行的结果，安全复用身份完全一致的场景视频。

```bash
# 普通渲染
kd1-anime generate "解释欧拉公式 e^{iπ}+1=0 的推导"

# 基于上一次运行进行增量渲染
kd1-anime generate "解释欧拉公式 e^{iπ}+1=0 的几何意义" --incremental 20260801-120000-1234abcd
```

### 增量渲染工作原理

1. 新运行仍会完成规划、代码生成、确定性校验和 Reviewer 审查。
2. 审查通过后，只有代码 SHA-256 与旧场景一致、渲染 profile 哈希一致、旧视频哈希仍匹配且元数据已验证时才复用。
3. 任一身份条件不满足都会重新提交 Slurm；复用不会伪造 Job ID。
4. 合并阶段再次校验场景 ID、类名、代码哈希、配置哈希和视频哈希。

### 增量渲染优势

实际节省取决于新旧代码是否逐字节一致。该模式节省的是已确认不变场景的 Slurm 渲染与视频编码成本，不跳过生成和审查安全门。

## 批量并行处理

批量处理允许你从文件读取多个 prompt，并行处理多个动画项目。

```bash
# 创建 prompts 文件
cat > prompts.txt << 'EOF'
解释欧拉公式 e^{iπ}+1=0 的推导
展示傅里叶级数的几何意义
可视化特征值和特征向量
EOF

# 批量处理
kd1-anime batch prompts.txt --max-parallel 3

# 使用 JSON 格式
cat > prompts.json << 'EOF'
{
  "prompts": [
    "解释欧拉公式 e^{iπ}+1=0 的推导",
    "展示傅里叶级数的几何意义",
    "可视化特征值和特征向量"
  ]
}
EOF

kd1-anime batch prompts.json

# 指定 --output-dir 后输出为 ~/.kd1-anime/exports/task_001.mp4、task_002.mp4……；
# 未指定时，输出保存在每个任务自己的 ~/.kd1-anime/workspace/runs/<run-id>/ 目录。
kd1-anime batch prompts.json --output-dir ~/.kd1-anime/exports --max-parallel 3
```

### 批量处理选项

- `--max-parallel, -j`：最大并行任务数（默认：3）
- `--dry-run`：只生成场景代码，不提交 Slurm 渲染
- `--output-dir, -o`：输出目录

`--max-parallel` 限制同时运行的项目数；所有项目还共享进程级
`LLM_PARALLEL_WORKERS` 和 `SLURM_MAX_IN_FLIGHT` 配额，避免每个项目各自放大并发。
输出路径在启动前统一解析并检查，重复目标或禁止覆盖的已存在文件会直接报错。

### 批量处理输出

批量处理完成后会显示摘要：

```
批量处理结果
  ✓ 任务 1 completed (120.5s) /path/to/task_001.mp4
  ✓ 任务 2 completed (118.2s) /path/to/task_002.mp4
  ✓ 任务 3 completed (125.7s) /path/to/task_003.mp4
```

## 许可证

MIT
