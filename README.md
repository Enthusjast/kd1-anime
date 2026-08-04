# kd1-anime

`kd1-anime` 是一个 AI Agent 驱动的 Manim Community Edition 数学动画生成器。用户用自然语言描述目标，程序会澄清需求、规划场景、生成并审查代码、提交 Slurm 并行渲染、自动修复失败场景，并用 FFmpeg 合并最终视频。

## 主要特性

- **对话式终端交互**：先追问受众、时长、内容重点和视觉风格，再开始生成。
- **两阶段规划**：先生成全片概要，再为每个场景生成详细导演分镜。
- **多 Agent 流水线**：Planner → Coder → Reviewer → AutoFixer，不依赖 LangChain 等重型框架。
- **并行执行**：独立场景的 LLM 请求可并发；所有场景分别提交 Slurm，可由集群并行渲染。
- **确定性安全校验**：在 LLM 审查之外，使用 Python AST 检查语法、Scene 结构、导入和危险调用。
- **运行隔离**：每次运行写入独立的 `workspace/runs/<run-id>/`，避免并发运行和旧产物互相污染。
- **中断恢复**：原子 `manifest.json` 保存 FSM、代码哈希和 Slurm Job ID，可查询并恢复中断运行。
- **可选容器隔离**：可用 Apptainer 执行 LLM 生成的 Manim 代码。
- **可恢复渲染**：监控 Slurm 状态、区分排队/运行超时、失败后读取日志并自动修复。
- **通用 LLM 接口**：通过 `.env` 配置任意 OpenAI-compatible API，不绑定 DeepSeek 或其他特定厂商。

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
export KD1_ANIME_REF=v0.3.0
export KD1_ANIME_ARCHIVE_SHA256=<release-zip-sha256>
bash /tmp/kd1-anime-install.sh
```

摘要不匹配或 ref 含路径遍历字符时，安装器会在调用 pip 前终止。

安装器会全自动完成：

1. 加载 `python3.12/3.12` 和 `miniconda/py312` module（若系统提供 module）。
2. 创建或复用 `manim_env` conda 环境。
3. 安装 Manim CE、FFmpeg 和 Noto CJK 字体。
4. 依次检查 PATH、`/usr/local/texlive` 和 `~/texlive` 中已有的 XeLaTeX；完整环境直接复用且不调用 `tlmgr`。
5. 仅当现有 TeX Live 缺失或无法无 sudo 补齐依赖时，才从 USTC CTAN 镜像安装最小用户目录版到 `~/texlive/<release>/`。
6. 只安装 Manim/XeLaTeX 所需包及 `ctex`、`xeCJK`、`fontspec`，不安装完整 TeX Live scheme/collection。
7. 安装 `kd1-anime` 命令，不保留远程源码目录。
8. 在 `~/.local/bin` 安装 `kd1-anime` / `manim-env` 包装器，并写入 conda 激活钩子、shell 函数和用户级配置模板。

Coder 生成的 `Tex`/`MathTex` 统一使用 `xelatex` 和 `.xdv`，并加载 `ctex`；普通中文文字使用 Noto CJK 字体和 Manim `Text`（Pango）。

安装完成后无需手动 `source` RC 文件或激活 conda，即可直接运行：

```bash
# 启动程序
kd1-anime

# 可选：进入已激活 manim_env 的交互 shell
manim-env

# 配置 OpenAI-compatible API
$EDITOR ~/.config/kd1-anime/.env
```

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
系统环境变量 > 当前目录 .env > ~/.config/kd1-anime/.env
```

最少需要填写：

```dotenv
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
LLM_MODEL=your-model-name
```

完整示例见 `.env.example`。安装脚本也会生成：

```text
~/.config/kd1-anime/.env
~/.config/kd1-anime/.env.example
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

# 对已有的单 Scene Manim 文件做安全检查并提交渲染
kd1-anime render scene.py --class MyScene --wait
# 也可仅提交后立即返回；输出会给出 run ID 和恢复命令
kd1-anime render scene.py --class MyScene

# 覆盖显式指定且已存在的输出文件
kd1-anime generate "..." --output final.mp4 --force

# 查看最近运行或某次运行的逐场景状态
kd1-anime status
kd1-anime status 20260728-120000-1234abcd

# 从清单恢复中断运行；不会重复提交仍有 Job ID 的场景
kd1-anime resume 20260728-120000-1234abcd

# 清理 30 天前的已结束运行目录
kd1-anime clean --older-than 30d --yes
```
`render` 不带 `--wait` 时只负责提交并立即返回，终端会显示 run ID。之后使用

`kd1-anime status <run-id>` 查看清单，或用 `kd1-anime resume <run-id>` 继续监控并合并。

### 每次运行的产物

默认输出位于：

```text
workspace/runs/<timestamp>-<uuid>/
├── prompt.md
├── manifest.json         # FSM、场景状态、代码哈希和 Slurm Job ID
├── scenes/              # 生成的 Python 和 sbatch 脚本
├── logs/                # Slurm stdout/stderr
├── videos/              # 当前 run 的 Manim 媒体目录
└── output_final.mp4
```

如设置 `OUTPUT_FILE=/path/to/final.mp4`，最终视频写到该路径；其余中间产物仍保留在独立 run 目录中。
`manifest.json` 和 `.run.lock` 的权限为 `0600`。`resume` 会校验代码 SHA-256，并持有运行级排他锁，防止两个进程同时恢复同一批作业。`clean` 只删除 run 目录，不会删除目录外的自定义输出。

## 关键配置

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| `LLM_BASE_URL` | OpenAI API 地址 | 任意 OpenAI-compatible 端点 |
| `LLM_MODEL` | 空 | 必须设置为实际模型名 |
| `LLM_PARALLEL_WORKERS` | `4` | 逐场景 LLM 并发数 |
| `MAX_SCENES` | `12` | 单次规划允许的最大场景数 |
| `MAX_PROMPT_CHARS` | `50000` | 用户需求最大字符数 |
| `MAX_LOG_CHARS` | `30000` | 发送给 AutoFixer 的错误日志字符上限 |
| `MANIM_RENDERER` | `cairo` | `cairo` 使用 CPU；`opengl` 可使用 GPU |
| `MANIM_QUALITY` | `h` | Manim 质量级别 `l/m/h/p/k` |
| `SLURM_CPUS_PER_TASK` | `4` | 每个场景作业的 CPU 数 |
| `SLURM_GPU_TYPE` | 空 | OpenGL 模式必须设置；Cairo 模式不会申请 GPU |
| `SLURM_MAX_IN_FLIGHT` | `0` | 最大在途场景作业数；`0` 表示不额外限制 |
| `SLURM_SUBMIT_RETRIES` | `3` | 明确失败时的 sbatch 重试次数；命令超时不会自动重提 |
| `MONITOR_QUEUE_TIMEOUT` | `3600` | 排队超时秒数，超时自动 `scancel` |
| `MONITOR_RUN_TIMEOUT` | `3600` | 运行超时秒数，超时自动 `scancel` |
| `ALLOW_PARTIAL_OUTPUT` | `false` | 是否允许缺失场景时合并部分视频 |
| `OVERWRITE_OUTPUT` | `false` | 是否允许覆盖已存在的自定义输出文件 |
| `SLURM_CONTAINER_IMAGE` | 空 | 可选 Apptainer 镜像路径 |
| `SLURM_REQUIRE_CONTAINER` | `false` | 为 `true` 时未配置镜像即拒绝执行 |

### 并行与 GPU 说明

- **场景级并行**：每个 Scene 是一个独立 Slurm job，调度器可将多个场景分配到不同节点或 CPU 核心并行渲染。这通常是最有效、最稳定的 Manim 并行方式。
- **并发限流**：共享集群可设置 `SLURM_MAX_IN_FLIGHT=4` 等值，程序会分批提交并在完成后继续下一批。
- **单场景内部**：本项目不把单个 Scene 的动画帧拆成多个进程；ManimCE 本身也没有通用的单 Scene 多进程渲染开关。
- **GPU**：只有 `MANIM_RENDERER=opengl` 时才申请 GPU。Cairo 是 CPU 渲染，配置了 `SLURM_GPU_TYPE` 也不会浪费 GPU 配额。

## 安全模型

LLM 生成代码在提交前会经过 AST 校验，包括顶层动态执行、装饰器、NumPy 文件 API 和本地图片/SVG 加载检查，但静态校验仍不能等价于完整沙箱。处理不可信输入或多人共享集群时，建议：

1. 构建包含 Manim、TeX Live、FFmpeg 和字体的只读 Apptainer 镜像；
2. 设置 `SLURM_CONTAINER_IMAGE=/path/to/image.sif`；
3. 设置 `SLURM_REQUIRE_CONTAINER=true`。

容器作业使用 `--containall --cleanenv --no-home`，仅绑定当前 run 目录；OpenGL 模式额外使用 `--nv`。

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

## 许可证

MIT


## 增量渲染

增量渲染允许你基于上一次运行的结果，只重新渲染受 prompt 变化影响的场景，节省时间和计算资源。

```bash
# 普通渲染
kd1-anime generate "解释欧拉公式 e^{iπ}+1=0 的推导"

# 基于上一次运行进行增量渲染
kd1-anime generate "解释欧拉公式 e^{iπ}+1=0 的几何意义" --incremental 20260801-120000-1234abcd
```

### 增量渲染工作原理

1. **场景级比较**：比较新旧场景的代码 hash，只重新渲染变化的场景
2. **视频复用**：未变化场景的视频直接从旧 run 目录复用
3. **智能合并**：VideoMerger 自动处理来自不同 run 目录的视频

### 增量渲染优势

| 场景 | 普通渲染 | 增量渲染 |
|------|---------|---------|
| 修改 1 个场景 | 渲染全部 10 个场景 | 只渲染 1 个 |
| 调整 prompt | 全部重新生成 | 智能复用 |
| 节省时间 | - | 70-90% |

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
```

### 批量处理选项

- `--max-parallel, -j`：最大并行任务数（默认：3）
- `--dry-run`：只生成场景代码，不提交 Slurm 渲染
- `--output-dir, -o`：输出目录

### 批量处理输出

批量处理完成后会显示摘要：

```
批量处理摘要
============
总任务数: 3
成功: 3
失败: 0
总耗时: 120.5秒

详细结果:
  ✓ 任务 1: /path/to/output_1.mp4
  ✓ 任务 2: /path/to/output_2.mp4
  ✓ 任务 3: /path/to/output_3.mp4
```

