# ARCHITECTURE

## 0. 全局系统提示词 (System Prompt for Claude Code)
> **给 Claude Code 的指令**:
> 你现在是一个资深的 Python 后端架构师和 AI 应用专家.我们需要构建一个名为 `kd1-anime` 的 CLI 工具.
> **核心原则**:
> 1. **拒绝重型框架**:绝对不要使用 LangChain, AutoGen, LangGraph 等复杂的 Agent 框架.请**仅使用原生 Python,Pydantic (用于数据校验和结构化输出) 和 状态机模式**来实现多 Agent 协同.
> 2. **高鲁棒性**:涉及 LLM API 调用,Slurm 集群交互,文件系统读取的地方,必须有完善的 `try-except` 和重试机制.
> 3. **模块化设计**:严格按照下文定义的目录结构和模块接口进行开发,保持代码解耦.
> 4. **CLI 体验**:使用 `rich` 库来美化命令行输出,展示 Agent 的思考过程,Slurm 任务状态和渲染进度.

---

## 1. 技术栈与依赖
*   **语言**:Python 3.10+
*   **LLM 交互**:`openai` 库 (兼容 DeepSeek/GLM 的 API 接口)
*   **数据校验与结构化**:`pydantic`, `pydantic-settings`
*   **CLI 美化与交互**:`rich`, `click` 或 `typer`
*   **系统交互**:`subprocess`, `pathlib`, `time`
*   **视频处理**:`ffmpeg` (通过 `subprocess` 调用,不依赖重型 Python 视频库)

---

## 2. 系统架构与工作流 (State Machine)

系统核心是一个**有限状态机 (FSM)**,由 `Orchestrator` 驱动.

```text
[用户输入 Prompt] 
      │
      ▼
┌─────────────────┐     失败/超时      ┌─────────────────┐
│ 1. Planner Agent│ ─────────────────▶ │   Error Handler │
│ (拆解场景为JSON)│                    └─────────────────┘
└────────┬────────┘
         │ 成功 (输出 Scene List)
         ▼
┌─────────────────┐ ◀──────────────────────────────┐
│ 2. Coder Agent  │                                │
│ (生成Manim代码) │                                │ (携带Error Log)
└────────┬────────┘                                │
         │                                         │
         ▼                                         │
┌─────────────────┐      Review不通过             │
│ 3. Reviewer Agent│ ──────────────────────────────┘
│ (代码审查/逻辑校验)│
└────────┬────────┘
         │ Review通过
         ▼
┌─────────────────┐
│ 4. Slurm Dispatcher│ (生成.sh脚本,sbatch提交,获取JobID)
└────────┬────────┘
         │
         ▼
┌─────────────────┐      渲染报错/Traceback     ┌─────────────────┐
│ 5. Monitor Agent  │ ─────────────────────────▶ │ 6. Auto-Fix Agent│
│ (轮询squeue,读log)│                            │ (提取错误上下文) │
└────────┬────────┘                             └─────────────────┘
         │ 渲染成功 (生成 mp4 片段)
         ▼
┌─────────────────┐
│ 7. Video Merger   │ (使用 ffmpeg 按序拼接所有 mp4)
└────────┬────────┘
         │
         ▼
[输出最终 output_final.mp4]
```

---

## 3. 目录结构规范
请 Claude Code 严格按照以下目录结构生成代码:

```text
manim-107/
├── config.py             # 环境变量,API Key,Slurm 分区配置 (使用 pydantic-settings)
├── main.py               # CLI 入口 (使用 typer/click)
├── orchestrator.py       # 核心状态机,串联所有 Agent 和模块
├── agents/               # 多 Agent 模块
│   ├── __init__.py
│   ├── base.py           # BaseAgent 基类 (封装 LLM 调用,重试,结构化输出)
│   ├── planner.py        # Planner Agent (场景拆解)
│   ├── coder.py          # Coder Agent (代码生成)
│   ├── reviewer.py       # Reviewer Agent (代码审查)
│   └── auto_fixer.py     # Auto-Fix Agent (错误修复)
├── cluster/              # 算力集群交互模块
│   ├── __init__.py
│   ├── slurm.py          # Slurm 任务提交,状态查询 (sbatch, squeue, sacct)
│   └── templates/        # Slurm 脚本模板 (render_job.sh.j2)
├── media/                # 本地媒体处理
│   ├── __init__.py
│   └── merger.py         # FFmpeg 视频拼接逻辑
├── workspace/            # 运行时生成的临时目录 (代码,日志,视频片段,需加入 .gitignore)
│   ├── scenes/           # 存放生成的 python 代码
│   ├── logs/             # 存放 Slurm 输出的 out/err 日志
│   └── videos/           # 存放渲染出的 mp4 片段
└── requirements.txt      # 依赖清单
```

---

## 4. 核心模块实现细节 (给 Claude Code 的具体指导)

### 4.1 基础 Agent 类 (`agents/base.py`)
*   **要求**:封装 `openai.ChatCompletion`.
*   **特性**:
    *   支持传入 `response_format={"type": "json_object"}` 或 Pydantic 模型进行强制结构化输出.
    *   内置指数退避重试机制 (Exponential Backoff),应对 API 限流或网络波动.
    *   使用 `rich.console` 打印 Agent 的"思考过程"(如:`[Planner] 正在拆解场景...`).

### 4.2 Planner Agent (`agents/planner.py`)
*   **输入**:用户的自然语言 Prompt.
*   **System Prompt 核心**:"你是一个数学动画导演.请将用户的需求拆解为多个独立的 Manim Scene.每个 Scene 必须是一个完整的,可独立渲染的动画片段.输出严格的 JSON 格式."
*   **Pydantic 输出结构**:
    ```python
    class ScenePlan(BaseModel):
        scene_id: int
        title: str
        description: str # 详细的视觉和数学逻辑描述
        math_concept: str # 涉及的数学概念,用于提示 Coder
    ```

### 4.3 Coder & Reviewer Agent (`agents/coder.py`, `agents/reviewer.py`)
*   **Coder Agent**:
    *   **输入**:单个 `ScenePlan` + **Manim 核心 API 知识库 (Few-shot)**.
    *   **System Prompt 核心**:"你是一个 Manim 专家.根据场景描述编写 Python 代码.必须继承 `Scene` 类,实现 `construct` 方法.不要包含任何解释,只输出纯 Python 代码,包裹在 ```python ``` 中."
    *   *关键*:必须在 System Prompt 中注入 Manim 的常用类(如 `MathTex`, `Axes`, `Create`, `Transform`)的正确用法,防止大模型幻觉.
*   **Reviewer Agent**:
    *   **输入**:Coder 生成的代码.
    *   **System Prompt 核心**:"审查以下 Manim 代码.检查:1. 是否缺少必要的 import? 2. 数学逻辑是否合理? 3. 是否使用了已废弃的 API? 输出 JSON:`{"is_valid": bool, "feedback": str}`."
    *   **流转**:如果 `is_valid` 为 False,将 `feedback` 拼接到 Coder 的上下文中,要求重写(最多 3 轮).

### 4.4 Slurm 调度模块 (`cluster/slurm.py`)
*   **核心功能**:
    1.  **生成脚本**:读取 `templates/render_job.sh.j2` (Jinja2 模板),填入 `scene_id`, `python_file_path`, `log_path`.
        *   *注意*:Manim 渲染命令需指定输出目录,例如:`manim -qh --media_dir ./workspace/videos/{scene_id} {python_file}`.
    2.  **提交任务**:`subprocess.run(["sbatch", script_path], capture_output=True, text=True)`.解析输出获取 `Job ID` (正则匹配 `Submitted batch job (\d+)`).
    3.  **状态轮询**:`subprocess.run(["squeue", "-j", job_id, "-h", "-o", "%T"])`.如果返回空或 `COMPLETED`/`FAILED`,则停止轮询.

### 4.5 监控与自愈模块 (`orchestrator.py` 中的逻辑)
*   **日志解析**:当 Slurm 任务状态为 `FAILED` 时,读取 `workspace/logs/scene_{id}_{job_id}.err`.
*   **截断策略**:只读取 `.err` 文件的**最后 80 行**(防止 Token 爆炸),提取包含 `Traceback` 和具体 `Error` 的部分.
*   **Auto-Fix Agent**:将截断的 Error Log 和原代码发给 Auto-Fix Agent,要求其输出修复后的代码,然后重新走 Coder -> Slurm 流程.

### 4.6 视频拼接模块 (`media/merger.py`)
*   **逻辑**:
    1.  遍历 `workspace/videos/`,找到所有 scene 生成的 mp4 文件(Manim 默认输出在 `media_dir/videos/类名/1080p60/` 下,需要写一个辅助函数去深层目录捞取 mp4).
    2.  按 `scene_id` 排序.
    3.  生成一个 `filelist.txt`,内容为 `file 'path/to/scene1.mp4'\nfile 'path/to/scene2.mp4'`.
    4.  调用 FFmpeg:`ffmpeg -f concat -safe 0 -i filelist.txt -c copy output_final.mp4`.

---

## 5. 给 Claude Code 的逐步执行指令 (Action Plan)

**请按照以下步骤为我生成代码,每完成一步请等待我的确认:**

*   **Step 1**: 生成 `config.py` 和 `requirements.txt`.定义好所有环境变量(如 `DEEPSEEK_API_KEY`, `SLURM_PARTITION` 等).
*   **Step 2**: 生成 `agents/base.py`,实现带有重试机制和 Rich 控制台输出的 LLM 调用基类.
*   **Step 3**: 生成 `agents/planner.py`, `coder.py`, `reviewer.py`, `auto_fixer.py`.请为它们编写高质量的 System Prompt(特别是 Coder,请务必在 Prompt 中内置 Manim 的基础 API 示例).
*   **Step 4**: 生成 `cluster/slurm.py` 和 `cluster/templates/render_job.sh.j2`.确保 Slurm 脚本能正确加载 Conda 环境并执行 Manim 渲染.
*   **Step 5**: 生成 `media/merger.py`,实现基于 FFmpeg 的视频无损拼接.
*   **Step 6**: 生成 `orchestrator.py`,实现核心的状态机流转逻辑,串联上述所有模块.
*   **Step 7**: 生成 `main.py`,使用 `typer` 构建 CLI 入口,接收用户输入并启动 Orchestrator.
