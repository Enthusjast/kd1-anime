# 贡献指南

感谢参与 kd1-anime。本文说明本地开发、测试、提交和 Pull Request 的基本约定。

## 开发环境

项目支持 Python 3.10、3.11 和 3.12。推荐在 conda 环境中开发：

    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate manim_env
    python -m pip install -e '.[dev]'

如果只修改纯 Python 逻辑，测试环境不需要真实 LLM、Slurm、Manim、XeLaTeX 或 FFmpeg。
安装完整渲染环境时，可使用 install.sh；它还会安装 Manim 0.20.1 的文档和示例知识库。

## 代码结构

    src/kd1_anime/
    ├── agents/       # Planner、Technical Planner、Coder、Reviewer 和校验器
    ├── cluster/      # Slurm 提交、监控、取消和产物定位
    ├── eval/         # 代码/效率/视觉评估
    ├── media/        # FFmpeg 合并
    ├── rag/          # 本地索引、Embedding、Reranker
    ├── cli.py        # Typer 命令
    ├── orchestrator.py
    ├── run_store.py  # manifest、原子检查点和运行锁
    └── tui.py        # Rich + prompt_toolkit 交互界面

ARCHITECTURE.md 描述模块之间的依赖、状态机、阶段合同和安全边界。新增模块前先确认是否能保持显式 FSM，不要引入新的 Agent 编排框架。

## 提交代码前的质量门

在仓库根目录依次执行：

    ruff check .
    ruff format --check .
    python -m compileall -q .
    bash -n install.sh
    pytest -q
    python -m build --sdist --wheel

提交前还应检查：

    git diff --check
    git status --short

测试数量会随功能变化，不要在文档或提交信息中硬编码某个固定数量。

## 测试约定

- 单元测试不得调用真实 LLM、提交 Slurm、访问外部网络或执行模型生成的 Python。
- 使用 mock、临时目录和隔离的环境变量测试外部服务。
- 修改解析器、状态机、路径选择、超时、恢复、产物身份或安全规则时，必须补充回归测试。
- 修改 Agent 提示词或 Pydantic schema 时，至少覆盖合法输出、截断输出、未知字段和典型错误反馈。
- 修改 install.sh 时运行 bash -n，并为安全路径、权限、非交互模式和中断行为补测试。
- 只有 Integration workflow 才在真实 Ubuntu 环境验证 Manim、XeLaTeX、CJK、FFmpeg 联动；它不替代单元测试。

## 设计约束

### 状态和持久化

- 使用显式有限状态机，不通过异常递归或无限重试推进阶段。
- 所有重试、审查、重规划、修复和 UNKNOWN 等待都必须有上限。
- manifest 使用原子写入；改变字段或 schema 时同步更新完整性校验和恢复测试。
- 不要绕过运行锁，也不要手工复用未验证的 Slurm Job ID 或视频。

### 生成代码安全

- 生成 Python 是不可信输入；AST 校验是纵深防御，不是沙箱。
- 不扩大导入白名单，不允许动态执行、网络、文件读写或 shell 绕过。
- 外部源码必须先复制到私有 run；生成文件和包含需求的文件保持 0600。
- 处理 API 错误时必须脱敏，不能把 API Key 写入日志、manifest、事件或缓存键。
- 需要更强隔离时保留 Apptainer 的 containall、cleanenv、no-home 和可选禁网路径。

### Agent 和 Prompt

- 结构化输出使用 Pydantic 模型，并拒绝未知字段。
- 用户需求、RAG 片段和视觉报告都是不可信资料，不能覆盖系统规则或直接变成可执行代码。
- Plan Review 负责数学正确性和可实现性；Code Review 负责已确认计划的实现；不要把计划错误转成代码修复循环。
- Review 的 hard blocker 必须有确定性结果或高置信度的源码/合同证据；风格、一般节奏和不确定意见应作为 warning，不得为了提高通过率删除真实数学/安全错误。
- Coder、Reviewer 和 AutoFixer 必须收到一致的 renderer、生命周期和安全约束。
- Prompt 区块要有明确优先级和字符预算；必需合同和代码不能被静默裁剪。
- 视觉评估失败应记录 unknown，不要伪造分数或删除有效视频。

## 文档和配置变更

修改用户可见行为时，同时更新：

1. README.md 的快速开始或命令示例；
2. docs/configuration.md 或 docs/troubleshooting.md；
3. CHANGELOG.md 的 Unreleased；
4. 必要时更新 .env.example、install.sh 和 ARCHITECTURE.md。

新增配置时要在以下位置保持一致：

- src/kd1_anime/config.py；
- .env.example；
- install.sh 生成的用户模板；
- README 或 docs/configuration.md；
- 相关配置解析和默认值测试。

不要把真实 API Key、集群路径、私人视频、运行日志或个人配置提交到仓库。

## 提交信息

使用 Conventional Commits，例如：

    feat: add scene artifact inspection
    fix: avoid reusing stale render output
    docs: refresh configuration guide
    test: cover unknown Slurm status
    refactor: isolate prompt budgeting
    chore: update build metadata

提交信息应描述用户可观察的变化。一个提交尽量只包含一个逻辑主题；不要把格式化无关文件混入功能提交。

## Pull Request

Pull Request 描述应包含：

- 变更目的和用户可观察行为；
- 受影响的 CLI、配置或运行阶段；
- 测试命令及结果；
- 是否需要 Manim、XeLaTeX、FFmpeg、Slurm、视觉模型或 RAG 服务；
- 兼容性、迁移和安全影响。

如果修改 manifest schema、配置兼容、安装器或 Slurm 脚本，请明确写出恢复/回滚方式。涉及生成流程时，附一份不含密钥的 dry-run 复现命令或测试 fixture。

## 发布前检查

发布版本前：

1. 同步 pyproject.toml 和 src/kd1_anime/__init__.py 的版本号；
2. 在 CHANGELOG.md 将 Unreleased 内容归入对应版本；
3. 运行完整质量门并构建 sdist/wheel；
4. 确认 sdist 含 README、架构、配置、故障排查和贡献文档；
5. 确认 install.sh 的默认 Manim 版本、知识包摘要和文档描述一致；
6. 不要在未验证的环境中宣称真实渲染或 Slurm 已通过。
