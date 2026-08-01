# 贡献指南

感谢你对 kd1-anime 项目的关注！本文档将帮助你了解如何参与项目开发。

## 目录

- [开发环境搭建](#开发环境搭建)
- [代码规范](#代码规范)
- [提交规范](#提交规范)
- [Pull Request 流程](#pull-request-流程)
- [测试指南](#测试指南)
- [文档贡献](#文档贡献)
- [问题反馈](#问题反馈)

## 开发环境搭建

### 前置要求

- Python 3.10+
- Git
- 可选：conda（用于完整环境）

### 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/Enthusjast/kd1-anime.git
cd kd1-anime

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\\Scripts\\activate  # Windows

# 3. 安装开发依赖
python -m pip install -e '.[dev]'

# 4. 运行测试验证安装
pytest -q

# 5. 检查环境
kd1-anime doctor
```

### 完整环境（含 Manim）

如果需要测试完整的渲染功能：

```bash
# 使用安装脚本（无 sudo）
bash install.sh

# 或手动安装
conda create -n manim_env python=3.12
conda activate manim_env
pip install manim
```

## 代码规范

### Python 版本

- 支持 Python 3.10+
- 使用现代 Python 语法（类型注解、match/case 等）

### 代码风格

我们使用 [Ruff](https://github.com/astral-sh/ruff) 进行代码检查和格式化：

```bash
# 检查代码
ruff check .

# 自动修复
ruff check --fix .

# 格式化
ruff format .
```

### 命名规范

- **模块名**：小写下划线（`run_store.py`）
- **类名**：大驼峰（`PlannerAgent`）
- **函数名**：小写下划线（`plan_outline`）
- **常量**：全大写下划线（`MAX_SCENES`）
- **私有成员**：单下划线前缀（`_callback`）

### 类型注解

所有公共 API 必须有类型注解：

```python
def plan_outline(self, user_prompt: str) -> list[SceneOutline]:
    ...

class ScenePlan(BaseModel):
    scene_id: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=200)
```

### 路径处理

使用 `pathlib.Path`：

```python
from pathlib import Path

config_dir = Path.home() / ".config" / "kd1-anime"
```

### 子进程调用

使用 `subprocess.run` 并禁用 shell：

```python
subprocess.run(["ffmpeg", "-i", input_file, output_file], check=True)
```

## 提交规范

我们遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

### 提交类型

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式调整（不影响逻辑）
- `refactor`: 代码重构
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建/工具/配置变更

### 提交格式

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### 示例

```bash
# 新功能
git commit -m "feat(validator): add dynamic import detection"

# Bug 修复
git commit -m "fix(slurm): handle timeout error correctly"

# 文档
git commit -m "docs(readme): update installation instructions"

# 重构
git commit -m "refactor(orchestrator): extract scene validation logic"
```

## Pull Request 流程

### 1. Fork 和分支

```bash
# Fork 仓库后
git clone https://github.com/YOUR_USERNAME/kd1-anime.git
cd kd1-anime
git checkout -b feature/your-feature-name
```

### 2. 开发和测试

```bash
# 进行修改...

# 运行检查
ruff check .
pytest -q

# 确保编译通过
python -m compileall -q .
```

### 3. 提交 PR

- 确保 PR 描述清晰说明修改内容
- 关联相关 Issue（如有）
- 确保 CI 通过

### 4. 代码审查

- 响应审查意见
- 根据反馈进行修改
- 保持 PR 最新（如有冲突）

## 测试指南

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_validator.py

# 运行特定测试
pytest tests/test_validator.py::test_validate_manim_code

# 显示覆盖率
pytest --cov=kd1_anime
```

### 编写测试

- 每个新功能都应有对应测试
- 测试文件放在 `tests/` 目录
- 使用 `pytest` 的 fixtures 和 markers

```python
import pytest
from kd1_anime.agents.validator import validate_manim_code

def test_valid_code():
    code = '''
from manim import *

class TestScene(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
'''
    result = validate_manim_code(code)
    assert result.is_valid

def test_invalid_import():
    code = '''
import os
from manim import *

class TestScene(Scene):
    def construct(self):
        pass
'''
    result = validate_manim_code(code)
    assert not result.is_valid
    assert "禁止导入模块 'os'" in result.feedback
```

### Mock 外部依赖

测试中 mock 所有外部依赖（LLM、Slurm、FFmpeg）：

```python
from unittest.mock import patch, MagicMock

@patch("kd1_anime.agents.base.BaseAgent.call_llm")
def test_planner(mock_call_llm):
    mock_call_llm.return_value = '{"items": [...]}'
    # 测试逻辑...
```

## 文档贡献

### 文档结构

- `README.md`：项目介绍和快速开始
- `ARCHITECTURE.md`：架构设计文档
- `AGENTS.md`：AI agent 开发指南
- `CONTRIBUTING.md`：本文档
- `CHANGELOG.md`：版本变更记录

### 文档规范

- 使用中文编写主要文档
- 提供代码示例
- 保持简洁清晰

## 问题反馈

### 报告 Bug

使用 [Issue 模板](https://github.com/Enthusjast/kd1-anime/issues/new?template=bug_report.md) 报告 Bug：

1. 描述问题现象
2. 提供复现步骤
3. 包含错误日志
4. 说明环境信息

### 功能请求

使用 [Feature Request 模板](https://github.com/Enthusjast/kd1-anime/issues/new?template=feature_request.md) 提交功能建议：

1. 描述使用场景
2. 说明期望行为
3. 提供替代方案

## 行为准则

- 尊重所有参与者
- 接受建设性批评
- 专注于对社区最有利的事情
- 对他人表示同理心

## 许可证

贡献即表示你同意你的贡献将在 [MIT 许可证](LICENSE) 下发布。

---

如有任何问题，请在 [GitHub Discussions](https://github.com/Enthusjast/kd1-anime/discussions) 中提问。
