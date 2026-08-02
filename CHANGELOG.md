# Changelog

本文件记录 kd1-anime 项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

### 新增
- **异常层次结构**：定义项目自定义异常类型，提供更细粒度的错误处理
  - `KD1Error`：所有项目异常的基类
  - `LLMError`/`LLMAuthError`/`LLMRateLimitError`/`LLMTimeoutError`/`LLMResponseError`：LLM 相关异常
  - `SlurmError`/`SlurmSubmitError`/`SlurmCancelError`/`SlurmTimeoutError`：Slurm 相关异常
  - `RenderError`/`RenderTimeoutError`/`RenderOOMError`：渲染相关异常
  - `ValidationError`/`CodeValidationError`/`ReviewError`：校验相关异常
  - `MediaError`/`FFmpegError`/`VideoNotFoundError`/`MergeError`：媒体处理异常
  - `RunError`/`RunNotFoundError`/`RunLockError`/`RunIntegrityError`：运行管理异常
  - `PipelineError`/`PipelineAbortedError`/`PipelineExhaustedError`：流水线异常
  - `ConfigError`：配置相关异常

- **结构化日志模块**：新增 `logging.py` 模块
  - 支持按模块/级别过滤
  - Rich 格式化输出
  - 可选文件日志
  - `AgentLogger` 类提供与原有 console.print 兼容的接口

- **doctor 命令**：新增 `kd1-anime doctor` 命令
  - 检查 Python 版本
  - 检查 conda 安装
  - 检查 manim 安装
  - 检查 ffmpeg 安装
  - 检查 Slurm (sbatch) 安装
  - 检查 xelatex 安装
  - 检查 apptainer 安装（可选）
  - 检查 LLM 配置

- **测试覆盖扩展**：
  - `test_coder.py`：CoderAgent 测试（代码生成、代码块提取、错误处理）
  - `test_auto_fixer.py`：AutoFixerAgent 测试（错误分类、修复逻辑、边界情况）
  - `test_planner.py`：PlannerAgent 测试（场景规划、outline/detail 生成）

- **文档完善**：
  - `CONTRIBUTING.md`：贡献指南（开发环境、代码规范、提交规范、PR 流程）
  - `CHANGELOG.md`：版本变更记录（本文件）

### 改进

- **AST 校验增强**：
  - 新增 `BANNED_IMPORT_MODULES` 集合，禁止更多危险模块（os、sys、subprocess、logging 等）
  - 增强对动态构造危险调用的检测（如 `getattr(os, "system")`）
  - 改进导入检查逻辑，跟踪已导入模块

- **异常处理改进**：
  - `orchestrator.py`：使用自定义异常替代宽泛的 `except Exception`
  - 添加更细粒度的异常捕获（LLMError、SlurmError 等）
  - 改进错误上下文信息

- **代码质量**：
  - 使用 `logging` 替代部分 `console.print` 调用
  - 改进错误消息的可读性

## [0.3.0] - 2026-07-29

### 新增
- 完整的两阶段规划系统（outline → detail）
- 多 Agent 并行处理
- Slurm 集群渲染支持
- 自动修复失败场景
- FFmpeg 视频合并
- Apptainer 容器隔离支持
- 原子化 manifest 持久化
- 中断恢复机制

### 改进
- 优化 LLM 调用重试逻辑
- 改进 Slurm 作业监控
- 增强 AST 安全校验

## [0.2.0] - 2026-07-20

### 新增
- TUI 交互界面
- 需求澄清对话
- 场景代码生成
- 代码审查机制

### 改进
- 优化配置加载
- 改进错误处理

## [0.1.0] - 2026-07-15

### 新增
- 初始版本发布
- 基础 CLI 框架
- 配置管理系统
- Manim 代码生成

---

## 版本说明

- **主版本号**：不兼容的 API 修改
- **次版本号**：向下兼容的功能性新增
- **修订号**：向下兼容的问题修正

## 链接

- [GitHub Releases](https://github.com/Enthusjast/kd1-anime/releases)
- [完整变更历史](https://github.com/Enthusjast/kd1-anime/compare/v0.2.0...main)
