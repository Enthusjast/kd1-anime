# Changelog

本文件记录 kd1-anime 项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

### 新增
- **可验证渲染产物**：新增 RenderProfile、ffprobe 元数据、代码/配置/视频哈希绑定和 Manifest v2 向后迁移
- **多帧视觉评估**：均匀抽取最终视频关键帧，以一次严格结构化多模态请求联合评估
- **批量全局配额**：多个批量项目共享 LLM 与 Slurm 并发限制，并预检输出冲突
- **Renderer 能力上下文**：Planner、Coder、Reviewer 和 AutoFixer 使用一致的 Cairo/OpenGL 约束
- **作业尝试隔离**：每次 Slurm 提交使用独立媒体目录，防止旧 MP4 污染修复重试
- **最终输出凭据**：清单保存最终视频 SHA-256，FFmpeg 临时产物通过 ffprobe 后才原子替换

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

- Slurm 监控区分 GONE 与 UNKNOWN，验证嵌套 Manim 成品，并修复抢占回退后的超时计时
- AutoFix 后强制重新审查；危险属性别名、XeLaTeX `.xdv` 和 renderer API 由确定性校验兜底
- LLM 非空截断响应不再被直接消费；持续截断会返回明确错误
- 视觉评估失败记为 unknown 并从总分排除；运行对比聚合所有场景指标
- 恢复运行会补发已完成场景快照，并保守处理旧版或无法验证的产物

- **AST 校验增强**：
  - 收紧允许导入根模块集合，禁止 os、sys、subprocess 等危险模块
  - 增强对动态构造危险调用和危险属性别名的检测
  - 补充 OpenGL 相机、Mobject 继承和 XeLaTeX 配置的确定性检查

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

## [0.4.0] - 2026-08-01

### 新增

- **增量渲染功能**：
  - 添加 `--incremental` CLI 选项，支持基于上一次运行只渲染变化的场景
  - 添加 `run_incremental()` 方法到 Orchestrator
  - 修改 manifest 支持记录增量渲染信息（`incremental`, `base_run_id`）
  - 添加场景变化计算和视频复用逻辑
  - VideoMerger 支持混合使用新旧视频
  - 自动检测代码 hash 变化，跳过未变化的场景

- **批量并行处理功能**：
  - 添加 `batch` CLI 命令，支持从文件读取多个 prompt
  - 添加 `BatchProcessor` 类，支持并行执行多个动画项目
  - 支持纯文本和 JSON 格式的 prompts 文件
  - 可配置最大并行任务数（`--max-parallel`）
  - 支持批量 dry-run 模式
  - 生成批量处理摘要报告

- **新模块**：
  - `src/kd1_anime/batch.py` - 批量并行处理模块

- **新测试**：
  - `tests/test_batch.py` - 批量处理功能测试
  - `tests/test_incremental.py` - 增量渲染功能测试

### 改进

- **CLI 增强**：
  - `generate` 命令添加 `--incremental` 选项
  - 新增 `batch` 命令用于批量处理

- **Orchestrator 改进**：
  - PipelineContext 添加增量渲染支持字段
  - _checkpoint 方法记录增量渲染信息
  - _handle_merging 方法支持增量渲染视频合并

- **VideoMerger 改进**：
  - 添加 `collect_incremental_videos()` 方法
  - 支持从不同 run 目录收集视频


### 新增 - 评估系统

- **多维度评估模块** (`src/kd1_anime/eval/`)：
  - `metrics.py`: 评估指标和数据结构定义
    - `EvalMetric` 枚举：代码质量、视觉效果、生成效率指标
    - `QualityScore`: 质量评分数据类 (1-5分)
    - `EvalResult`: 评估结果，支持几何平均分计算
    - `ComparisonResult`: 运行对比结果
  - `prompts.py`: LLM 评估提示词模板
    - 代码质量评估提示词
    - 视觉效果评估提示词
    - 渲染结果分析提示词
    - 场景复杂度评估提示词
  - `code_eval.py`: 代码质量评估器
    - AST 语法分析
    - 安全性检查 (禁止模块、危险函数)
    - 复杂度计算
    - 风格检查
  - `visual_eval.py`: 视觉效果评估器
    - LLM 截图分析
    - 多帧合并评估
    - 视觉相关性、质量、一致性、布局评分
  - `evaluator.py`: 主评估器
    - 整合代码和视觉评估
    - 批量评估支持
    - 运行对比功能
    - 评估报告生成

- **CLI evaluate 命令**：
  - 评估完整运行: `kd1-anime evaluate <run-id>`
  - 评估代码: `kd1-anime evaluate --code "..."` 或 `--code-file scene.py`
  - 评估截图: `kd1-anime evaluate --image screenshot.png`
  - 对比运行: `kd1-anime evaluate <run-id> --compare <baseline-id>`
  - 支持 JSON 输出: `--json`
  - 支持保存报告: `--output report.json`

- **测试**：
  - `tests/test_eval.py`: 评估模块完整测试

### 新增 - 自评估-自改进循环

- **FSM 状态扩展**：
  - 新增 `EVALUATING` 状态，位于 `MERGING` 之后
  - 评估完成后根据分数决定是否进入改进循环

- **自动评估-改进流程**：
  ```
  MERGING → EVALUATING → (分数达标) → DONE
                      → (分数不达标) → CODING → ... → MERGING → EVALUATING
  ```
  - 最多循环 `MAX_EVAL_ROUNDS` 次
  - 只重新生成低分场景的代码
  - 保留高分场景的代码和渲染结果

- **新增配置项**：
  - `ENABLE_AUTO_EVAL`: 是否启用自动评估-改进循环 (默认 False)
  - `ENABLE_VISUAL_EVAL`: 是否启用视觉效果评估 (默认 False)
  - `EVAL_THRESHOLD`: 评估通过阈值 1-5 (默认 3.5)
  - `MAX_EVAL_ROUNDS`: 最大评估-改进轮数 (默认 2)
  - `EVAL_VISUAL_MODEL`: 视觉评估使用的模型

- **TUI 事件支持**：
  - `eval_complete`: 评估完成，显示分数
  - `eval_below_threshold`: 分数低于阈值，触发改进
  - `eval_improvement_mode`: 进入改进模式
  - `eval_passed`: 评估通过
  - `eval_max_rounds_reached`: 达到最大轮数

### 改进 - JSON 模式兼容性

- **添加 JSON 模式配置选项**：
  - `LLM_USE_JSON_MODE`: 控制是否使用 response_format=json_object (默认 True)
  - 某些 LLM 端点不支持此参数时可禁用

- **改进降级逻辑**：
  - 当端点返回空内容时，自动在系统提示中添加明确的 JSON 输出要求
  - 提高降级后的 JSON 输出成功率

- **添加 LLM 测试命令**：
  - `kd1-anime test-llm`: 测试 LLM 端点连接和 JSON 模式支持
  - 帮助用户诊断端点兼容性问题

### 改进 - TexTemplate 校验反馈

- **改进 validator.py**：
  - 在 TexTemplate 校验错误消息中添加修复提示
  - 引导 LLM 参考 coder.py 中的 TexTemplate 模板

- **改进 coder.py**：
  - 在核心要求中添加强制性的 TexTemplate 模板代码块
  - 强调 TexTemplate 是不可省略的强制要求
