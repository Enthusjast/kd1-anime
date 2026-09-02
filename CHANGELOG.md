# Changelog

本文件记录 kd1-anime 的重要用户可见变更，格式参考
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，版本号遵循
[语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 新增

- TechnicalSpec 生命周期归一化和确定性编译增强：
  - 从 active 对象、依赖关系和事件说明中安全推断缺失的 Transform source；
  - 将 Transform 中误列出的新对象拆分为独立的 introducer 事件；
  - 为边界必须保留的对象区分原地 Transform 与 ReplacementTransform；
  - 支持明确的“退出后重新创建”生命周期。
- 计划审查的最小上下文重试和截断兜底：保留确定性问题时继续有限重规划，
  确定性检查通过但语义响应被截断时以 warning 继续，不把网络/输出截断误报为结构错误。
- 生成、配置、故障排查和贡献文档，统一说明当前 CLI、manifest v6、RAG、
  视觉评估、OpenGL/Cairo 和 dry-run 行为。

### 改进

- 细化计划审查、连续性审查和代码审查的职责边界；实现层的数学/连续性错误
  优先留在 Coder 修复，只有证据明确指向计划本身时才回到 Planner 或 Continuity。
- 统一 required 导出对象的活动身份，避免用变量重绑定替代 Manim 的
  FadeIn、Transform 或 ReplacementTransform，降低跨场景交接失败率。
- 将时间线中的显式淡出/清空动作同步到元素移除合同，防止已退出对象被错误要求
  在下一场景继续继承。
- 连续性审查达到修正上限后沿用已通过确定性检查的计划并记录 warning；
  恢复运行最多自动重查一次，避免连续性审查死循环。
- 高风险几何方案耗尽有限审查/修复预算后可切换保守教学方案，并清理受影响的
  下游交接状态。
- 扩展 TechnicalSpec、生命周期和计划编译测试，覆盖复合 Mobject、辅助方法、
  alias 生命周期、已退出对象重新创建和代码 finding 路由。
- sdist 纳入配置、故障排查和贡献文档，确保发布归档与仓库文档一致。

### 修复

- 修复缺少 Transform source、目标对象同时被标记为 create、边界对象被错误
  ReplacementTransform 移除等模型输出导致的确定性合同失败。
- 修复 Reviewer 把带有代码级证据的实现错误错误升级为计划重规划的问题。
- 修复连续性机械修复与计划重启互相触发、最后一个场景产生悬空 handoff、
  上游已移除元素被重新提升为 keep 等重复循环。
- 修复 required 导出变量与 initial/base 别名混用导致的未 active 对象操作。
- 补强计划/代码审查在 JSON 截断、模型输出矛盾和重复反馈情况下的有界行为。
- 引入 Review 分级策略：确定性或高置信度、有证据的核心错误才阻断；风格、一般节奏和不确定模型意见记录为 warning；唯一局部修复先校验后复审。

## [0.4.0] - 2026-08-01

### 新增

- 增量渲染：
  - 通过 generate --incremental 基于已有运行复用身份完全一致的场景视频；
  - 校验代码、Render Profile、视频哈希、场景 ID、类名和环境凭据后才允许复用；
  - 复用视频会复制到新 run 的私有目录，不创建伪造 Job ID。
- 批量处理：
  - 通过 batch 从文本或 JSON prompts 文件读取多个任务；
  - 支持项目级并行、dry-run 和 output-dir；
  - 多个任务共享 LLM、RAG、视觉评估和 Slurm 配额，并在执行前检查输出冲突。
- 多维评估模块：确定性代码/效率评估、视觉评估、运行对比和 JSON 报告。
- 独立视觉质量门：
  - 使用单独的多模态端点分析场景关键帧；
  - 检查数学正确性、相关性、可读性、布局和跨帧一致性；
  - 支持视觉修复、最佳候选恢复和成片报告。
- RAG 知识检索：
  - 使用 SQLite 保存文档分块和 Embedding；
  - 使用独立 Embedding 与 Reranker 服务；
  - 提供 rag index、rag status 和 rag search 命令。
- 环境能力探针和 doctor 命令，能够验证 FFmpeg、XeLaTeX、CJK、MathTex
  和当前 renderer 的真实最小视频产物。
- 运行级产物凭据、语义状态账本、相邻边界关键帧和脱敏事件日志。
- 默认启用的私有 SQLite LLM 响应缓存及 cache status、cache clear 命令。
- 计划批准闸门、Renderer 能力上下文、分阶段 token 预算和有界 Prompt 构造。

### 改进

- 安装器默认锁定 Manim Community Edition 0.20.1，安装最小 XeLaTeX/CJK
  依赖，并将文档和示例解压到 ~/.kd1-anime/knowledge。
- 默认用户数据统一位于 ~/.kd1-anime，旧配置以非破坏方式迁移。
- Slurm 监控区分 COMPLETED、GONE 和 UNKNOWN，按当前 Job 的媒体目录递归定位
  最终 MP4，并处理共享文件系统传播延迟、抢占回退和超时取消。
- 每次提交使用独立媒体目录；视频仅在 ffprobe、哈希、分辨率和帧率验证通过后
  才被视为成功产物。
- 多场景合并改为 FFmpeg xfade/acrossfade，并通过临时文件和原子替换生成最终视频。
- 生成代码经过 AST、导入白名单、Manim Scene 结构、XeLaTeX、renderer API
  和生命周期校验；AutoFix 后必须重新审查。
- Plan Review 在编码前检查数学、定义域、时间线、几何可实现性和元素交接；
  代码审查只检查已确认计划的实现。
- 启动或进入生成流程前可探测主模型；启用 RAG 时探测 Embedding/Reranker；
  视觉端点使用独立配置，网络故障可安全记录为 unknown。
- Rich 仪表盘区分分镜、编码、审查、渲染和视觉阶段；不再把仅完成分镜的场景
  提前显示为最终完成。
- 批量任务共享资源配额，避免每个项目独立放大并发。
- Manifest 升级为 v6；v4/v5 可只读查看，旧清单不猜测迁移或继续写回。

## [0.3.0] - 2026-07-29

### 新增

- 完整的两阶段规划系统（outline → detail）。
- 多 Agent 并行处理、Slurm 集群渲染和失败场景自动修复。
- FFmpeg 视频合并、Apptainer 容器隔离、原子 manifest 和中断恢复。

### 改进

- 优化 LLM 调用重试和错误反馈。
- 改进 Slurm 作业监控。
- 增强生成代码 AST 安全校验。

## [0.2.0] - 2026-07-20

### 新增

- Rich + prompt_toolkit 终端交互界面。
- 需求澄清对话、场景代码生成和代码审查机制。

### 改进

- 优化配置加载和错误处理。

## [0.1.0] - 2026-07-15

### 新增

- 初始 CLI、配置管理和 Manim 代码生成能力。

## 版本说明

- 主版本号：不兼容的 API 修改。
- 次版本号：向下兼容的功能新增。
- 修订号：向下兼容的问题修正。

## 相关链接

- [GitHub Releases](https://github.com/Enthusjast/kd1-anime/releases)
- [项目主页](https://github.com/Enthusjast/kd1-anime)
