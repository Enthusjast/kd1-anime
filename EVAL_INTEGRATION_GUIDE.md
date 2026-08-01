# 自评估-自改进循环使用指南

## 概述

kd1-anime 现已集成自动评估-改进循环功能，参照 TheoremExplainAgent 的评估系统设计。该功能可以在视频生成完成后自动评估质量，并在质量不达标时自动触发代码改进和重新渲染。

## 工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                         生成流程                                  │
├─────────────────────────────────────────────────────────────────┤
│  INIT → PLANNING → DETAILING → CODING → REVIEWING               │
│           → DISPATCHING → MONITORING → FIXING → MERGING         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    评估循环 (可选)                        │   │
│  │  MERGING → EVALUATING ─┬─→ (分数 ≥ 阈值) → DONE          │   │
│  │                        │                                 │   │
│  │                        └─→ (分数 < 阈值) → CODING → ...  │   │
│  │                              (最多 MAX_EVAL_ROUNDS 轮)   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 配置说明

在 `.env` 文件中添加以下配置：

```bash
# 启用自动评估-改进循环
ENABLE_AUTO_EVAL=true

# 评估通过阈值 (1-5分，低于此分数触发改进)
EVAL_THRESHOLD=3.5

# 最大评估-改进轮数 (0表示禁用)
MAX_EVAL_ROUNDS=2

# 启用视觉效果评估 (需要多模态LLM支持)
ENABLE_VISUAL_EVAL=false

# 视觉评估使用的模型 (可选，默认使用 LLM_MODEL)
# EVAL_VISUAL_MODEL=gpt-4o
```

## 使用方式

### 1. 命令行方式

```bash
# 启用自动评估生成
kd1-anime generate "勾股定理的证明" --enable-eval

# 自定义评估参数
kd1-anime generate "勾股定理的证明" \
  --enable-eval \
  --eval-threshold 4.0 \
  --max-eval-rounds 3
```

### 2. TUI 交互方式

在 TUI 中输入需求后，系统会根据配置自动执行评估循环：

```
╭──────────────────────────────────────────────╮
│  kd1-anime - AI 数学动画生成器                 │
╰──────────────────────────────────────────────┘

请输入你的动画需求: 勾股定理的证明

────── 场景概要 ──────
...

────── 视频拼接 ──────
  ✓ 输出: workspace/runs/run-xxx/output_final.mp4 (15.2 MB)

质量评估完成
  综合分数: 3.20/5.00 (阈值: 3.5)
  代码质量: 3.50/5.00
  视觉效果: 2.90/5.00

质量分数 3.20 低于阈值 3.5，触发自动改进...
改进模式: 重新生成场景 [2, 3]

────── 代码生成 ──────
  ▸ Scene 2: 勾股定理几何证明 开始生成代码
  ✓ workspace/runs/run-xxx/scenes/scene_2.py
  ▸ Scene 3: 代数验证 开始生成代码
  ✓ workspace/runs/run-xxx/scenes/scene_3.py

...

质量评估完成
  综合分数: 4.10/5.00 (阈值: 3.5)
  代码质量: 4.30/5.00
  视觉效果: 3.90/5.00

✓ 质量评估通过 (分数: 4.10)
```

### 3. 编程方式

```python
from kd1_anime.orchestrator import Orchestrator

# 创建编排器
orchestrator = Orchestrator()

# 运行生成（自动评估由配置控制）
final_video = orchestrator.run(
    "勾股定理的证明",
    callback=my_callback,
)

# 或者手动评估已有运行
from kd1_anime.eval import Evaluator

evaluator = Evaluator()
result = evaluator.evaluate_run("run-123")
print(f"总分: {result.overall_score:.2f}")

# 对比两个运行
comparison = evaluator.compare_runs("run-100", "run-123")
print(f"改进: {comparison.improvements}")
print(f"退化: {comparison.regressions}")
```

## 评估维度

### 代码质量评估 (自动)

| 指标 | 说明 | 评分标准 |
|------|------|----------|
| `code_syntax` | 语法正确性 | 1-5分 |
| `code_security` | 安全性检查 | 1-5分 |
| `code_complexity` | 复杂度分析 | 1-5分 |
| `code_style` | 代码风格 | 1-5分 |

### 视觉效果评估 (可选)

| 指标 | 说明 | 评分标准 |
|------|------|----------|
| `visual_relevance` | 视觉相关性 | 1-5分 |
| `visual_quality` | 视觉质量 | 1-5分 |
| `visual_consistency` | 视觉一致性 | 1-5分 |
| `element_layout` | 元素布局 | 1-5分 |

### 生成效率评估 (自动)

| 指标 | 说明 | 评分标准 |
|------|------|----------|
| `render_time` | 渲染时间 | 1-5分 |
| `success_rate` | 成功率 | 1-5分 |
| `retry_count` | 重试次数 | 1-5分 |

## 评估结果

评估结果保存在运行目录下的 `eval_result.json` 文件中：

```json
{
  "run_id": "run-20260801_120000",
  "timestamp": "2026-08-01T12:30:00",
  "overall_score": 4.10,
  "summary": "Auto-evaluation: 4.10/5.00",
  "scores": [
    {
      "metric": "code_syntax",
      "score": 5,
      "level": "excellent",
      "justification": "Code has valid syntax"
    },
    {
      "metric": "code_security",
      "score": 5,
      "level": "excellent",
      "justification": "Found 0 security issues"
    },
    ...
  ]
}
```

## 手动评估命令

```bash
# 评估已有运行
kd1-anime evaluate run-20260801_120000

# 评估代码文件
kd1-anime evaluate --code-file scene.py

# 评估截图
kd1-anime evaluate --image screenshot.png --desc "勾股定理"

# 对比两个运行
kd1-anime evaluate run-123 --compare run-100

# 输出 JSON 报告
kd1-anime evaluate run-123 --json --output report.json
```

## 最佳实践

1. **合理设置阈值**
   - 初始建议使用 3.5 作为阈值
   - 过高会导致不必要的改进循环
   - 过低会放过质量问题

2. **控制最大轮数**
   - 建议设置为 2-3 轮
   - 过多轮数会增加生成时间和成本

3. **选择性启用视觉评估**
   - 视觉评估需要多模态 LLM 支持
   - 会增加 API 调用成本
   - 建议在高质量需求时启用

4. **监控评估日志**
   - 查看 `eval_result.json` 了解详细评分
   - 分析低分原因，优化提示词

## 故障排除

### 评估一直不通过

1. 检查阈值设置是否过高
2. 查看 `eval_result.json` 找出低分指标
3. 优化动画描述，提供更明确的指导
4. 增加 `MAX_EVAL_ROUNDS` 允许更多改进

### 评估超时或失败

1. 检查 LLM API 配置
2. 确保网络连接正常
3. 检查 `ENABLE_VISUAL_EVAL` 配置
4. 查看日志获取详细错误信息

### 改进后分数反而下降

1. 代码改进可能引入新问题
2. 考虑降低 `MAX_EVAL_ROUNDS`
3. 检查改进场景的选择逻辑
4. 手动审查改进后的代码
