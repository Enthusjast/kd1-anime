# 自动评估与改进指南

kd1-anime 可在视频合并后执行代码质量、渲染效率和可选视觉质量评估。评估报告用于诊断；只有低分能够归因到具体场景代码时，系统才会触发有界改进循环。

## 流程

```text
场景流水线 → MERGING → EVALUATING
                         ├─ 分数达到阈值 → DONE
                         ├─ 分数未知/问题不可归因 → DONE（保留 errors）
                         └─ 可归因场景低分 → CODING → REVIEWING → RENDERING
                                              （最多 MAX_EVAL_ROUNDS）
```

评估不会绕过正常安全门：重新生成的代码仍需确定性校验和 Reviewer 审查，渲染后生成新的产物凭据。

## 配置

```dotenv
# 合并后启用自动评估；默认关闭
ENABLE_AUTO_EVAL=true

# 1–5 分；低于阈值且可归因时触发改进
EVAL_THRESHOLD=3.5

# 最大自动改进轮数
MAX_EVAL_ROUNDS=2

# 可选：抽取最终视频关键帧并调用多模态模型
ENABLE_VISUAL_EVAL=false

# 留空则回退 LLM_MODEL
EVAL_VISUAL_MODEL=
```

启用视觉评估前，应确认 OpenAI-compatible 端点和模型支持 `image_url` 多模态消息。视觉评估会增加调用成本。

## 使用

```bash
# 临时为本次命令启用自动评估（也可写入 .env）
ENABLE_AUTO_EVAL=true kd1-anime generate "勾股定理的证明"

# 评估已有运行；默认启用视觉维度
kd1-anime evaluate 20260801-120000-1234abcd

# 只运行确定性代码/效率指标
kd1-anime evaluate 20260801-120000-1234abcd --no-visual

# 评估单个代码文件或截图
kd1-anime evaluate --code-file scene.py
kd1-anime evaluate --image screenshot.png --desc "勾股定理"

# 对比两个运行
kd1-anime evaluate 20260802-120000-deadbeef \
  --compare 20260801-120000-1234abcd

# 输出 JSON 报告
kd1-anime evaluate 20260801-120000-1234abcd --json --output report.json
```

编程接口：

```python
from kd1_anime.eval import Evaluator

evaluator = Evaluator(enable_visual_eval=False)
result = evaluator.evaluate_run("20260801-120000-1234abcd", enable_visual=False)

if result.overall_score is None:
    print("评分未知", result.errors)
else:
    print(f"总分: {result.overall_score:.2f}")
```

## 指标语义

### 代码质量（确定性）

- `code_syntax`
- `code_security`
- `code_complexity`
- `code_style`

### 视觉质量（可选多模态模型）

- `visual_relevance`
- `visual_quality`
- `visual_consistency`
- `element_layout`

系统从最终视频中均匀抽取多帧，并在一次请求中联合评价整段序列，不把单帧分数伪装成完整视频评价。

### 运行效率（确定性）

- `render_time`
- `success_rate`
- `retry_count`

重试次数为 0 时也会写入最佳重试分。运行对比会聚合同一指标在所有场景中的分数，而不是只比较第一个场景。

## unknown 与总分

端点不可用、ffmpeg/ffprobe 失败、视频缺失或视觉 JSON 不符合严格 schema 时：

1. 对应错误写入 `EvalResult.errors`；
2. 该维度标记为 unknown，不生成占位分数；
3. 总分只由实际存在的指标计算；完全没有有效指标时 `overall_score` 为 `null`；
4. unknown 不会自动触发代码重写。

这避免基础设施故障被误判为低质量动画。

## 报告

自动评估写入：

```text
workspace/runs/<run-id>/eval_result.json
workspace/runs/<run-id>/eval_frames/       # 启用视觉评估时
```

示例：

```json
{
  "run_id": "20260801-120000-1234abcd",
  "overall_score": 4.1,
  "scores": [
    {
      "metric": "code_syntax",
      "score": 5,
      "level": "excellent",
      "justification": "Code has valid syntax",
      "details": {}
    }
  ],
  "errors": {}
}
```

不要直接信任报告文件里手工编辑的 `overall_score`；加载时会根据实际指标重新计算。

## 排障

- **视觉评分 unknown**：确认模型支持图片输入，检查 `errors.visual`、FFmpeg 和最终视频。
- **一直低于阈值**：检查低分是否来自代码指标；渲染耗时或视觉问题未必能靠重写代码可靠解决。
- **成本过高**：关闭 `ENABLE_VISUAL_EVAL`，降低 `MAX_EVAL_ROUNDS`，先使用确定性指标。
- **对比显示 unknown**：任一运行没有有效指标时，差值也为 unknown，不应解释为 0。
