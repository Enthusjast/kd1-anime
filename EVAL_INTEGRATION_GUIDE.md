# 自动评估与改进指南

kd1-anime 提供两条相互独立的质量链路：

1. **逐场景视觉质量门**：渲染成功后、合并前，用独立多模态 LLM 检查当前场景视频，并可触发有界 Coder 修复。
2. **确定性自动评估**：合并后评估代码质量和渲染效率；只有低分能归因到具体场景代码时才触发改进。

视觉模型只输出诊断和结构化分数，不直接生成或执行 Python 代码。Planner、Coder、Reviewer 与 AutoFixer 始终使用主 LLM 配置。
`ENABLE_VISUAL_EVAL` 与 `ENABLE_AUTO_EVAL` 可独立开启：前者控制场景视觉闭环，后者控制合并后的确定性代码/效率改进循环。

## 流程

```text
场景渲染 → VISUAL_EVALUATING
              ├─ 通过 → MERGING
              ├─ API/抽帧失败 → unknown → MERGING
              ├─ 低分 → CODING → REVIEWING → RENDERING（有界）
              └─ 修复耗尽 → 恢复最佳候选/记录 warning → MERGING

MERGING → 成片视觉报告（只诊断） → EVALUATING（代码/效率） → DONE
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

# 启用逐场景视觉质量门与成片报告
ENABLE_VISUAL_EVAL=true

# 独立多模态端点；不会继承 LLM_API_KEY/BASE_URL/MODEL
VISUAL_LLM_API_KEY=your-visual-api-key
VISUAL_LLM_BASE_URL=https://your-visual-endpoint/v1
VISUAL_LLM_MODEL=your-multimodal-model

# 每段视频关键帧数、通过阈值、每场景最大视觉修复次数
VISUAL_EVAL_FRAME_COUNT=6
VISUAL_EVAL_THRESHOLD=3.5
MAX_VISUAL_FIX_ATTEMPTS=2
VISUAL_LLM_PARALLEL_WORKERS=2
```

启用前可运行 `kd1-anime doctor --probe-visual-llm`，确认 OpenAI-compatible 端点实际接受 `image_url` 消息。视觉 API 运行时故障会记录为 `unknown`，不会伪造低分或删除已经成功渲染的视频；配置缺失则在启动前直接报错。

## 使用

```bash
# 临时为本次命令启用自动评估（也可写入 .env）
ENABLE_AUTO_EVAL=true kd1-anime generate "勾股定理的证明"

# 评估已有运行；视觉请求使用独立端点
kd1-anime evaluate 20260801-120000-1234abcd

# 只评估清单中某个场景的精确 SceneArtifact
kd1-anime evaluate 20260801-120000-1234abcd --scene-id 3

# 只运行确定性代码/效率指标
kd1-anime evaluate 20260801-120000-1234abcd --no-visual

# 评估单个代码文件或截图；--image 会自动启用视觉评估
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
- `visual_math_accuracy`
- `visual_quality`
- `visual_consistency`
- `element_layout`

系统在 5%–95% 时间范围内抽取有序关键帧，记录帧 ID、时间戳和 SHA-256，并在一次请求中联合评价整段序列，不把单帧分数伪装成完整视频评价。模型返回的问题必须引用实际存在的帧 ID，否则整次结果按无效处理。

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
~/.kd1-anime/workspace/runs/<run-id>/eval_result.json
~/.kd1-anime/workspace/runs/<run-id>/eval_frames/              # 关键帧及其哈希
~/.kd1-anime/workspace/runs/<run-id>/eval_reports/scene_<n>/   # 场景每轮视觉报告
~/.kd1-anime/workspace/runs/<run-id>/eval_reports/final_visual.json
~/.kd1-anime/workspace/runs/<run-id>/visual_candidates/        # 可恢复的候选代码
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

- **视觉评分 unknown**：运行 `doctor --probe-visual-llm`，检查场景报告中的 `error`、FFmpeg 和视频哈希。
- **一直低于阈值**：查看逐帧 evidence；达到 `MAX_VISUAL_FIX_ATTEMPTS` 后系统会停止重写并保留更好的候选。
- **成本过高**：关闭 `ENABLE_VISUAL_EVAL`，或降低 `VISUAL_EVAL_FRAME_COUNT` / `MAX_VISUAL_FIX_ATTEMPTS`；这不会关闭确定性评估。
- **对比显示 unknown**：任一运行没有有效指标时，差值也为 unknown，不应解释为 0。
