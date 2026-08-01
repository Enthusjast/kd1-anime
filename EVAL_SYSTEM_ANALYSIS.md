# TheoremExplainAgent 评估系统详解

## 一、系统架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    evaluate.py (主入口)                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  text_utils  │  │ video_utils  │  │ image_utils  │          │
│  │  文本评估     │  │ 视频评估     │  │ 图像评估     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                │                  │                   │
│         ▼                ▼                  ▼                   │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              prompts_raw (评估提示词)                  │      │
│  │  - _text_eval_new    (文本评估提示)                    │      │
│  │  - _video_eval_new   (视频评估提示)                    │      │
│  │  - _image_eval       (图像评估提示)                    │      │
│  │  - _fix_transcript   (转录修复提示)                    │      │
│  └──────────────────────────────────────────────────────┘      │
│         │                │                  │                   │
│         ▼                ▼                  ▼                   │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              mllm_tools (LLM 接口层)                  │      │
│  │  - LiteLLMWrapper   (OpenAI 兼容)                    │      │
│  │  - GeminiWrapper    (Gemini 原生)                     │      │
│  │  - VertexAIWrapper  (Vertex AI)                      │      │
│  └──────────────────────────────────────────────────────┘      │
│         │                │                  │                   │
│         ▼                ▼                  ▼                   │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              utils (工具函数)                          │      │
│  │  - extract_json           (JSON 提取)                 │      │
│  │  - convert_score_fields   (分数转换)                  │      │
│  │  - calculate_geometric_mean (几何平均)                │      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、三大评估维度详解

### 2.1 文本评估 (Text Evaluation)

#### 评估目标
分析视频转录文本的内容质量，不涉及视觉部分。

#### 评估标准

| 维度 | 说明 | 评分范围 |
|------|------|----------|
| **Accuracy and Depth** | 定理讲解是否准确？是否提供了直观和/或严格的解释？ | 1-5 |
| **Logical Flow** | 视频是否遵循清晰的逻辑结构？是否呈现连贯的观点递进？ | 1-5 |

#### 评分标准
- **1**: 非常差，完全不符合标准
- **2**: 低于平均水平，存在显著问题
- **3**: 可接受，基本符合标准，有小问题
- **4**: 良好，表现良好，无重大问题
- **5**: 优秀，完全符合或超出预期

#### 提示词模板
```python
_text_eval_new = """You are a specialist in evaluating theorem explanation videos...
### Evaluation Criteria
1. **Accuracy and Depth**
    - Does the narration explain the theorem accurately?
    - Does the video provide intuitive and/or rigorous explanations for why the theorem holds?
2. **Logical Flow**
    - Does the video follow a clear and logical structure?
    - Does the video present a coherent buildup of ideas?
...
"""
```

#### 输出格式
```json
{
  "overall_analysis": "整体分析文本",
  "evaluation": {
    "accuracy_and_depth": {
      "comprehensive_evaluation": "准确性与深度分析",
      "score": 4
    },
    "logical_flow": {
      "comprehensive_evaluation": "逻辑流畅性分析",
      "score": 5
    }
  }
}
```

---

### 2.2 视频评估 (Video Evaluation)

#### 评估目标
分析视频帧的视觉一致性和动画质量。

#### 评估流程
```
原始视频 → 分块(10块) → 降低帧率(可选) → 逐块评估 → 汇总分数
```

#### 评估标准

| 维度 | 说明 | 评分范围 |
|------|------|----------|
| **Visual Consistency** | 风格一致性 + 运动流畅性 | 1-5 |

#### 详细子维度
- **Style Consistency**: 视觉风格在帧间是否保持一致？
- **Smoothness**: 动作和过渡是否平滑？

#### 技术实现
```python
def evaluate_video_chunk_new(model, video_path, ...):
    # 1. 降低帧率以减少 token 消耗
    if target_fps is not None:
        processed_video_path = reduce_video_framerate(video_path, target_fps=target_fps)
    
    # 2. 构造提示词
    prompt = _video_eval_new.format(description=description)
    
    # 3. 准备多模态输入 (文本 + 视频)
    inputs = _prepare_text_video_inputs(prompt, video_to_use)
    
    # 4. 调用模型评估
    response = model(inputs)
    
    # 5. 提取并转换分数
    response_json = extract_json(response)
    response_json = convert_score_fields(response_json)
    
    return response_json
```

#### 帧率降低处理
```python
def reduce_video_framerate(input_path, target_fps=1, output_path=None):
    """
    通过只保留目标间隔的帧来降低视频帧率。
    
    Args:
        input_path: 输入视频路径
        target_fps: 目标帧率 (默认 1fps)
        
    Returns:
        处理后的视频路径
    """
    cap = cv2.VideoCapture(input_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / target_fps)
    
    # 只写入间隔帧
    while cap.isOpened():
        ret, frame = cap.read()
        if frame_count % frame_interval == 0:
            out.write(frame)
```

---

### 2.3 图像评估 (Image Evaluation)

#### 评估目标
从视频中采样关键帧，评估视觉元素的布局和相关性。

#### 评估流程
```
原始视频 → 提取关键帧(10帧) → 逐帧评估 → 计算几何平均分
```

#### 评估标准

| 维度 | 说明 | 评分范围 |
|------|------|----------|
| **Visual Relevance** | 视频帧是否与定理概念和推导一致？ | 1-5 |
| **Element Layout** | 视觉元素的布局、大小、重叠、清晰度 | 1-5 |

#### 关键帧提取算法
```python
def extract_key_frames(video_path, output_dir, num_chunks=10):
    """
    将视频分成 N 个块，从每个块中选择"非黑色空间最多"的帧作为关键帧。
    """
    clip = VideoFileClip(video_path)
    frames = list(clip.iter_frames(fps=1))  # 每秒一帧
    
    # 按块处理
    for i in range(num_chunks):
        chunk_frames = frames[start_idx:end_idx]
        # 选择非黑色空间最多的帧
        output_path = os.path.join(output_dir, f"key_frame_{i+1}.jpg")
        result = image_with_most_non_black_space(chunk_frames, output_path)
```

#### 评估逻辑
```python
def evaluate_sampled_images(model, video_path, description, num_chunks=10):
    # 1. 提取关键帧
    key_frames = extract_key_frames(video_path, temp_dir, num_chunks)
    
    # 2. 逐帧评估
    responses = []
    for key_frame in key_frames:
        prompt = _image_eval.format(description=description)
        inputs = _prepare_text_image_inputs(prompt, key_frame)
        response = model(inputs)
        responses.append(extract_json(response))
    
    # 3. 计算几何平均分
    for key, scores in scores_dict.items():
        res_score[key] = {"score": calculate_geometric_mean(scores)}
    
    return {
        "evaluation": res_score,
        "image_chunks": responses  # 保留每帧的详细评估
    }
```

---

## 三、核心工具函数

### 3.1 JSON 提取
```python
def extract_json(response: str) -> dict:
    """
    从 LLM 响应中提取 JSON，支持:
    - 纯 JSON 字符串
    - ```json ... ``` 代码块
    - ``` ... ``` 通用代码块
    """
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # 尝试从代码块提取
        match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
        if not match:
            match = re.search(r'```\n(.*?)\n```', response, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError("Failed to extract valid JSON content")
```

### 3.2 分数字段转换
```python
def convert_score_fields(data: dict) -> dict:
    """
    递归转换字典中的 "score" 字段为整数。
    处理 LLM 可能返回字符串分数的情况。
    """
    for key, value in data.items():
        if key == "score":
            if isinstance(value, str) and value.isdigit():
                converted_data[key] = int(value)
        elif isinstance(value, dict):
            converted_data[key] = convert_score_fields(value)
    return converted_data
```

### 3.3 几何平均分计算
```python
def calculate_geometric_mean(scores: List[int]) -> float:
    """
    计算分数的几何平均值。
    相比算术平均，几何平均对低分更敏感，鼓励全面提升。
    """
    scores = [s for s in scores if s is not None]
    if not scores:
        return 0.0
    product = prod(scores)
    return product ** (1 / len(scores))
```

---

## 四、多模态输入处理

### 4.1 输入格式统一
```python
# 文本输入
_prepare_text_inputs(texts) → [{"type": "text", "content": "..."}]

# 文本 + 图像输入
_prepare_text_image_inputs(texts, images) → [
    {"type": "text", "content": "..."},
    {"type": "image", "content": "path_or_pil_image"}
]

# 文本 + 视频输入
_prepare_text_video_inputs(texts, videos) → [
    {"type": "text", "content": "..."},
    {"type": "video", "content": "video_path"}
]
```

### 4.2 模型适配
```python
# Gemini/Vertex AI: 原生支持视频输入
if model_name.startswith('gemini/') or model_name.startswith('vertex_ai/'):
    return [{"type": "video", "content": video_path}]

# 其他模型: 转换为图像序列
else:
    frames = extract_frames(video_path)
    return [{"type": "image", "content": frame} for frame in frames]
```

---

## 五、评估流程示例

### 5.1 单文件评估
```bash
python evaluate.py \
    --file_path ./output/theorem_abc/video.mp4 \
    --output_folder ./eval_results \
    --eval_type all \
    --model_text azure/gpt-4o \
    --model_video gemini/gemini-1.5-pro-002 \
    --model_image azure/gpt-4o \
    --retry_limit 3
```

### 5.2 批量评估
```bash
python evaluate.py \
    --file_path ./output/ \
    --output_folder ./eval_results \
    --bulk_evaluate \
    --combine \
    --max_workers 4
```

### 5.3 评估结果结构
```json
{
  "theorem_abc": {
    "text_evaluation": {
      "overall_analysis": "...",
      "evaluation": {
        "accuracy_and_depth": {"comprehensive_evaluation": "...", "score": 4},
        "logical_flow": {"comprehensive_evaluation": "...", "score": 5}
      }
    },
    "video_evaluation": {
      "overall_analysis": "...",
      "evaluation": {
        "visual_consistency": {"comprehensive_evaluation": "...", "score": 4}
      }
    },
    "image_evaluation": {
      "evaluation": {
        "visual_relevance": {"score": 4.2},
        "element_layout": {"score": 3.8}
      },
      "image_chunks": [...]
    }
  }
}
```

---

## 六、设计亮点总结

### 6.1 多维度覆盖
- **文本维度**: 准确性、逻辑性
- **视频维度**: 一致性、流畅性
- **图像维度**: 相关性、布局质量

### 6.2 模型灵活性
- 支持多种 LLM 后端 (OpenAI, Gemini, Vertex AI)
- 文本评估用文本模型，视频评估用多模态模型
- 可独立配置各维度使用的模型

### 6.3 鲁棒性设计
- 重试机制 (`retry_limit`)
- JSON 提取容错
- 分数类型自动转换

### 6.4 性能优化
- 视频帧率降低减少 token 消耗
- 关键帧选择算法 (非黑色空间最大化)
- 并行处理支持 (`max_workers`)

### 6.5 结果可追溯
- 保存每帧/每块的详细评估
- 支持合并结果导出
- 时间戳标记

---

## 七、对 kd1-anime 的借鉴建议

### 7.1 可直接复用的设计
1. **评估维度划分** - 代码质量、视觉效果、生成效率
2. **几何平均分** - 对低分敏感，鼓励全面提升
3. **JSON 输出格式** - 结构化评估结果
4. **重试机制** - 提高评估稳定性

### 7.2 需要适配的部分
1. **评估提示词** - 针对数学动画定制
2. **视觉评估** - 适配 Slurm 渲染输出
3. **代码评估** - 集成 AST 分析

### 7.3 可扩展的功能
1. **A/B 对比评估** - 比较不同版本质量
2. **回归测试** - 检测质量下降
3. **自动基准** - 建立质量基线
