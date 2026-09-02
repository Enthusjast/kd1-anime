# 故障排查

建议先记录 run ID，再按下面的顺序排查。不要把 API Key、完整 .env 或含凭据的日志粘贴到 issue。

## 1. 先确认运行状态

    kd1-anime status
    kd1-anime status <run-id>
    kd1-anime status <run-id> --json

查看某个场景的日志：

    kd1-anime logs <run-id> --scene-id 2 --lines 160
    kd1-anime logs <run-id> --scene-id 2 --stderr --lines 160

status、logs、version、clean 不会调用 LLM，也不会自动扫描或恢复历史运行。

## 2. LLM API 不可用

### 现象

启动时提示 LLM API 不可用、配置不完整、鉴权失败、模型不存在或请求超时。

### 处理

    kd1-anime doctor
    kd1-anime doctor --probe-llm
    kd1-anime test-llm --verbose

检查：

1. LLM_API_KEY、LLM_BASE_URL、LLM_MODEL 是否都已填写；
2. Base URL 是否为 OpenAI-compatible 地址，通常包含 /v1；
3. 模型名是否是服务端实际支持的名称；
4. 当前 shell 是否读取了正确的配置文件。优先级是进程环境变量、当前目录 .env、用户配置；
5. 集群或登录节点是否需要代理。

不使用 HTTP 代理时：

    LLM_TRUST_ENV=false kd1-anime doctor --probe-llm

如果只是某些结构化请求失败，可以先测试不使用 JSON response format：

    kd1-anime test-llm --no-json-mode --verbose

程序会在进入 chat、plan 或 generate 前做短超时探测；探测失败会立即退出，不会先消耗多轮澄清或规划请求。

## 3. RAG 显示 degraded、索引过期或无法启动

### 区分两类情况

- 启动时缺少索引、索引过期或两个服务配置不完整：生成入口会直接拒绝继续；
- 运行中的一次检索请求失败：该次收据会标记 degraded，流水线尽可能继续，不会伪造检索结果。

### 处理

    kd1-anime rag status
    kd1-anime doctor --probe-rag
    kd1-anime rag index
    kd1-anime rag index --rebuild

检查：

1. RAG_ENABLED 是否真的为 true；
2. RAG_EMBEDDING_BASE_URL/MODEL/KEY 是否属于独立 Embedding 服务；
3. RAG_RERANK_BASE_URL/MODEL/KEY 是否属于独立 Reranker 服务；
4. RAG_DOCS_DIR 和 RAG_EXAMPLES_DIR 是否存在且包含 md、rst 或 py 文件；
5. 知识库文件、Embedding 模型或分块参数变化后是否重新建立索引；
6. 当前网络是否被代理变量影响。

不使用代理时：

    RAG_TRUST_ENV=false kd1-anime doctor --probe-rag

rag status 只读取本地状态，不联网；doctor --probe-rag 才会发送最小 Embedding 和 Reranker 请求。

## 4. Clarifier 或 READY 解析异常

Clarifier 的 READY 是内部协议，不应当作为普通聊天内容展示。成功结果会保存为每个 run 的 prompt.md，而不是 prompt.txt。

如果多行粘贴、JSON 包裹或 LaTeX 转义导致澄清反复进行，可以绕过 Clarifier：

    kd1-anime generate --file prompt.md --dry-run

也可以直接使用：

    kd1-anime generate "完整的动画需求" --dry-run

长需求建议放在 UTF-8 Markdown 文件中。检查 MAX_PROMPT_CHARS 和
MAX_CLARIFY_CONTEXT_CHARS，过长的多轮对话会按规则保留初始需求和最近回答。

## 5. 计划审查失败或场景数量过多

Plan Review 失败通常表示计划中的数学关系、定义域、时间线、几何实现或场景交接确实不完整。它不会把问题推给 Coder 反复修补。

先只查看计划：

    kd1-anime plan --file prompt.md
    kd1-anime plan --file prompt.md --no-review
    kd1-anime plan --file prompt.md --output plan.json

--no-review 只是预览，不代表计划可以安全编码。正式生成或使用
generate --plan plan.json 时仍会重新执行确定性编译和计划审查。

如果一个需求要求同一画布同时展示一组对象，却被拆成很多 Scene，可以在原始需求中明确“同屏、同一坐标系、对象逐个出现后保持显示”，并检查 MAX_SCENES。Planner 的默认规则是按最小必要视觉单元拆分，而不是按对象清单机械拆分。

## 6. 连续性审查反复修正

代码生成按 Scene ID 顺序进行，下一场景只接收上一场景通过审查后导出的最小 Mobject 定义。若连续性审查发现边界冲突，系统只重规划尚未编码的相关场景，不会随意改写已经渲染的场景。

达到 MAX_CONTINUITY_FIX_ROUNDS 后，当前计划会以 warning 继续；这不是自动失败。可在以下位置查看原因：

    kd1-anime status <run-id>
    kd1-anime logs <run-id> --scene-id 2
    ~/.kd1-anime/workspace/runs/<run-id>/artifacts/
    ~/.kd1-anime/workspace/runs/<run-id>/events.jsonl

如果上游代码已经变化，旧的下游交接代码会被清除并按顺序重建，这是为了避免继续使用陈旧的元素定义。

## 7. Slurm 作业已结束，但被识别为失败

先查看清单和日志：

    kd1-anime status <run-id>
    kd1-anime logs <run-id> --scene-id 3 --lines 200
    kd1-anime logs <run-id> --scene-id 3 --stderr --lines 200

Manim 的最终视频可能位于媒体目录下的嵌套路径，例如：

    videos/scene_3/videos/scene_3/1080p30/Scene3.mp4

系统会递归查找当前精确 Job 的最终 MP4，排除 partial_movie_files，并校验代码哈希、提交时间、分辨率、帧率和 ffprobe 元数据。不要把 partial_movie_files 中的片段手工当作正式产物。

如果 squeue/sacct 暂时不可用，状态会显示 UNKNOWN，而不是立即假定作业失败。达到
UNKNOWN 的次数和持续时间阈值后才会尝试取消；取消失败时不会自动重复提交同一作业。可稍后再次执行：

    kd1-anime resume <run-id>

如果作业确实 COMPLETED 但共享文件系统尚未同步 MP4，可适当增大
MONITOR_ARTIFACT_GRACE。仍找不到经过验证的最终视频时，检查对应 job 的 stdout/stderr，
不要只看终端最后一行。

## 8. OpenGL、Cairo 和 should_render/frame 报错

三个配置含义不同：

- MANIM_RENDERER=cairo 或 opengl：选择渲染器；
- MANIM_OPENGL_PLATFORM=egl 或 glx：只选择 OpenGL 上下文后端；
- SLURM_GPU_TYPE：仅在 OpenGL 作业中申请 GPU。

无显示的 headless 节点通常使用：

    MANIM_RENDERER=opengl
    MANIM_OPENGL_PLATFORM=egl
    SLURM_GPU_TYPE=your-gpu

有显示服务的环境才适合 glx。切换 OPENGL_PLATFORM 不会把 Cairo 变成 OpenGL，也不会修复不兼容的 Manim API。

OpenGL 不支持 Cairo 的 MovingCameraScene 和 self.camera.frame。遇到：

    AttributeError: 'OpenGLCamera' object has no attribute 'frame'

应删除该类 frame 运镜、改用 OpenGL/3D 支持的相机 API，或切换到 Cairo。修改后先运行：

    kd1-anime doctor --probe

遇到：

    AttributeError: VGroup/Polygon object has no attribute 'should_render'

优先检查：

1. 运行时实际使用的 Python 和 Manim 是否来自同一个环境；
2. Manim 是否为项目默认的 0.20.1；
3. Slurm 脚本是否激活了错误的 conda 环境；
4. renderer、Manim 版本和生成代码是否匹配。

    kd1-anime version
    python -c 'import manim; print(manim.__version__)'
    kd1-anime doctor --probe

不要只重复提交同一份代码；如果环境和 renderer 已确认一致，再根据 traceback 的第一处项目代码行修复 Scene。

## 9. XeLaTeX、MathTex 或中文渲染失败

检查：

    kd1-anime doctor --probe

该探针会检查 xelatex、dvisvgm、ctex/CJK、MathTex 和当前 renderer 的真实最小视频产物。

常见原因：

- Python 普通字符串中的反斜杠被当成转义，LaTeX 内容应使用 raw string 或正确转义；
- 花括号不配对，或传给 MathTex 的内容不是合法 LaTeX；
- 远端 conda 环境没有使用安装器配置的 XeLaTeX；
- Manim 版本与文档/代码不一致；
- 中文字体或 ctex/xeCJK 缺失。

检查具体错误时，查看 run 的 stderr 和对应 Tex 目录中的 .log 文件。不要把一个失败生成的
.tex 文件直接复制到下一场景作为代码修复依据；先修正源代码并重新通过校验。

## 10. 视频合并失败

检查：

    kd1-anime status <run-id>
    kd1-anime logs <run-id> --lines 200

合并要求每个必需场景都有经过 ffprobe 验证的最终 MP4。默认
ALLOW_PARTIAL_OUTPUT=false，因此缺少一个场景时会拒绝输出，而不是生成看似完整但内容缺失的
视频。

多场景合并默认使用 xfade=transition=fade，短场景会自动缩短转场时长。若自定义输出已经存在：

    kd1-anime generate "需求" --output final.mp4 --force

或在配置中设置 OVERWRITE_OUTPUT=true。合并过程先写临时文件，验证成功后才原子替换目标文件。

## 11. 恢复运行显示未开始或恢复失败

启动时不会自动扫描历史运行。请显式查询和恢复：

    kd1-anime status
    kd1-anime status <run-id> --json
    kd1-anime resume <run-id>

恢复使用原子 manifest 和运行级锁。它会重新核对代码 SHA-256、Renderer/Merge Profile、精确
Slurm Job 和视频哈希；已完成场景会从清单补发状态，不会因为重启而默认为未开始。

如果 manifest 不是 v6，旧清单可以只读查看，但不能安全写回。建议保留旧目录用于诊断，并重新生成
新的运行，而不是手工修改 manifest。

## 12. 运行很慢或看起来卡住

1. 用 status 确认是在等待 LLM、排队、渲染还是合并；
2. 用 logs 查看远端 stdout/stderr；
3. 降低 MANIM_QUALITY 或 Smoke Render 质量进行排查；
4. 设置合理的 SLURM_MAX_IN_FLIGHT，避免共享队列拥堵；
5. 对重复调试请求使用 cache status，确认是否命中了旧响应；
6. 适当调整阶段级 LLM_*_MAX_TOKENS，不要盲目提高所有阶段的预算；
7. 设置 MONITOR_QUEUE_TIMEOUT、MONITOR_RUN_TIMEOUT 和 MONITOR_UNKNOWN_TIMEOUT，避免无限等待。

如果任务在 Slurm 中仍为 PENDING，通常是队列、分区、账户、QoS 或 GPU 资源问题，不是 Coder
问题。先检查 sbatch 提交输出和集群队列，不要反复触发代码修复。

## 13. 建议提交 issue 的信息

请提供：

- kd1-anime 版本和 Manim 版本；
- 使用的 renderer、质量、分辨率和帧率；
- run ID 和失败 Scene ID；
- status 输出和相关日志尾部；
- 是否启用了 RAG、视觉评估、OpenGL 或容器；
- 脱敏后的错误信息和复现命令。

请不要提供 API Key、Authorization header、完整 .env、未脱敏 URL 凭据或私人视频。
