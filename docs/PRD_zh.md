# advX 发音评测服务产品需求文档

> 文档状态：团队交接版  
> 项目版本：0.1.0  
> 更新时间：2026-07-24  
> 主要读者：新加入的产品、后端、音频算法及测试工程师

## 1. 项目概述

advX Speech Service 是一个面向单词发音练习场景的 Python 服务。系统从麦克风采集语音，将实时音频帧写入 Redis Streams，组合出完整单词录音，完成音频清洗和发音评测，最后将结构化评测结果保存到 SQLite 并推送给前端。

当前仓库主要完成了：

- 麦克风音频采集与标准化分帧；
- Redis 音频流、会话状态和 UI 事件适配；
- Redis 帧校验与完整单词窗口组装；
- WAV/PCM 音频清洗；
- SQLite 基准词与评测记录持久化；
- FastAPI 健康检查和 WebSocket 事件推送；
- 清洗流程与数据库模块的黑盒测试。

真实的语音识别或音素级评分引擎目前只有接口契约，尚未在仓库中实现。

## 2. 产品目标

### 2.1 核心目标

为一个给定单词提供稳定、可追踪、可复现的发音评测流程：

1. 采集用户录音；
2. 可靠地传输完整音频；
3. 将音频转换为统一格式并清理噪声；
4. 与目标音素及可接受变体进行比较；
5. 返回置信度、得分和检测到的音素；
6. 保存最终结果，供前端展示和后续分析。

### 2.2 成功标准

- 发音引擎始终收到完整单词窗口，而不是单个 40 ms 帧；
- 对外音频格式始终为单声道、16 kHz、signed int16 little-endian PCM；
- 音频清洗结果具有确定性；
- SQLite 仅保存结构化业务数据，不长期保存原始音频；
- Redis 中的音频和会话状态具有明确的长度或 TTL 限制；
- 失败的处理不能被错误确认，且能够恢复；
- API 返回的每次评测结果都可以关联到数据库中的 attempt；
- 黑盒验收测试覆盖正常输入、边界输入和异常输入。

## 3. 用户与使用场景

### 3.1 主要用户

- 发音练习者：朗读指定单词并接收评分；
- 教师或内容运营人员：维护目标单词、标准音素和可接受变体；
- 前端开发者：订阅会话事件并展示实时状态和最终结果；
- 音频/算法工程师：实现或替换 cleaning 和 pronunciation engine；
- 后端工程师：维护 Redis、SQLite、API 和任务消费流程。

### 3.2 核心用户流程

1. 客户端或采集程序创建 `session_id`；
2. 用户朗读目标单词；
3. producer 以 40 ms 为单位发布音频帧；
4. worker 从 Redis Stream 消费并校验帧；
5. VAD 或上层逻辑确定完整单词窗口；
6. cleaning 将窗口转换为规范 PCM；
7. pronunciation engine 进行音素识别和评分；
8. 服务将 attempt 写入 SQLite；
9. 服务更新 Redis 会话状态并发布 UI 事件；
10. 前端通过 WebSocket 接收结果。

## 4. 系统流程与边界

```text
麦克风
  │ 16 kHz / mono / int16
  ▼
MicrophoneProducer
  │ 每帧 40 ms、640 samples、1280 bytes
  ▼
Redis Stream: audio:{session_id}
  ▼
Worker 帧校验与连续窗口组装
  ▼
Cleaning: bytes -> bytes
  │ mono / 16 kHz / int16 little-endian
  ▼
PronunciationEngine
  ├── SQLite: benchmarks / attempts
  └── Redis: session state / UI events
          ▼
       WebSocket 前端
```

### 4.1 公共音频契约

每条 Redis 音频消息必须包含：

| 字段 | 类型 | 说明 |
|---|---|---|
| `sequence` | bytes 表示的整数 | 帧序号 |
| `captured_at_ns` | bytes 表示的整数 | 单调时钟采集时间 |
| `sample_rate` | bytes 表示的整数 | 当前必须为 `16000` |
| `frame_ms` | bytes 表示的整数 | 当前必须为 `40` |
| `pcm_s16le` | bytes | 长度必须为 `1280` |

完整窗口的 cleaning 公共入口为：

```python
app.cleaning.clean_pcm(pcm_s16le: bytes) -> bytes
```

输出必须满足：

- 单声道；
- 16 kHz；
- signed int16 little-endian；
- 样本完整，字节数为偶数；
- 不包含 NaN 或 Inf；
- 峰值不高于约 -1 dBFS；
- 建议时长为 0.35–15 秒。

## 5. 功能需求

### FR-01 音频采集

- 默认生成 UUID 作为 `session_id`；
- 使用配置中的 sample rate、声道、dtype 和 frame size；
- 音频 callback 不直接访问 Redis；
- callback 将音频复制到有界内存队列；
- 队列溢出必须记录丢帧；
- 停止采集时，应在限定时间内处理剩余队列或明确报告未发送数据。

### FR-02 Redis 音频发布

- 音频帧尺寸不正确时拒绝写入；
- Stream key 为 `audio:{session_id}`；
- Stream 设置最大长度，避免无限增长；
- 序号和采集时间必须能够暴露音频缺口；
- Redis 错误不得静默丢失。

### FR-03 Worker 消费

- 创建 consumer group 时应允许重复初始化；
- 仅在 frame processor 成功后执行 `XACK`；
- 必须处理 worker 崩溃后遗留的 pending messages；
- 必须拒绝字段缺失、格式错误、采样率错误或尺寸错误的帧；
- 完整窗口必须按采集顺序排列；
- 序号或时间戳出现不可接受的跳变时必须拒绝窗口。

### FR-04 音频清洗

文件级 cleaning 需要：

- 校验文件大小、声道数、采样率和时长；
- 转为 mono 16 kHz；
- 去除 DC offset 和 70 Hz 以下 rumble；
- 检测有效语音区域；
- 在需要时执行保守的 spectral denoise；
- 保留首尾 padding，不删除单词内部真实停顿；
- 将语音响度调整到目标 RMS，并限制峰值；
- 输出 PCM-16 WAV 和 JSON report。

清洗报告必须至少包含：

- pipeline version；
- 接受状态；
- 输入/输出格式和时长；
- speech duration 与 speech ratio；
- peak、RMS、noise floor 和 SNR；
- clipping ratio 与 DC offset；
- 是否执行 denoise；
- warnings。

状态语义：

- `accept`：可以进入发音引擎；
- `accept_with_warning`：可以进入引擎，但应保存或展示警告；
- `retry_recommended`：默认不应继续评分，应要求用户重新录音。

### FR-05 发音评测

发音引擎必须实现以下接口：

```python
evaluate(
    pcm_s16le: bytes,
    sample_rate: int,
    expected_phonemes: list[str],
    accepted_variants: list[list[str]],
) -> PronunciationResult
```

结果包含：

- 检测到的音素；
- 每个音素的置信度；
- 总体置信度；
- 可选总分；
- engine version。

当前状态：仅定义 Protocol 和结果数据结构，真实引擎待实现。

### FR-06 数据持久化

SQLite 保存：

- `benchmarks`：语言、单词、标准音素、可接受变体、难度和版本；
- `attempts`：session、benchmark、检测音素、置信度、得分、引擎版本和创建时间。

要求：

- 启用 foreign keys；
- attempt 插入使用事务；
- 插入失败必须 rollback；
- benchmark lookup 应大小写无关；
- schema 初始化应幂等；
- 原始音频不写入 SQLite。

### FR-07 前端事件

- 会话状态保存在 `session:{session_id}:state` Hash；
- 状态必须设置 TTL；
- UI 事件写入 `session:{session_id}:events` Stream；
- WebSocket 路径为 `/sessions/{session_id}/events`；
- 最终事件类型为 `pronunciation.evaluated`；
- 事件需要包含 attempt、benchmark 和 engine 版本。

## 6. 非功能需求

### 6.1 性能

- cleaning 平均处理时间应小于输入音频的真实时长；
- Redis callback 不执行网络 I/O；
- 单次读取最多 25 个音频帧；
- WebSocket 的阻塞 Redis 读取必须在线程中执行，避免阻塞事件循环。

### 6.2 可靠性

- Redis consumer 采用至少一次处理语义；
- worker 重启后必须恢复 pending messages；
- 不允许将有缺口的音频误判为连续窗口；
- 输出文件建议使用临时文件加原子 rename，避免半完成结果；
- Redis 不可用时需要可观察的错误和重试策略。

### 6.3 安全与隐私

音频可能包含个人数据，尤其在涉及未成年人时：

- 生产环境必须启用认证和 TLS；
- session 和 WebSocket 必须实施访问控制；
- 不得公开暴露 Redis 或 SQLite；
- 明确定义音频保留期限和用户同意流程；
- 日志记录元数据，不记录原始音频；
- `session_id` 不应直接信任客户端任意输入。

## 7. 当前实现状态

| 模块 | 状态 | 备注 |
|---|---|---|
| 配置管理 | 已实现 | 环境变量 + Pydantic |
| 麦克风 producer | 已实现，需加固 | 存在丢帧与 publisher 退出风险 |
| Redis 音频 Stream | 已实现 | 帧结构和最大长度已定义 |
| Worker 消费 | 部分实现 | 缺少 pending recovery 和完整 VAD 调度 |
| 完整窗口组装 | 已实现 | 当前只检查 sequence 连续 |
| 文件级 cleaning | 已实现 | 包含 VAD、denoise、trim、normalization |
| PCM cleaning adapter | 已实现，需修正 | 当前丢弃 cleaning status/report |
| 发音引擎 | 未实现 | 只有 Protocol |
| SQLite repository | 已实现 | benchmark 和 attempt |
| HTTP/WebSocket API | 部分实现 | health + 单向事件订阅 |
| 自动化测试 | 已实现 | 当前共 46 个测试 |
| 认证、授权、TLS | 未实现 | 上线前必须完成 |

## 8. 已知问题与技术风险

### P0：VAD 对连续语音存在误拒绝

当前 noise floor 使用所有 frame 的第 20 百分位。当语音占比约为 80% 或更高时，该值可能已经是语音能量，最终阈值高于真实语音。测试中，语音占比 90% 和 100% 的有效合成录音会被拒绝。

建议：

- 优先使用明确的首尾噪声区间；
- 或使用更稳健的双阈值、自适应 VAD；
- 增加“少静音/无静音”黑盒测试。

### P0：Redis pending message 无恢复流程

worker 只读取 `>` 新消息。崩溃前未确认的消息会长期停留在 PEL 中。

建议：

- worker 启动和周期运行时使用 `XAUTOCLAIM`；
- 设置 claim idle time；
- 设计幂等 frame processor；
- 增加 worker crash/restart 集成测试。

### P0：丢帧无法通过 sequence 检出

sequence 在发布成功后才递增，callback 队列中丢失的帧不会产生序号缺口；`captured_at_ns` 当前也未用于连续性校验。

建议：

- 在采集 callback 为每个输入 frame 分配 capture sequence；
- 将 sequence 与时间戳一起入队；
- worker 同时校验 sequence 和时间间隔。

### P1：`retry_recommended` 未阻止后续评分

PCM adapter 调用文件 cleaning 后忽略 report，只返回清洗后的 bytes。

建议：

- 将公共返回值升级为结构化 `CleaningResult`；
- 或在 `retry_recommended` 时抛出专用异常；
- 明确前端如何展示 retry 原因。

### P1：SNR 与增益计算包含内部静音

当前计算使用首个和最后一个 voice frame 之间的完整区间，而不是仅使用 voice mask 对应样本，可能低估 voice RMS 并导致过度增益或误触发 denoise。

建议基于 voice mask 提取语音样本后再计算 RMS/SNR。

### P1：Producer 网络异常可能终止 publisher thread

Redis publish 异常会退出后台线程。主采集线程不会立即感知，后续队列可能持续增长并丢帧。

建议：

- 在线程间传播异常；
- 添加有限重试和 backoff；
- 达到失败阈值后主动停止录音并通知用户。

## 9. 验收标准

### AC-01 标准录音

给定有效的 16 kHz mono PCM 单词窗口：

- cleaning 返回非空、偶数字节长度的 int16 PCM；
- 输出时长为 0.35–15 秒；
- 输出峰值不超过 -1 dBFS；
- 相同输入多次执行得到相同输出。

### AC-02 异常帧

当 Redis frame 缺字段、尺寸错误、采样率错误、顺序错误或存在时间缺口时：

- 不调用 pronunciation engine；
- 不执行 `XACK`；
- 产生可追踪错误。

### AC-03 Worker 恢复

worker 在读取后、确认前崩溃并重启：

- pending frame 能被新 worker claim；
- frame 最终被处理并确认；
- 不产生重复 attempt。

### AC-04 Cleaning 重试

当 report 为 `retry_recommended`：

- 不调用 pronunciation engine；
- 不创建最终 attempt；
- 发布包含 retry 原因的 UI 事件。

### AC-05 持久化

成功评测后：

- SQLite 存在且只存在一个对应 attempt；
- attempt 引用有效 benchmark；
- Redis UI event 中的 `attempt_id` 与 SQLite 一致；
- Unicode 音素、单词和 engine version 可正确往返。

## 10. 测试策略

当前使用 `pytest`，主要分为：

- 单元测试：frame geometry、PCM split、基础 API；
- black-box cleaning：真实 Redis payload 输入、规范 PCM 输出；
- VoiceBank-DEMAND fixture：4 组 noisy/clean 参考音频与 transcript；
- 性能测试：cleaning 快于实时；
- SQLite black-box：初始化、事务、Unicode、异常输入；
- Redis adapter black-box：命令格式、TTL、JSON 和确定性轻量 fuzz。

下一阶段应增加：

- Redis 容器集成测试；
- pending/claim 和 worker crash 测试；
- producer 网络失败测试；
- 无静音、长静音、爆音、削波及极低 SNR 测试；
- Hypothesis property-based 测试；
- WebSocket 断线重连测试；
- 真实 pronunciation engine 的契约测试。

运行全部测试：

```bash
uv run pytest
```

## 11. 新队友建议上手顺序

1. 阅读本 PRD 和根目录 `README.md`；
2. 执行 `uv sync`；
3. 运行 `uv run pytest`，建立当前行为基线；
4. 阅读 `app/config.py` 和 `app/redis_bus.py`；
5. 跟踪 `audio_producer.py -> worker.py -> cleaning.py`；
6. 阅读 `audio_cleaning.py` 理解 DSP pipeline；
7. 阅读 `pronunciation_engine.py` 的待实现接口；
8. 阅读 `benchmark_repository.py` 和 `scripts/init_db.py`；
9. 优先修复 P0 风险，再接入真实发音引擎。

## 12. 本地启动

准备环境：

```bash
uv sync
cp .env.example .env
uv run python -m scripts.init_db
```

启动 Redis：

```bash
docker run --name advx-redis \
  -p 127.0.0.1:6379:6379 \
  -d redis:7-alpine \
  redis-server --appendonly yes
```

启动 API：

```bash
uv run uvicorn app.api:app \
  --host 127.0.0.1 \
  --port 8000 \
  --env-file .env \
  --reload
```

启动麦克风 producer：

```bash
uv run python -m app.audio_producer
```

## 13. 待团队确认的问题

1. 单词窗口由客户端、VAD 还是独立 orchestrator 决定？
2. `retry_recommended` 是否必须严格阻止评分？
3. 发音引擎采用本地模型、云 API 还是混合方案？
4. benchmark 内容由谁维护，版本发布流程是什么？
5. attempt 是否允许同一 session 和 benchmark 多次创建？
6. Redis 音频的生产保留时间和最大容量是多少？
7. WebSocket 重连后是否需要补发历史事件？
8. 目标语言和音素标准使用 IPA 还是引擎自有符号集？
9. 评分的产品解释、阈值和用户提示文案是什么？
10. 生产环境的认证、授权、同意和数据删除策略是什么？
