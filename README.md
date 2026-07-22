# 论文对齐版医疗对话 Self-play 原型

本目录实现了 AMIE 论文 inner self-play 的交互原型：Vignette Generator、Patient、Doctor、Moderator 与 Critic 通过 WebSocket 编排，最多进行三轮同病例问诊和两次 Critic 改进反馈。每轮结束后按 `Doctor DDx → Critic → Evaluation` 执行，并生成论文量表对齐的模型代理评分。

## 启动

先创建本地模型 API 配置：

```bash
cp config/model_apis.example.json config/model_apis.json
uv sync
uv run uvicorn amie_self_play.app:app --app-dir src --reload
```

打开 <http://127.0.0.1:8000>。默认配置：

- 模型列表和默认模型来自 `config/model_apis.json`
- 服务启动时读取配置，网页通过 `/api/models` 加载可选模型
- 配置文件不应提交到 Git；仓库提供的 `config/model_apis.example.json` 只包含示例地址
- 超时、重试次数也可以在配置文件中设置，网络错误或 5xx 会按配置重试

### 模型 API 配置

`config/model_apis.json` 顶层是一个 object，至少需要 `models` 数组：

```json
{
  "default_model": "primary-chat",
  "timeout_seconds": 120,
  "max_retries": 1,
  "models": [
    {
      "name": "primary-chat",
      "real_name": "我的模型",
      "endpoint": "https://llm.example.com/v1/chat/completions",
      "model": "my-model-id",
      "api_key_env": "MY_LLM_API_KEY",
      "context_length": 128000,
      "support_stream": false,
      "image_input": false
    }
  ]
}
```

字段说明：

- `name`：网页和 WebSocket 请求使用的唯一标识。
- `real_name`：网页中展示的上游模型名称；省略时使用 `name`。
- `endpoint`：该模型的 HTTP API 地址。默认请求体兼容 OpenAI Chat Completions。
- `model`：发送给上游 API 的真实模型名；省略时使用 `name`。
- `api_key_env`：存放 API 密钥的环境变量名。服务启动/调用时从环境变量读取，不要把密钥写入 JSON。
- `model_field`：上游请求体中的模型字段，默认是 `model`；私有网关使用 `model_name` 时可以覆盖。
- `headers`：额外 HTTP headers，支持 `${ENV_VAR}` 环境变量替换。
- `body`：合并到请求体的额外字段，例如 `{"enable_thinking": false}`。
- `context_length`、`support_stream`、`image_input`：仅用于网页展示。

#### API Key 配置

配置文件不直接接受 `"api_key": "..."`。这是为了避免密钥随 `model_apis.json` 被误提交到 Git。请在模型配置中通过 `api_key_env` 填写“保存密钥的环境变量名称”，再在启动服务前设置该环境变量。

OpenAI、DeepSeek 官方 API、火山引擎方舟等使用 Bearer Token 的接口可以这样配置：

```json
{
  "name": "primary-chat",
  "endpoint": "https://llm.example.com/v1/chat/completions",
  "model": "my-model-id",
  "api_key_env": "MY_LLM_API_KEY",
  "api_key_header": "Authorization",
  "api_key_prefix": "Bearer "
}
```

启动服务前设置真实密钥：

```bash
export MY_LLM_API_KEY='your-api-key'
uv run uvicorn amie_self_play.app:app --app-dir src
```

服务会自动生成请求头 `Authorization: Bearer your-api-key`。真实密钥只存在于进程环境中，不会返回给网页。

Azure OpenAI 等使用 `api-key` 请求头且不需要前缀的接口可以这样配置：

```json
{
  "api_key_env": "AZURE_OPENAI_API_KEY",
  "api_key_header": "api-key",
  "api_key_prefix": ""
}
```

如果接口不需要认证，可以省略 `api_key_env`、`api_key_header` 和 `api_key_prefix`。若配置了 `api_key_env` 但启动进程中没有对应环境变量，选择该模型时会返回明确的配置错误。

默认配置路径为 `config/model_apis.json`。也可以通过环境变量指定其他位置：

```bash
export AMIE_MODEL_CONFIG=/absolute/path/to/model_apis.json
```

如果上游 API 使用自定义模型字段，可以这样配置：

```json
{
  "name": "private-gateway",
  "endpoint": "https://gateway.example.com/chat",
  "model": "gateway-model",
  "model_field": "model_name",
  "api_key_env": "GATEWAY_API_KEY",
  "body": {"enable_thinking": false}
}
```

服务不会把 `endpoint`、`headers`、`body` 或密钥返回给浏览器；网页只接收模型展示元数据。

## 测试

```bash
uv run pytest
```

## 机制边界

- 不执行论文中的 Web Search Retrieval 和 Passage Filtering。
- 每次只生成一个 vignette，而非论文原 prompt 的两个。
- Vignette 包含 ground truth、3–10 项 accepted differential 和参考管理计划；Case File 对研究用户可见，但 Doctor、Moderator 和 DDx 均不可见。
- 基线 Doctor prompt 不提前加入 Critic 的“至少两个鉴别诊断”等强化标准。
- 每轮结束后自动显示 Critic 评价；Round 1/2 的 Critic 可分别用于生成 Round 2/3，Evaluation 评分不会进入后续 Doctor 或 Patient 上下文。
- Doctor DDx 是独立的会诊后输出，只读取本轮 transcript，并按可能性给出 3–10 个诊断。
- Evaluation 使用当前所选基座模型并行运行 Accuracy、Patient Actor、Specialist 和 Auto PACES 四组独立评审。单组无效会修复一次；失败只产生 partial 结果，不阻塞回合完成。
- Patient Actor、Specialist 和 Auto PACES 分别固定为论文的 26、32 和 4 个评分轴。页面明确将其标为 model-based proxy，不能替代真实患者或专科医生评价。
- Accuracy 仅展示当前单病例相对 ground truth 和 accepted differential 的 Top-1、Top-3、Top-10 二元命中，不计算跨病例百分比或跨量表总分。
- 模型下拉列表由服务启动时读取的 `config/model_apis.json` 提供；所选模型用于该次模拟的所有 agent 调用。
- 每轮最多 30 条医患消息；Critic 和 Moderator 不计入。
- 页面及输出仅用于研究模拟，不能用于真实诊断或治疗决策。

原有的最小命令行调用示例仍可使用：

```bash
uv run python call_model.py --model primary-chat "请只回复：连接成功"
```

真实模型验收（默认运行腕管综合征的 Baseline 和一次 Critic 改进轮）：

```bash
uv run python scripts/run_acceptance.py
```
