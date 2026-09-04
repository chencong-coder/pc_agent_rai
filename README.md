# PC Agent — 自然语言控制无人车

基于 [RAI](https://github.com/RobotecAI/rai) 框架，通过**文字或语音**控制搭载 Jetson Orin 的无人车完成 3D 室内目标检测与自主导航。

## 架构

```
PC (LLM) ←→ ROS 2 DDS ←→ Orin (Votenet + Nav2)
   │                          │
   ├─ 理解自然语言/语音输入     ├─ 3D 目标检测 (持续发布)
   ├─ 决策调用工具              ├─ Nav2 导航 (等待目标)
   └─ 执行 ROS 2 通信         └─ 零代码改动
```

**LLM 做决策，Agent 做执行，Orin 做动作。** 默认使用支持工具调用的 DeepSeek `deepseek-chat`。

## 特性

- ✅ 文字 + 🎤 语音双输入（RAI ASR 语音识别）
- ✅ Streamlit 网页前端
- ✅ DeepSeek OpenAI 兼容接口 + 原生工具调用
- ✅ TF 坐标自动变换 (rslidar → map)
- ✅ Docker 一键部署到 Orin

---

## 已有 RAI 项目？直接使用

```bash
cd 你的RAI项目
git clone https://github.com/chencong-coder/pc_agent_rai.git tmp_pc
cp tmp_pc/examples/pc_agent examples/pc_agent -r
cp tmp_pc/config.toml ./
rm -rf tmp_pc
source setup_shell.sh
export DEEPSEEK_API_KEY="你的 DeepSeek API Key"

# 文字输入
python -m examples.pc_agent.main

# 网页输入
streamlit run examples/pc_agent/streamlit_app.py

# 语音输入（需安装 ASR）
uv sync --group s2s
streamlit run examples/pc_agent/streamlit_app.py
```

## 全新安装

```bash
git clone https://github.com/RobotecAI/rai.git
cd rai
uv sync
git clone https://github.com/chencong-coder/pc_agent_rai.git tmp
cp tmp/examples/pc_agent examples/pc_agent -r
cp tmp/config.toml ./
rm -rf tmp
colcon build --symlink-install
source setup_shell.sh
export DEEPSEEK_API_KEY="你的 DeepSeek API Key"
streamlit run examples/pc_agent/streamlit_app.py
```

---

## 配置模型

仓库中的 `config.toml` 已切换到 DeepSeek。运行前只需要设置 API Key：

```bash
export DEEPSEEK_API_KEY="你的 DeepSeek API Key"
```

RAI 的 OpenAI 适配器读取 `OPENAI_API_KEY`，代码会自动把
`DEEPSEEK_API_KEY` 映射过去；也可以直接设置 `OPENAI_API_KEY`。

关键配置如下：

```toml
[vendor]
simple_model = "openai"
complex_model = "openai"

[openai]
simple_model = "deepseek-chat"
complex_model = "deepseek-chat"
base_url = "https://api.deepseek.com"
```

DeepSeek 当前不提供 embeddings 接口，因此 `embeddings_model` 保留为
Ollama 备用配置；PC Agent 本身不会初始化 embeddings。

---

## Docker 部署（Orin）

```bash
cd pc_agent_rai
docker build -t pc-agent -f docker/pc_agent.dockerfile .
docker run -d --name pc-agent --network=host --restart=always \
  -e DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
  pc-agent
```

PC 浏览器访问: `http://<orin-ip>:8501`

---

## 文件结构

```
examples/pc_agent/
├── agent.py              # Agent + System Prompt
├── tools.py              # 检测查询 / 导航发送 / 取消导航
├── streamlit_app.py      # Streamlit 前端（文字+语音）
├── main.py               # 命令行交互
├── test_connection.py    # 通信测试
├── test_raw_sub.py       # 原生订阅测试
├── mock_orin.py          # Orin 模拟器
└── REPORT.md             # 完整技术文档
```

---

## 使用示例

```
你: 周围有什么
Agent: 检测到: bed (2.12, -1.62), chair (1.50, 2.00)...

你: 找床
Agent: 已找到床，正在导航前往 (2.12m, -1.62m)...

你: 停下
Agent: 导航已取消。
```

## License

Apache 2.0
