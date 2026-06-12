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

**LLM 做决策，Agent 做执行，Orin 做动作。** 支持原生 Function Calling（qwen2.5:32b）。

## 特性

- ✅ 文字 + 🎤 语音双输入（RAI ASR 语音识别）
- ✅ Streamlit 网页前端
- ✅ 原生 Function Calling（qwen2.5:32b 支持）
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

# 文字输入
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
streamlit run examples/pc_agent/streamlit_app.py
```

---

## 配置模型

编辑 `config.toml`：

```toml
[ollama]
simple_model = "qwen2.5:32b"
complex_model = "qwen2.5:32b"
base_url = "http://<Ollama服务器IP>:11434"
```

推荐 `qwen2.5:32b`（支持原生 Function Calling）或 `gemma2:27b`（手动 ReAct 兼容）。

---

## Docker 部署（Orin）

```bash
cd pc_agent_rai
docker build -t pc-agent -f docker/pc_agent.dockerfile .
docker run -d --name pc-agent --network=host --restart=always pc-agent
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
