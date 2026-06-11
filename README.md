# PC Agent — 自然语言控制无人车

基于 [RAI](https://github.com/RobotecAI/rai) 框架，通过自然语言指令控制搭载 Jetson Orin 的无人车完成 3D 室内目标检测与自主导航。

## 架构

```
PC (LLM) ←→ ROS 2 DDS ←→ Orin (Votenet + Nav2)
   │                          │
   ├─ 理解自然语言指令          ├─ 3D 目标检测 (持续发布)
   ├─ 决策调用哪个工具          ├─ Nav2 导航 (等待目标)
   └─ 执行 ROS 2 通信         └─ 零代码改动
```

**LLM 只做决策，Agent 做执行，Orin 做动作。** 手动 ReAct 协议驱动工具调用，不依赖模型原生 Function Calling，兼容任意 LLM（gemma2 / llama / qwen）。

---

## 已有 RAI 项目？直接使用

```bash
cd 你的RAI项目
git clone https://github.com/chencong-coder/pc_agent_rai.git tmp_pc
cp tmp_pc/examples/pc_agent examples/pc_agent -r
cp tmp_pc/config.toml ./
rm -rf tmp_pc
source setup_shell.sh
python -m examples.pc_agent.main
```

## 全新安装

```bash
# 1. 克隆 RAI
git clone https://github.com/RobotecAI/rai.git
cd rai

# 2. 安装依赖
uv sync

# 3. 安装 PC Agent
git clone https://github.com/chencong-coder/pc_agent_rai.git pc_agent_repo
cp pc_agent_repo/examples/pc_agent examples/pc_agent -r
cp pc_agent_repo/config.toml ./
rm -rf pc_agent_repo

# 4. 构建并启动
colcon build --symlink-install
source setup_shell.sh
python -m examples.pc_agent.main
```

---

## 配置模型

编辑 `config.toml`，改成你的 Ollama 模型和地址：

```toml
[ollama]
simple_model = "qwen2.5:7b"
complex_model = "qwen2.5:7b"
base_url = "http://<你的Ollama服务器IP>:11434"
```

---

## 启动方式

```bash
# 命令行
python -m examples.pc_agent.main

# 网页（Streamlit）
streamlit run examples/pc_agent/streamlit_app.py
```

---

## 文件说明

```
examples/pc_agent/
├── agent.py              # ReAct Agent + System Prompt
├── tools.py              # 3 个工具: 检测查询/导航发送/取消导航
├── main.py               # 命令行交互入口
├── streamlit_app.py      # Streamlit 网页前端
├── test_connection.py    # 通信测试
├── test_raw_sub.py       # 原生订阅测试
├── mock_orin.py          # Orin 模拟器（无真机时开发用）
├── REPORT.md             # 完整技术文档
└── README.md
```

---

## 使用示例

```
你: 周围有什么
Agent: 检测到: bed (2.12, -1.62), chair (1.50, 2.00)...

你: 找床
Agent: 已找到床，正在导航前往 (2.12m, -1.62m)...

你: 停下
Agent: 导航已取消，小车停止。
```

## License

Apache 2.0
