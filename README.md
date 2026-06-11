# PC Agent — 自然语言控制无人车

基于 RAI 框架，通过自然语言指令控制搭载 Jetson Orin 的无人车完成 3D 室内目标检测与自主导航。

## 架构

```
PC (LLM) ←→ ROS 2 DDS ←→ Orin (Votenet + Nav2)
   │                          │
   ├─ 理解自然语言指令          ├─ 3D 目标检测 (持续发布)
   ├─ 决策调用哪个工具          ├─ Nav2 导航 (等待目标)
   └─ 执行 ROS 2 通信         └─ 零代码改动
```

**LLM 只做决策，Agent 做执行，Orin 做动作。** 核心创新：通过文字协议驱动工具调用，不依赖模型原生 Function Calling，兼容任意 LLM。

## 文件结构

```
examples/pc_agent/
├── agent.py              # ReAct Agent + System Prompt
├── tools.py              # 3 个工具: 检测查询/导航发送/取消导航
├── main.py               # 命令行交互入口
├── streamlit_app.py      # Streamlit 网页前端
├── test_connection.py    # 通信测试脚本
├── test_raw_sub.py       # 原始订阅测试
├── mock_orin.py          # Orin 模拟器
├── REPORT.md             # 技术文档
└── README.md             # 使用说明
```

## 快速开始

### 1. 前置条件

- ROS 2 Humble/Jazzy + RAI 框架
- Orin 端: Votenet 检测 + Nav2 导航 + TF
- 大模型: llama3.2 / qwen2.5:7b / gemma2:27b (Ollama)

### 2. 配置模型

编辑 `config.toml`:

```toml
[ollama]
simple_model = "qwen2.5:7b"
complex_model = "qwen2.5:7b"
base_url = "http://<ollama服务器>:11434"
```

### 3. 启动

```bash
source /opt/ros/humble/setup.bash
source setup_shell.sh

# 命令行:
python -m examples.pc_agent.main

# 或网页:
streamlit run examples/pc_agent/streamlit_app.py
```

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
