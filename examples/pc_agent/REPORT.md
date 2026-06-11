# RAI PC Agent — 自然语言控制机器人系统

## 汇报总结

---

## 一、项目概述

### 目标

在 **PC + Orin 分布式架构** 下，实现**通过中文自然语言控制室内机器人**完成目标检测与自主导航。

### 核心成果

| 指标 | 状态 |
|------|------|
| PC ↔ Orin ROS 2 通信 | ✓ 已打通 |
| VoteNet 3D 检测数据接收 | ✓ 持续订阅可用 |
| TF 坐标变换 (rslidar→map) | ✓ 自动转换 |
| LLM 指令解析 (Ollama llama3.2) | ✓ ReAct Agent |
| Nav2 导航 (Action 调用) | ✓ 预留接口 |
| 交互式命令行界面 | ✓ 可用 |

---

## 二、系统架构

```
┌─────────────────────────────────────────────────────────┐
│  PC (决策层) - 你的笔记本电脑                              │
│                                                         │
│  自然语言输入 → ReAct Agent → Ollama (llama3.2)           │
│         │                                               │
│         ├─ get_detections (读 Orin 检测)                  │
│         ├─ navigate_to_coordinates (发导航目标)            │
│         └─ cancel_navigation (取消导航)                    │
│                                                         │
│  通信方式: ROS 2 DDS (Topic + Action)                    │
└──────────────┬──────────────────────────────────────────┘
               │  ROS 2 DDS 网络
               ↓
┌─────────────────────────────────────────────────────────┐
│  Orin (执行层) - 小车的 Jetson Orin                       │
│                                                         │
│  VoteNet 3D检测 ──→ /detect_bbox3d (持续发布, ~0.4Hz)    │
│  TF 变换         ──→ /tf (rslidar→turtle_bot→odom→map)  │
│  Nav2 导航       ←── /navigate_to_pose (Action Server)   │
│                                                         │
│  Orin 端零代码改动，原样运行                                │
└─────────────────────────────────────────────────────────┘
```

### 三层职责

| 层 | 做什么 | 不做什么 |
|----|--------|---------|
| **LLM (决策)** | 理解自然语言，选择工具，决定参数 | 不知道底层是 ROS 2 |
| **Agent (执行)** | 翻译工具调用为 ROS 2 操作 | 不做决策 |
| **Orin (动作)** | 持续检测 + 等待导航指令 | 不写任何额外代码 |

---

## 三、一次指令的完整数据流

以用户说"**找床**"为例：

```
①  用户输入 "找床"
②  Agent 将 [系统提示词 + "找床"] 发给 Ollama
③  LLM 决策 → 输出: { tool: "get_detections", args: {} }
④  Agent 从检测缓存中读取最新一帧 VoteNet 数据
    (缓存由独立 subscriber 持续更新，无需每次请求 Orin)
⑤  解析 Detection3DArray → 提取: bed (2.12, -1.62, 0), conf=0.696
⑥  TF 变换: rslidar 坐标 → map 坐标 (通过 /tf 话题自动查询)
⑦  结果返回 LLM: "检测到 1个目标: bed 坐标(4.5, 1.2, 0)"
⑧  LLM 决策 → 输出: { tool: "navigate_to_coordinates", args: {x:4.5, y:1.2} }
⑨  Agent 封装 PoseStamped → 发 Action Goal 到 Orin Nav2
⑩  Orin Nav2 收到 → 规划路径 → 控制底盘
⑪  LLM 最终回复: "已找到床，正在导航前往 (4.5m, 1.2m)"
```

### LLM 推理轮次

一条指令通常触发 **2-3 次** LLM 推理，每次输入输出不同：

| 轮次 | LLM 输入 | LLM 输出 |
|------|----------|----------|
| 1 | 用户指令 + 系统提示 | tool_call: get_detections |
| 2 | 检测结果列表 | tool_call: navigate_to_coordinates |
| 3 | 导航成功确认 | 最终中文回复 |

---

## 四、代码结构

```
examples/pc_agent/
├── __init__.py           # 包标记
├── tools.py              # 3个LLM工具 (500行)
│   ├── GetDetectionsTool       # 读检测 → 独立rclpy节点持续订阅
│   ├── NavigateToCoordinatesTool # 发导航 → ROS 2 Action Client
│   └── CancelNavigationTool    # 取消导航
├── agent.py              # Agent工厂 + 系统提示词
│   ├── SYSTEM_PROMPT           # 中文行为规范 (802字)
│   └── create_pc_agent()       # 组装: LLM + Tools → ReAct Agent
├── main.py               # 交互式命令行入口
├── test_connection.py    # 通信测试脚本
└── REPORT.md             # 本文件
```

### 关键技术点

| 技术点 | 方案 | 原因 |
|--------|------|------|
| 检测订阅 | 独立 rclpy 节点 + SingleThreadedExecutor | 绕过 RAI connector QoS 兼容问题 |
| TF 变换 | connector.get_transform("map", "rslidar") | 纯数学旋转+平移，无外部依赖 |
| 消息解析 | 原生 vision_msgs.msg | 匹配 VoteNet 标准输出 |
| LLM 框架 | LangGraph ReAct + Ollama | RAI 内置，开箱即用 |
| 导航 | Nav2 Action Client | 标准 ROS 2 接口 |

---

## 五、遇到的技术问题与解决方案

### 问题 1: Docker 拉取失败

**现象**: `docker pull` 时报 "access denied"

**原因**: 没有 `docker login`，或镜像仓库需要认证

**解决**: 先 `docker login`，或使用公开镜像

---

### 问题 2: VoteNet Docker 文件权限

**现象**: VoteNet 结果保存失败，Permission denied

**原因**: Docker 容器内的用户 UID 与宿主机目录权限不匹配

**解决**: `chmod 777` 或在 docker run 时加 `--user $(id -u):$(id -g)`

---

### 问题 3: `nav2_msgs` 导入失败

**现象**: `from rai.tools.ros2.base import BaseROS2Tool` 报错 `ModuleNotFoundError: nav2_msgs`

**原因**: `rai.tools.ros2.__init__.py` 无条件导入了 `nav2` 模块，而 PC 上未安装 `nav2_msgs`

**解决**: 工具类直接继承 `langchain_core.tools.BaseTool`，不经过 RAI 的 `BaseROS2Tool`

---

### 问题 4: RAI `receive_message` 收不到消息

**现象**: `connector.receive_message()` 返回 timeout，但 `ros2 topic echo` 正常

**原因**: 
- `create_subscriber` 不传 `msg_type` 时依赖 DDS 发现查类型，发现可能不及时
- RAI 的 QoS 自动匹配 (`adapt_requests_to_offers`) 可能产生不兼容的 QoS 配置

**解决**: 
- 传 `msg_type` 显式指定，跳过 DDS 发现
- 最终方案：不使用 RAI 的 subscriber，改用独立 rclpy 节点 + SingleThreadedExecutor

---

### 问题 5: `ros2 topic echo --once` 无输出

**现象**: `ros2 topic echo /detect_bbox3d --once` 无输出，但持续 `ros2 topic echo` 有输出

**原因**: VoteNet 发布频率低 (~0.4Hz)，且 QoS 可能为 TRANSIENT_LOCAL，`--once` 的短暂订阅窗口可能错过消息

**解决**: 使用持续订阅模式（独立 rclpy 节点后台 spin），Agent 调用时只读缓存

---

### 问题 6: VoteNet 检测结果偶尔为空

**现象**: 通信正常但 `detections: []`

**原因**: VoteNet 按帧检测，空场景或物体不在视野内时空结果也是合法的

**解决**: Agent 的 System Prompt 中明确"如果没检测到，诚实告诉用户"

---

## 六、当前状态与待完成

### 已完成

| 功能 | 状态 |
|------|------|
| PC ↔ Orin ROS 2 网络发现 | ✓ |
| VoteNet 检测数据持续订阅 | ✓ |
| Detection3DArray 解析 (类名+坐标+置信度) | ✓ |
| TF 坐标变换 rslidar→map | ✓ |
| LLM ReAct Agent 指令解析 | ✓ |
| Nav2 导航 Action 调用接口 | ✓ (预留，待 Nav2 启动后测试) |
| 交互式命令行 | ✓ |

### 待完成

| 事项 | 说明 |
|------|------|
| Nav2 联调 | Orin 启动 Nav2 后测试完整导航链路 |
| TF 变换端到端验证 | 对比 rslidar 坐标和 map 坐标是否正确 |
| Llama3.2 性能评估 | 确认推理速度和指令理解准确率 |
| 异常处理完善 | 网络断开、VoteNet 崩溃等场景 |

---

## 七、运行指南

### Orin 端（无需改动）

```bash
# 确保以下正在运行:
#   VoteNet  → /detect_bbox3d
#   TF       → /tf (rslidar→map)
#   Nav2     → /navigate_to_pose (导航时需启动)
```

### PC 端

```bash
cd ~/Desktop/rai-main
source /opt/ros/humble/setup.bash
source setup_shell.sh
python -m examples.pc_agent.main
```

### 交互示例

```
🧑 你: 周围有什么
🤖 Agent: 检测到: bed (4.50, 1.20, 0.00) 置信度 0.70

🧑 你: 去床那里
🤖 Agent: 已找到 bed，正在导航前往 (4.50m, 1.20m)...

🧑 你: 停下
🤖 Agent: 导航已取消，小车停止。
```

---

## 八、一句话总结

> **LLM 做决策，Agent 做执行，Orin 做动作。**
> **Orin 零代码改动，通信由 ROS 2 DDS 全自动完成。**
> **自然语言 → 3D 检测 → TF 变换 → Nav2 导航，端到端闭环。**
