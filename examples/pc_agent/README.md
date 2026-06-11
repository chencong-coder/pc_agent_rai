# PC Agent - RAI 自然语言控制示例

## 架构

```
┌─────────────────────────────────────────────────────────┐
│  PC (你的电脑)                                           │
│                                                         │
│  你 → 输入文字 → LLM (llama3.2) → ReAct Agent           │
│                      ↓                                  │
│            ┌────────┴────────┐                          │
│            ↓                 ↓                          │
│    get_detections    navigate_to_coordinates            │
│        (订阅)              (Action)                     │
└──────────┬──────────────────┬───────────────────────────┘
           │   ROS 2 DDS      │
           ↓                  ↓
┌──────────────────────────────────────────────────────────┐
│  Orin (小车)                                             │
│                                                          │
│  YOLO 目标检测 ──→ /detection_result                     │
│  Nav2 导航     ←── /navigate_to_pose                     │
│                                                          │
│  不需要写任何额外代码！                                    │
└──────────────────────────────────────────────────────────┘
```

## 前置条件

1. ROS 2 已安装并 source
2. RAI 框架已构建
3. `config.toml` 已配置 LLM（Ollama 或其它）
4. Orin 端：YOLO 发布 `/detection_result`，Nav2 运行中

## 快速开始

### Step 1: 启动 Orin 模拟器（无真机时）

```bash
# 终端1: 启动模拟检测发布
python examples/pc_agent/mock_orin.py
```

### Step 2: 启动 PC Agent

```bash
# 终端2: 启动 Agent
source setup_shell.sh
python -m examples.pc_agent.main
```

### Step 3: 输入指令

```
🧑 你: 周围有什么？
🤖 Agent: (调用 get_detections) → 检测到: 2把椅子, 1张桌子, 2个人...

🧑 你: 找椅子
🤖 Agent: (调用 get_detections, 获取椅子坐标, 调用 navigate_to_coordinates)
         导航指令已发送，目标位置 x=1.50m, y=2.00m

🧑 你: 停下
🤖 Agent: (调用 cancel_navigation) → 导航任务已取消
```

## 支持的命令示例

| 用户输入 | Agent 行为 |
|---------|-----------|
| "周围有什么" | 调用 get_detections 获取检测结果 |
| "找椅子" | 检测→找到椅子坐标→导航到椅子 |
| "去桌子那里" | 检测→找到桌子坐标→导航到桌子 |
| "找人" | 检测→找到人→导航到人 |
| "停下" / "停止" | 取消导航 |

## 配置参数

```bash
python -m examples.pc_agent.main \
  --detection-topic /detection_result \  # Orin 检测话题
  --nav-action navigate_to_pose \        # Nav2 Action 名
  --frame-id map \                       # 导航坐标系
  --model complex_model \                # LLM 模型
  --vendor ollama \                      # LLM 供应商
  --timeout 3.0                          # 检测超时
```

## Orin 端检测格式

`/detection_result` 话题使用 `std_msgs/String`，JSON 格式：

```json
{
  "detections": [
    {"class": "chair", "x": 1.2, "y": 3.4, "z": 0.0, "confidence": 0.92},
    {"class": "person", "x": 2.1, "y": 1.8, "z": 0.0, "confidence": 0.87}
  ],
  "timestamp": 1712345678.9
}
```

## 工作原理

1. **ReAct Agent 循环**: LLM 收到指令 → 思考需要什么工具 → 调用工具 → 观察结果 → 继续思考或回复
2. **工具共享 ROS 2 Connector**: 所有工具共享同一个 `ROS2Connector` 实例（同一个 ROS 2 节点）
3. **无状态执行**: 每条指令独立执行，不依赖历史上下文
