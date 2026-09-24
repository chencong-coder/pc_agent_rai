# PC Agent - RAI 自然语言控制示例

PC Agent 使用 DeepSeek 理解中文指令，通过 ROS 2 工具读取 VoteNet 检测结果并调用 Orin 上的 Nav2，让小车执行检测、导航和停止操作。

## 架构

```
PC / RAI Agent
  用户指令 -> DeepSeek (OpenAI 兼容接口)
                  |
                  +-- get_detections -------- TCP socket -------- Orin VoteNet bridge
                  +-- navigate_to_coordinates ROS 2 Action ----- Orin Nav2
                  +-- cancel_navigation ------ ROS 2 Action ----- Orin Nav2
```

## 前置条件

1. 已安装并 source ROS 2 和 RAI。
2. `config.toml` 已放在 RAI 工作空间根目录（本仓库版本默认使用 DeepSeek）。
3. 已设置 DeepSeek API Key：

   ```bash
   export DEEPSEEK_API_KEY="你的 DeepSeek API Key"
   ```

   RAI 的 OpenAI 适配器使用 `OPENAI_API_KEY`。PC Agent 会自动完成
   `DEEPSEEK_API_KEY -> OPENAI_API_KEY` 映射，也可以直接设置
   `OPENAI_API_KEY`。

4. Orin 上 VoteNet 正在发布 `/detect_bbox3d`，并运行 socket bridge。
5. Orin 上 Nav2 已启动，`/navigate_to_pose` action server 可用。

## 配置模型

`config.toml` 中的聊天模型配置为：

```toml
[vendor]
simple_model = "openai"
complex_model = "openai"

[openai]
simple_model = "deepseek-chat"
complex_model = "deepseek-chat"
base_url = "https://api.deepseek.com"
```

DeepSeek 不提供 embeddings 接口，因此仓库把 embeddings vendor 保留为
Ollama 备用配置；本示例不会初始化 embeddings。

## 启动

### Orin：VoteNet socket bridge

```bash
source /opt/ros/foxy/setup.bash
source ~/mm3d_ws/install/setup.bash
ros2 run mmdet3d_ros2 detect_bbox3d_socket_bridge
```

bridge 默认监听 `0.0.0.0:8765`，需要保持该终端运行。

### PC 或 RAI 容器：启动 Agent

在 RAI 工作空间根目录执行：

```bash
source /opt/ros/humble/setup.bash
source setup_shell.sh
export ROS_DOMAIN_ID=0
export DEEPSEEK_API_KEY="你的 DeepSeek API Key"

python -m examples.pc_agent.main \
  --detection-source socket \
  --socket-host <orin-ip> \
  --socket-port 8765 \
  --frame-id map \
  --target-frame map
```

如果 Agent 和 bridge 位于同一台机器并使用 host network，
`--socket-host 127.0.0.1` 即可。跨机器运行时必须填写 Orin 的实际 IP。

也可以启动 Streamlit：

```bash
streamlit run examples/pc_agent/streamlit_app.py
```

### Docker

```bash
docker build -t pc-agent -f docker/pc_agent.dockerfile .
docker run -d --name pc-agent --network=host --restart=always \
  -e DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
  pc-agent
```

浏览器访问 `http://<主机IP>:8501`。

## 指令示例

| 用户输入 | Agent 行为 |
|---|---|
| `周围有什么` | 获取并列出当前 VoteNet 检测结果 |
| `找椅子` | 获取椅子坐标，然后调用 Nav2 导航 |
| `去桌子那里` | 获取桌子坐标，然后调用 Nav2 导航 |
| `去 map 坐标 x=-4.2, y=2.97` | 直接调用坐标导航工具，不经过 VoteNet 检测 |
| `停下` | 取消当前导航任务 |

导航坐标必须来自检测结果或用户提供的 `map` 坐标，Agent 不应编造坐标。
导航工具只需要 `x、y`。点击页面的“自动定位”后，若 Agent 已收到（包括点击前收到的）最新 `/initialpose`，立即将它作为已确认的定位起点，不控制小车旋转；后续 `/amcl_pose` 用于实时更新坐标。小车静止、AMCL 暂时没有新消息时保留最后确认位置。如果没有收到初始位姿，则自动调用 AMCL 全局重定位并让小车原地旋转搜索，只有该分支要求连续 3 帧收敛。使用 2D Pose 定位时，导航目标朝向沿用用户画箭头选定的 `yaw`；没有 2D Pose 时才根据当前位置到目标计算 `yaw`。二维目标的 `z` 固定为 `0`。定位失败、超时或取消后，导航会拒绝启动，直到再次定位成功。

检测确认门槛是 2 秒内 3 次类别和位置一致的观测，并允许短暂漏帧；socket
bridge 和 ROS 直连模式使用同一套门槛。目标导航只使用最近一轮检测对话对应的
结构化快照，不解析普通聊天文字，也不会回退到更早轮次。从未检测时工具自动检测
一次再导航；最近一轮失败或没有匹配目标时直接提示未找到。
“周围有什么”的回复按精确相对方向和类别统计数量，例如“小车左前方有 2 把椅子、右侧有 1 个柜子”，随后保留每个目标的 `map` 坐标和置信度。
检测物体中心通常落在代价地图障碍物内。物体导航按检测框尺寸计算接近距离：小物体至少保留 `0.6 m`，大物体使用“平面半径 + 0.35 m”；随后围绕目标生成多个候选点，并通过 Nav2 global costmap 避开障碍、未知区域和地图外位置。costmap 不可用时才回退到目标近侧点；用户直接输入的地图坐标不做偏移。
车头附近会进一步区分“前方偏左、正前方、前方偏右”。多个同类目标匹配时按方向选择，例如“去前方偏右的椅子”，不要求用户选择检测序号。

Streamlit 发送导航目标后会显示“正在导航”，并根据 Nav2 Action 的最终结果在
对话区提示导航完成、取消或失败。检测列表中的每一项统一显示 `map` 坐标，
不会只给某一项添加特殊的“导航坐标”前缀。

## 命令行参数

```text
--detection-source {socket,ros}  检测来源，默认 socket
--socket-host HOST               bridge 地址，默认 127.0.0.1
--socket-port PORT               bridge 端口，默认 8765
--nav-action NAME                Nav2 action，默认 navigate_to_pose
--frame-id FRAME                 导航坐标系，默认 map
--target-frame FRAME             检测结果转换目标坐标系，默认 map
--model {simple_model,complex_model}
--vendor openai                 覆盖 config.toml 中的 vendor
--timeout SECONDS                检测等待时间，默认 10
```

直接使用 ROS 2 检测话题时，可将 `--detection-source` 改为 `ros`，并确保
PC 端安装了 `vision_msgs` 且 ROS 2 DDS 网络发现正常。

## 数据与坐标

socket bridge 发送逐行 JSON。每帧至少包含 `frame_id` 和 `detections`，检测项
包含 `class_id`、`score` 和 `center.x/y/z`。当 `frame_id` 不是 `map` 时，Agent
通过 TF 将坐标转换到 `--target-frame`，再把结果交给导航工具。
