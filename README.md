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
- ✅ TF 坐标自动变换 (`rslidar/velodyne` → `map`)
- ✅ TF 转换失败时拒绝返回激光雷达坐标，避免发送错误导航目标
- ✅ 导航只需要 `x、y`，根据小车当前位置自动计算到目标的朝向
- ✅ Streamlit 执行记录按请求分组，显示工具原始返回和 Agent 回复
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

## 坐标与导航

进入 Streamlit 后需要先点击“自动定位”。AMCL 连续 3 帧收敛后，页面才显示小车
的 `map` 坐标 `x/y`、朝向 `yaw`（弧度）和本地更新时间；之后随有效 AMCL 位姿
实时更新。定位中显示“正在定位”，未定位、失败、超时、取消或数据过期时显示
“暂无有效位置”，不会保留无效坐标。时间默认按 `Asia/Shanghai` 显示；如部署地区
不同，可通过 `PC_AGENT_TIMEZONE` 指定时区。
网页需要 Streamlit 1.40 或更新版本（同时支持实时刷新和已有的录音控件）。

发送坐标后，顶部状态栏显示“正在导航”，并持续显示目标坐标；Nav2 返回成功、
取消或失败结果后，状态栏会更新，且对话区追加对应的“导航完成”“导航已取消”
或失败消息。

VoteNet 输出的检测中心通常在激光雷达坐标系，例如 `rslidar` 或
`velodyne`。Agent 会读取检测消息的 `frame_id`，通过 TF 转换到 `map` 后，
才把坐标交给大模型和导航工具。

检测结果中的 `x、y` 是目标在地图中的平面位置；第三个值 `z` 是检测框中心的
高度信息，主要用于感知显示，不作为二维 Nav2 的目标高度。导航时只使用 `x、y`。

必须存在完整的 TF 链：

```text
map -> odom -> base_link -> velodyne
```

检查转换是否可用：

```bash
ros2 run tf2_ros tf2_echo map velodyne
```

如果 TF 不可用，Agent 会提示定位或 TF 问题，并且不会把原始雷达坐标当作
Nav2 目标。页面的“自动定位”会优先使用 Agent 已收到的最新 RViz 2D Pose Estimate，
即使它是在点击按钮前发布的；未收到初始位姿时才会触发 AMCL 全局定位并让小车
原地旋转搜索。导航前必须先完成页面定位。

坐标导航工具只接收地图坐标 `x、y`：

```text
你: 去 map 坐标 x=-4.2, y=2.97
```

发送前，工具会读取最近一次已确认且仍有效的 AMCL 位姿，用目标点和该当前位置
计算朝向：

```text
yaw = atan2(target_y - robot_y, target_x - robot_x)
```

随后将这个朝向转换成 Nav2 所需的四元数。用户不需要输入 `yaw` 或 `z`；二维
导航会把目标高度固定为 `z=0`。

---

## Orin 启动顺序

在 Orin 上启动 VoteNet 和 socket bridge：

```bash
source /opt/ros/foxy/setup.bash
source ~/mm3d_ws/install/setup.bash

ros2 run mmdet3d_ros2 detect_bbox3d_socket_bridge
```

确认 bridge 输出类似：

```text
Bridging /detect_bbox3d to tcp://0.0.0.0:8765
```

在运行 PC Agent 的 RAI 容器中启动：

```bash
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
source /rai/.venv/bin/activate

export DEEPSEEK_API_KEY="你的 DeepSeek API Key"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"

python -m examples.pc_agent.main \
  --detection-source socket \
  --socket-host 127.0.0.1 \
  --socket-port 8765 \
  --frame-id map \
  --target-frame map
```

启动网页控制台：

```bash
streamlit run examples/pc_agent/streamlit_app.py \
  --server.address=0.0.0.0 \
  --server.port=8501
```

如果 Agent 和 socket bridge 不在同一台机器，把 `--socket-host` 改成 Orin 的
实际 IP，并确保 TCP 端口 `8765` 可访问。

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
Agent: 检测到 chair，map 坐标 (-5.63, 2.18)，置信度 0.93...

你: 找床
Agent: 已找到床，正在导航前往 map 坐标 (2.12m, -1.62m)...

你: 停下
Agent: 导航已取消。
```

## License

Apache 2.0
