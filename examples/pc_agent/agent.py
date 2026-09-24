# Copyright (C) 2025 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PC Agent — 原生 Function Calling（DeepSeek OpenAI 兼容接口）
"""

import logging
import os
from typing import List

from langchain_core.tools import BaseTool
from rai import get_llm_model
from rai.agents.langchain.core.react_agent import create_react_runnable

from .localization import LocalizationManager
from .tools import (
    CancelNavigationTool,
    GetDetectionsTool,
    NavigateToCoordinatesTool,
    NavigateToDetectedTargetTool,
)

logger = logging.getLogger(__name__)


def _prepare_llm_credentials() -> None:
    """Map a DeepSeek-specific key to the variable used by RAI's OpenAI adapter."""
    if not os.environ.get("OPENAI_API_KEY"):
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        if deepseek_key:
            os.environ["OPENAI_API_KEY"] = deepseek_key

SYSTEM_PROMPT = """你是一个无人车控制助手。根据用户指令使用工具完成任务。

## 行为规则
- 用户直接提供地图坐标时（例如“去 map 坐标 x=-4.2, y=2.97”或“去 (-4.2, 2.97)”），直接调用 navigate_to_coordinates；不要调用 get_detections，也不要把用户给出的坐标当成编造坐标
- navigate_to_coordinates 只接收目标 map 坐标 x、y；若定位使用了 RViz 2D Pose Estimate，工具沿用用户画箭头选择的 yaw 作为导航目标朝向；没有 2D Pose 时才根据当前位置到目标计算朝向；不要传起点、yaw 或 z
- 两个导航入口都只检查 AMCL 定位质量，不会自行触发全局定位；未定位时提示用户先点击页面上的“自动定位”（未发布 2D Pose Estimate 时，页面自动定位会回退到旋转搜索）
- navigate_to_coordinates 是二维导航；不要把物体检测的高度 z 当成小车导航高度
- "找XX"/"去XX那里": 只使用最近一轮 get_detections 对应的结构化快照，绝不从普通聊天文字解析坐标，也不搜索更早检测轮次。若从未调用过 get_detections，先调用一次再导航；若最近一轮检测失败或没有匹配目标，直接提示未找到，不得回退到旧轮次；不要用 navigate_to_coordinates 替代
- 调用 navigate_to_detected_target 时必须传入 target 字段，例如 {"target": "前方偏右的椅子"}；不要传空参数或自行编造坐标
- 检测物体中心通常是障碍物占用点；navigate_to_detected_target 会自动生成物体前方的安全接近点。回复中必须区分“物体中心坐标”和“实际导航接近点”，不要声称小车会进入物体中心
- "周围有什么": 调 get_detections。回复第一句必须按工具的“方向汇总”精确说明每个方向有几件什么物体，例如“小车左前方有2把椅子，右侧有1个柜子”；随后逐项列出类别、精确方向、map 坐标和置信度
- 方向只能原样使用工具给出的“正前方、前方偏左、左前方、左侧、左后方、正后方、右后方、右侧、右前方、前方偏右、方向未知”，不能省略“偏左/偏右”，不能根据 map 坐标自行推测方向
- 同类别物体位于不同方向时必须分开统计；回复中的数量必须与工具方向汇总及逐项目标数量一致，不得漏掉、合并或编造目标
- "找XX"/"去XX那里": 最终回复中明确写出快照中选中目标的 map 坐标 x、y
- 检测结果中的 z 是物体检测高度，不是 Nav2 导航参数；导航工具只使用 x、y
- "停下": 调 cancel_navigation
- 多个同类物体分别列出，不要合并；导航条件仍匹配多个快照目标时，用工具提供的方向选项让用户说“前方偏左”或“前方偏右”，绝不让用户选择序号，也不要擅自选择
- 没有用户坐标或检测结果时不要编造坐标
- 用中文简短回复"""


def create_pc_agent(
    detection_topic: str = "/detect_bbox3d",
    detection_source: str = "socket",
    socket_host: str = "127.0.0.1",
    socket_port: int = 8765,
    nav_action_name: str = "navigate_to_pose",
    frame_id: str = "map",
    target_frame: str = "map",
    detection_timeout: float = 10.0,
    model_type: str = "complex_model",
    vendor: str | None = None,
    verbose: bool = True,
):
    from rai.communication.ros2.connectors import ROS2Connector

    connector = ROS2Connector(node_name="rai_pc_agent")
    localization_manager = LocalizationManager(connector)

    get_detections_tool = GetDetectionsTool(
        connector=connector,
        topic=detection_topic,
        detection_source=detection_source,
        socket_host=socket_host,
        socket_port=socket_port,
        target_frame=target_frame,
        timeout_sec=detection_timeout,
    )
    tools: List[BaseTool] = [
        get_detections_tool,
        NavigateToCoordinatesTool(
            connector=connector,
            localization_manager=localization_manager,
            frame_id=frame_id,
            action_name=nav_action_name,
        ),
        CancelNavigationTool(
            connector=connector,
            localization_manager=localization_manager,
        ),
    ]
    tools.append(NavigateToDetectedTargetTool(
        navigate_tool=tools[1],
        detection_tool=get_detections_tool,
    ))

    tools_by_name = {t.name: t for t in tools}

    # RAI's OpenAI adapter reads OPENAI_API_KEY. DeepSeek uses the same
    # protocol, so accept DEEPSEEK_API_KEY without exposing it in config.toml.
    _prepare_llm_credentials()
    llm = get_llm_model(model_type=model_type, vendor=vendor, streaming=True)

    if verbose:
        model_id = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))
        logger.info(f"LLM: {model_id}")
        logger.info(f"工具: {list(tools_by_name.keys())}")

    agent = create_react_runnable(llm=llm, tools=tools, system_prompt=SYSTEM_PROMPT)

    return agent, tools, connector
