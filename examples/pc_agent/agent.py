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

from .tools import (
    CancelNavigationTool,
    GetDetectionsTool,
    NavigateToCoordinatesTool,
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
- navigate_to_coordinates 的 x、y 是 map 坐标，必须作为数字传入；用户没有提供 yaw 时传 yaw=0
- navigate_to_coordinates 是二维导航，z 使用默认值 0.0；不要把物体检测的高度 z 当成小车导航高度
- "找XX"/"去XX那里": 先调 get_detections 查看周围，找到目标后调用 navigate_to_coordinates，用检测到的真实坐标
- "周围有什么": 调 get_detections，列出所有检测到的物体
- "停下": 调 cancel_navigation
- 多个同类物体选置信度最高的
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

    tools: List[BaseTool] = [
        GetDetectionsTool(
            connector=connector,
            topic=detection_topic,
            detection_source=detection_source,
            socket_host=socket_host,
            socket_port=socket_port,
            target_frame=target_frame,
            timeout_sec=detection_timeout,
        ),
        NavigateToCoordinatesTool(
            connector=connector,
            frame_id=frame_id,
            action_name=nav_action_name,
        ),
        CancelNavigationTool(connector=connector),
    ]

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
