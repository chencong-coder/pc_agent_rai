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
PC Agent - 手动 ReAct Agent（不依赖原生 Function Calling）

支持任何模型，通过文字协议驱动工具调用。
"""

import json
import logging
import re
from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from rai import get_llm_model
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict
from langchain_core.tools import BaseTool

from .tools import (
    CancelNavigationTool,
    GetDetectionsTool,
    NavigateToCoordinatesTool,
)

logger = logging.getLogger(__name__)

# ─── System Prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一个无人车控制助手。你必须通过调用工具来完成用户指令。你只能通过输出 JSON 来调用工具，每次只输出一个 JSON 对象，不要额外文字。

可用工具:
- get_detections: 获取检测结果。可选 object_class 参数过滤类别。
- navigate_to_coordinates: 导航到坐标。参数 x, y, z(默认0), yaw(默认0)。
- cancel_navigation: 取消导航。无参数。

行为规则:
- "找XX"/"去XX那里": 先 get_detections，再用结果坐标 navigate_to_coordinates
- "周围有什么": get_detections，然后列出结果
- "停下": cancel_navigation，然后说"已停止"

示例对话:

用户: 周围有什么
助手: {"tool": "get_detections", "args": {}}

(系统返回: "检测到: bed (2.12, -1.62, 0) 置信度0.70")

助手: 检测到一张床，在坐标(2.12, -1.62)处，置信度70%。

用户: 找床
助手: {"tool": "get_detections", "args": {"object_class": "bed"}}

(系统返回: "检测到: bed (2.12, -1.62, 0) 置信度0.70")

助手: {"tool": "navigate_to_coordinates", "args": {"x": 2.12, "y": -1.62}}

(系统返回: "导航指令已发送")

助手: 已找到床，正在导航前往 (2.12m, -1.62m)。

=== 现在开始。记住: 调工具时只输出 JSON，不调工具时输出中文回复。 ===
"""


# ─── Agent State ──────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: list
    tool_results: list


# ─── 手动 ReAct 循环 ──────────────────────────────────────────────────────

MAX_TOOL_CALLS = 5  # 最多调用工具次数，防止死循环


def react_loop(llm, tools_by_name: dict, state: AgentState, config):
    """手动 ReAct: LLM 输出 → 解析工具调用 → 执行 → 反馈 → 循环"""
    messages = list(state.get("messages", []))
    tool_count = 0

    # 插入系统提示
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

    while tool_count < MAX_TOOL_CALLS:
        # 调用 LLM
        response = llm.invoke(messages, config=config)
        content = response.content.strip()
        logger.info(f"LLM 输出 ({len(content)} 字符): {content[:300]}")

        # 尝试解析 JSON 工具调用
        tool_call = _parse_tool_json(content)

        if tool_call is None:
            # 不是工具调用，就是最终回复
            messages.append(AIMessage(content=content))
            state["messages"] = messages
            return state

        # 是工具调用
        tool_name = tool_call["tool"]
        args = tool_call.get("args", {})

        if tool_name not in tools_by_name:
            result = f"错误: 未知工具 {tool_name}，可用: {list(tools_by_name.keys())}"
        else:
            try:
                tool = tools_by_name[tool_name]
                result = tool._run(**args)
                logger.info(f"工具 {tool_name}({args}) → {result[:200]}")
            except Exception as e:
                result = f"工具执行失败: {e}"

        # 追加对话历史
        messages.append(AIMessage(content=content))           # LLM 的工具调用
        messages.append(ToolMessage(content=result, tool_call_id=str(tool_count)))
        tool_count += 1

    # 超限，强制要求 LLM 总结
    messages.append(HumanMessage(content="请根据以上工具执行结果，用中文简短回复用户。"))
    response = llm.invoke(messages, config=config)
    messages.append(AIMessage(content=response.content))
    state["messages"] = messages
    return state


def _parse_tool_json(content: str) -> dict | None:
    """从 LLM 输出中提取工具调用 JSON"""
    # 去掉 markdown 代码块
    text = content.strip()
    for marker in ["```json", "```"]:
        if text.startswith(marker):
            text = text[len(marker):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    # 尝试直接解析
    try:
        obj = json.loads(text)
        if "tool" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    # 尝试从内容中匹配 JSON
    m = re.search(r'\{[^{]*"tool"\s*:\s*"[^"]+"\s*[,}][^}]*\}', content, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if "tool" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    return None


# ─── Agent 工厂 ───────────────────────────────────────────────────────────

def create_pc_agent(
    detection_topic: str = "/detect_bbox3d",
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

    llm = get_llm_model(model_type=model_type, vendor=vendor, streaming=True)

    # 预热检测订阅 — 提前开始 DDS 发现
    det_tool = tools_by_name.get("get_detections")
    if det_tool:
        det_tool._ensure_subscribed()
        logger.info("检测订阅已预热，等待 DDS 发现...")

    if verbose:
        model_id = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))
        logger.info(f"LLM: {model_id}")
        logger.info(f"工具: {list(tools_by_name.keys())}")

    # 构建 LangGraph（单节点，手动 ReAct 循环在节点内）
    graph = StateGraph(AgentState)
    graph.add_node("react", lambda state, config: react_loop(llm, tools_by_name, state, config))
    graph.add_edge(START, "react")
    graph.add_edge("react", END)
    agent = graph.compile()

    return agent, tools, connector
