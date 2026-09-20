#!/usr/bin/env python3
"""PC Agent Streamlit 控制台。"""

import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_workspace_root = Path(__file__).resolve().parents[2]
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from examples.pc_agent.agent import create_pc_agent
from examples.pc_agent.localization import (
    LocalizationError,
    cancel_global_localization,
    get_localization_status,
    start_global_localization,
)
from examples.pc_agent.detection_selection import is_detection_navigation_request
from examples.pc_agent.tools import (
    get_detection_snapshot,
    get_navigation_status,
)


st.set_page_config(
    page_title="无人车控制台",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }
    [data-testid="stMetric"] {
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        background: #ffffff;
        min-width: 0;
        overflow: hidden;
    }
    [data-testid="stColumn"] {
        min-width: 0;
    }
    .status-ready {
        color: #047857;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── 语音 ──────────────────────────────────────────────────────────────────

@st.cache_resource
def get_asr():
    try:
        from rai_s2s.asr.models.local_whisper import LocalWhisper

        return LocalWhisper(model_name="tiny")
    except Exception:
        return None


def transcribe(audio_bytes: bytes):
    asr = get_asr()
    if asr is None:
        return None

    import io

    import numpy as np
    import soundfile as sf

    data, _ = sf.read(io.BytesIO(audio_bytes))
    if data.ndim > 1:
        data = data[:, 0]
    return asr.transcribe(data.astype(np.float32))


def _display_timezone():
    """Return the timezone used for timestamps shown in the web UI."""
    timezone_name = os.environ.get("PC_AGENT_TIMEZONE", "Asia/Shanghai")
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        return datetime.now().astimezone().tzinfo or timezone.utc


def _navigation_target(status: dict) -> str:
    try:
        return (
            f"x={float(status.get('x')):.3f} m，"
            f"y={float(status.get('y')):.3f} m"
        )
    except (TypeError, ValueError):
        return "目标坐标未知"


def _navigation_label(status: dict) -> str:
    return {
        "idle": "未导航",
        "navigating": "正在导航",
        "canceling": "正在取消",
        "completed": "已完成",
        "canceled": "已取消",
        "failed": "失败",
    }.get(status.get("status"), "未知")


def _append_navigation_notice(status: dict) -> bool:
    """Append one terminal navigation message to the conversation."""
    terminal_states = {"completed", "canceled", "failed"}
    if status.get("status") not in terminal_states:
        return False

    event_id = str(status.get("event_id") or "")
    if not event_id or event_id == st.session_state.get("last_navigation_notice"):
        return False
    if "messages" not in st.session_state:
        return False

    target = _navigation_target(status)
    messages = {
        "completed": f"导航完成：小车已到达 map 坐标 {target}。",
        "canceled": "导航已取消：小车已停止。",
        "failed": f"导航未完成：目标 {target}。{status.get('message', '')}",
    }
    notice_time = datetime.now(_display_timezone()).strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.messages.append(
        AIMessage(
            content=messages[status["status"]],
            additional_kwargs={
                "time": notice_time,
                "navigation_event": True,
            },
        )
    )
    st.session_state.last_navigation_notice = event_id
    return True


def _render_navigation_status(status: dict) -> None:
    state = status.get("status")
    target = _navigation_target(status)
    if state == "navigating":
        st.info(f"正在导航中 · 目标 map 坐标：{target}")
    elif state == "canceling":
        st.warning(f"正在取消导航 · 当前目标：{target}")
    elif state == "completed":
        st.success(f"导航完成 · 已到达 map 坐标：{target}")
    elif state == "canceled":
        st.warning("导航已取消 · 小车已停止")
    elif state == "failed":
        st.error(status.get("message") or "导航失败")


def render_robot_status() -> None:
    localization_status = get_localization_status()
    localization_state = localization_status.get("status")
    pose = localization_status.get("pose")
    navigation_status = get_navigation_status()
    if _append_navigation_notice(navigation_status):
        # The notice belongs in the normal chat area, so redraw the full app
        # after adding it from the live status fragment.
        st.rerun()

    st.markdown("#### 小车实时位置")
    if localization_state == "localizing":
        st.info("正在定位")
    elif localization_state == "localized" and pose:
        updated_at = datetime.fromtimestamp(
            float(pose["updated_at"]),
            _display_timezone(),
        ).strftime("%H:%M:%S")
        columns = st.columns(4, gap="small")
        columns[0].metric("X（map，米）", f"{pose['x']:.3f}")
        columns[1].metric("Y（map，米）", f"{pose['y']:.3f}")
        columns[2].metric("Yaw（弧度）", f"{pose['yaw']:.3f}")
        columns[3].metric("本地更新时间", updated_at)
    elif localization_state == "failed":
        detail = localization_status.get("message") or "AMCL 定位失败"
        st.error(f"暂无有效位置：{detail}")
    else:
        st.info("暂无有效位置")

    _render_navigation_status(navigation_status)

    status_columns = st.columns(4, gap="small")
    status_columns[0].metric("Agent", "READY")
    status_columns[1].metric("坐标系", "map")
    status_columns[2].metric(
        "定位状态",
        localization_status.get("label", "等待定位"),
    )
    status_columns[3].metric("导航状态", _navigation_label(navigation_status))


@st.fragment(run_every="1s")
def render_live_robot_status() -> None:
    render_robot_status()


@st.fragment(run_every="1s")
def render_localization_controls() -> None:
    localization = get_localization_status()
    navigation = get_navigation_status()
    localization_busy = localization.get("status") == "localizing"
    navigation_busy = navigation.get("status") in {"navigating", "canceling"}

    control_columns = st.columns(2, gap="small")
    locate_submitted = control_columns[0].button(
        "自动定位",
        key="start_global_localization",
        type="primary",
        use_container_width=True,
        disabled=localization_busy or navigation_busy,
    )
    cancel_submitted = control_columns[1].button(
        "取消定位",
        key="cancel_global_localization",
        use_container_width=True,
        disabled=not localization_busy,
    )

    if locate_submitted:
        try:
            started = start_global_localization()
        except LocalizationError as exc:
            st.error(f"无法启动自动定位：{exc}")
        else:
            if started:
                st.info("自动定位已启动：有 2D Pose Estimate 时直接使用，否则执行旋转搜索")
            else:
                st.info("自动定位正在进行中")

    if cancel_submitted:
        try:
            canceled = cancel_global_localization()
        except LocalizationError as exc:
            st.error(f"无法取消自动定位：{exc}")
        else:
            if canceled:
                st.info("自动定位已取消，小车正在停止")
            else:
                st.info("当前没有正在进行的自动定位")

    localization = get_localization_status()
    st.caption(f"AMCL：{localization.get('label', '等待定位')}")


# ── 状态与 Agent ───────────────────────────────────────────────────────────

def queue_prompt(prompt: str) -> None:
    st.session_state.pending_prompt = prompt


def queue_cancel() -> None:
    st.session_state.pending_prompt = "停下"
    st.session_state.direct_cancel = True


def clear_session() -> None:
    st.session_state.messages = [AIMessage(content="会话已清空。")]
    st.session_state.tool_events = []
    st.session_state.execution_records = []
    st.session_state.pop("pending_prompt", None)


def initialize_agent() -> None:
    st.session_state.setdefault("execution_records", [])
    if "agent" in st.session_state:
        return

    with st.spinner("正在连接 LLM 和 ROS 2..."):
        try:
            agent, tools, connector = create_pc_agent(
                detection_source="socket",
                socket_host="127.0.0.1",
                socket_port=8765,
                frame_id="map",
                target_frame="map",
            )
        except Exception as exc:
            st.session_state.agent_error = str(exc)
            return

    st.session_state.agent = agent
    st.session_state.pop("agent_error", None)
    st.session_state.tools = tools
    st.session_state.connector = connector
    st.session_state.messages = [
        AIMessage(content="已连接。请先自动定位，再输入地图坐标或目标类别。")
    ]
    st.session_state.tool_events = []
    st.session_state.last_audio_hash = None


def message_content(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    # Text blocks are common with newer LangChain model adapters.
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts) if parts else json.dumps(content, ensure_ascii=False)


def collect_execution(record: dict, messages: list) -> None:
    """Keep tool results with their calls, including repeated calls to one tool."""
    tools_by_call_id = {}
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                tool_record = {
                    "id": call.get("id", ""),
                    "name": call.get("name", "tool"),
                    "args": call.get("args", {}),
                    "result": None,
                    "status": "pending",
                }
                record["tools"].append(tool_record)
                if tool_record["id"]:
                    tools_by_call_id[tool_record["id"]] = tool_record
        elif isinstance(message, ToolMessage):
            call_id = message.tool_call_id
            tool_record = tools_by_call_id.get(call_id)
            if tool_record is None:
                tool_record = {
                    "id": call_id,
                    "name": message.name or "tool",
                    "args": None,
                    "result": None,
                    "status": "pending",
                }
                record["tools"].append(tool_record)
            tool_record["result"] = message_content(message)
            tool_record["status"] = getattr(message, "status", "success")

    if messages and isinstance(messages[-1], AIMessage) and not messages[-1].tool_calls:
        record["reply"] = message_content(messages[-1])


def invoke_agent(prompt: str) -> None:
    """Store one request, its tool results and the displayed reply together."""
    record = {
        "id": uuid4().hex,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": prompt,
        "tools": [],
        "reply": "",
        "error": "",
    }
    metadata = {"execution_id": record["id"], "time": record["time"]}
    st.session_state.messages.append(
        HumanMessage(content=prompt, additional_kwargs=metadata)
    )

    try:
        direct_cancel = st.session_state.pop("direct_cancel", False)
        direct_cancel = direct_cancel or prompt.strip() in {
            "停下", "停止", "取消导航", "取消当前导航"
        }
        if direct_cancel:
            cancel_tool = next(
                tool for tool in st.session_state.tools
                if tool.name == "cancel_navigation"
            )
            call_id = f"cancel-{uuid4().hex}"
            tool_output = cancel_tool.invoke({
                "type": "tool_call",
                "name": "cancel_navigation",
                "args": {},
                "id": call_id,
            })
            result = {"messages": [
                AIMessage(content="", tool_calls=[{
                    "id": call_id,
                    "name": "cancel_navigation",
                    "args": {},
                }]),
                tool_output,
                AIMessage(content=str(tool_output.content)),
            ]}
        elif (
            get_detection_snapshot()
            and is_detection_navigation_request(prompt)
        ):
            snapshot_tool = next(
                tool for tool in st.session_state.tools
                if tool.name == "navigate_to_detected_target"
            )
            call_id = f"snapshot-navigation-{uuid4().hex}"
            tool_output = snapshot_tool.invoke({
                "type": "tool_call",
                "name": "navigate_to_detected_target",
                "args": {"target": prompt},
                "id": call_id,
            })
            result = {"messages": [
                AIMessage(content="", tool_calls=[{
                    "id": call_id,
                    "name": "navigate_to_detected_target",
                    "args": {"target": prompt},
                }]),
                tool_output,
                AIMessage(content=str(tool_output.content)),
            ]}
        else:
            conversation = list(st.session_state.messages[-12:])
            result = st.session_state.agent.invoke(
                {"messages": conversation}
            )
        collect_execution(record, result.get("messages", []))
    except Exception as exc:
        record["error"] = str(exc)
        record["reply"] = f"Agent 执行失败：{exc}"

    if not record["reply"]:
        record["reply"] = "未收到 Agent 最终回复，请查看本次工具返回。"
    st.session_state.messages.append(
        AIMessage(content=record["reply"], additional_kwargs=metadata)
    )
    st.session_state.execution_records.append(record)
    st.session_state.execution_records = st.session_state.execution_records[-20:]


def render_tool_results(record: dict) -> None:
    for tool_record in record["tools"]:
        st.markdown(f"**{tool_record['name']}**")
        if tool_record["args"] is not None:
            st.json(tool_record["args"])
        if tool_record["result"] is None:
            st.warning("未收到工具返回结果")
        else:
            st.caption("工具原始返回")
            st.text(tool_record["result"])


def render_chat() -> None:
    for message in st.session_state.messages:
        execution_id = message.additional_kwargs.get("execution_id")
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                if execution_id:
                    st.caption(f"{message.additional_kwargs['time']} · {execution_id[:8]}")
                st.write(message.content)
        elif isinstance(message, AIMessage) and message.content:
            with st.chat_message("assistant"):
                st.write(message.content)


def render_activity() -> None:
    records = list(reversed(st.session_state.execution_records))
    legacy_events = st.session_state.get("tool_events", [])
    if not records and not legacy_events:
        st.info("暂无执行记录")
        return

    for record in records:
        with st.expander(
            f"{record['time']} · {record['id'][:8]} · {record['prompt']}"
        ):
            if record["error"]:
                st.error(record["error"])
            render_tool_results(record)
            st.markdown("**Agent 对话回复**")
            st.write(record["reply"])

    # Old flat events lack request IDs; do not guess their associated reply.
    if legacy_events:
        with st.expander("旧版未分组记录"):
            for event in reversed(legacy_events):
                st.caption(f"{event['time']} · {event['name']} · {event['kind']}")
                if event["kind"] == "call":
                    st.json(event["content"])
                else:
                    st.text(event["content"])


initialize_agent()
if "agent_error" in st.session_state:
    st.error(f"Agent 初始化失败：{st.session_state.agent_error}")
    st.stop()


# ── 侧栏控制 ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🚗 无人车控制台")
    st.markdown('<span class="status-ready">● Agent 已就绪</span>', unsafe_allow_html=True)
    st.divider()

    st.markdown("### 定位")
    render_localization_controls()

    st.divider()
    st.markdown("### 坐标导航")
    # Keep the fields empty so the placeholder remains visible until the user
    # enters a target. A regular button also avoids Streamlit's English form
    # submission hint that appears below focused form inputs.
    x_text = st.text_input(
        "目标 X（map，米）",
        placeholder="例如：-0.550",
        key="target_x",
    )
    y_text = st.text_input(
        "目标 Y（map，米）",
        placeholder="例如：2.340",
        key="target_y",
    )
    navigate_submitted = st.button(
        "发送目标点",
        key="send_coordinate_goal",
        type="primary",
        use_container_width=True,
    )

    if navigate_submitted:
        try:
            x = float(x_text.strip())
            y = float(y_text.strip())
        except (AttributeError, TypeError, ValueError):
            st.error("请输入有效的 X、Y 坐标，例如：X=-0.550，Y=2.340")
        else:
            if not all(math.isfinite(value) for value in (x, y)):
                st.error("X、Y 坐标必须是有限数字")
            else:
                queue_prompt(
                    f"导航到地图坐标：x={x:.3f} m，y={y:.3f} m（朝向按小车当前位置自动计算）"
                )

    st.divider()
    st.markdown("### 快捷指令")
    quick_commands = [
        ("周围有什么", "查看检测结果"),
        ("找椅子", "导航到椅子"),
        ("找桌子", "导航到桌子"),
        ("停下", "取消当前导航"),
    ]
    for command, label in quick_commands:
        st.button(
            label,
            key=f"quick_{command}",
            use_container_width=True,
            on_click=queue_cancel if command == "停下" else queue_prompt,
            args=() if command == "停下" else (command,),
        )

    st.divider()
    st.markdown("### 语音")
    audio = st.audio_input("录音", key="audio_input")
    if audio:
        audio_bytes = audio.getvalue()
        audio_hash = hashlib.sha1(audio_bytes).hexdigest()
        if audio_hash != st.session_state.get("last_audio_hash"):
            st.session_state.last_audio_hash = audio_hash
            transcript = None
            try:
                transcript = transcribe(audio_bytes)
            except Exception as exc:
                st.warning(f"语音识别失败：{exc}")
            else:
                transcript = transcript.strip() if transcript else ""
                if transcript:
                    queue_prompt(transcript)
                    st.toast(f"已识别：{transcript}")
                else:
                    st.warning("语音识别不可用")

    st.divider()
    st.caption("ROS 2 / Nav2")
    st.caption("Frame: map")
    st.caption("Detection socket: 127.0.0.1:8765")
    if st.button("清空会话", use_container_width=True):
        clear_session()
        st.rerun()


# ── 主区 ──────────────────────────────────────────────────────────────────

st.title("无人车控制台")
st.caption("PC Agent  ·  DeepSeek  ·  Nav2")
render_live_robot_status()

st.divider()

chat_tab, activity_tab = st.tabs(["对话", "执行记录"])
with chat_tab:
    render_chat()
with activity_tab:
    render_activity()


pending_prompt = st.session_state.pop("pending_prompt", None)
typed_prompt = st.chat_input("输入指令，例如：导航到地图坐标 x=0.913 m，y=10.206 m")
prompt = typed_prompt or pending_prompt

if prompt:
    with st.spinner("Agent 执行中..."):
        invoke_agent(prompt)
    st.rerun()
