#!/usr/bin/env python3
"""PC Agent Streamlit 控制台。"""

import hashlib
import sys
import time
from pathlib import Path

_workspace_root = Path(__file__).resolve().parents[2]
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from examples.pc_agent.agent import create_pc_agent


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


# ── 状态与 Agent ───────────────────────────────────────────────────────────

def queue_prompt(prompt: str) -> None:
    st.session_state.pending_prompt = prompt


def clear_session() -> None:
    st.session_state.messages = [AIMessage(content="会话已清空。")]
    st.session_state.tool_events = []
    st.session_state.pop("pending_prompt", None)


def initialize_agent() -> None:
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
        AIMessage(content="已连接。可以输入地图坐标、目标类别，或发送停止指令。")
    ]
    st.session_state.tool_events = []
    st.session_state.last_audio_hash = None


def invoke_agent(prompt: str) -> None:
    """执行一次 Agent 调用，并把工具调用保存到执行记录。"""
    st.session_state.messages.append(HumanMessage(content=prompt))
    started_at = time.strftime("%H:%M:%S")

    try:
        result = st.session_state.agent.invoke(
            {"messages": [HumanMessage(content=prompt)]}
        )
    except Exception as exc:
        message = f"Agent 执行失败：{exc}"
        st.session_state.messages.append(AIMessage(content=message))
        st.session_state.tool_events.append(
            {
                "time": started_at,
                "kind": "error",
                "name": "agent",
                "content": str(exc),
            }
        )
        return

    final_text = ""
    events = []
    for message in result.get("messages", []):
        if isinstance(message, AIMessage):
            for call in getattr(message, "tool_calls", []) or []:
                events.append(
                    {
                        "time": started_at,
                        "kind": "call",
                        "name": call.get("name", "tool"),
                        "content": call.get("args", {}),
                    }
                )
            if message.content and not getattr(message, "tool_calls", None):
                final_text = str(message.content)
        elif isinstance(message, ToolMessage):
            events.append(
                {
                    "time": started_at,
                    "kind": "result",
                    "name": getattr(message, "name", None) or "tool",
                    "content": str(message.content),
                }
            )

    if not final_text:
        final_text = "操作已执行。"
    st.session_state.messages.append(AIMessage(content=final_text))
    st.session_state.tool_events.extend(events)
    st.session_state.tool_events = st.session_state.tool_events[-40:]


def render_chat() -> None:
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage) and message.content:
            with st.chat_message("assistant"):
                st.write(message.content)


def render_activity() -> None:
    events = list(reversed(st.session_state.tool_events))
    if not events:
        st.info("暂无执行记录")
        return

    for event in events:
        kind = event["kind"]
        name = event["name"]
        content = event["content"]
        title = {
            "call": f"调用 {name}",
            "result": f"{name} 返回结果",
            "error": "Agent 错误",
        }.get(kind, name)
        with st.expander(f"{event['time']}  ·  {title}", expanded=False):
            if kind == "call":
                st.json(content)
            else:
                st.text(content)


initialize_agent()
if "agent_error" in st.session_state:
    st.error(f"Agent 初始化失败：{st.session_state.agent_error}")
    st.stop()


# ── 侧栏控制 ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🚗 无人车控制台")
    st.markdown('<span class="status-ready">● Agent 已就绪</span>', unsafe_allow_html=True)
    st.divider()

    st.markdown("### 坐标导航")
    with st.form("coordinate_navigation", clear_on_submit=False):
        x = st.number_input("X (m)", value=0.0, step=0.1, format="%.3f")
        y = st.number_input("Y (m)", value=0.0, step=0.1, format="%.3f")
        yaw = st.number_input("Yaw (rad)", value=0.0, step=0.1, format="%.3f")
        navigate_submitted = st.form_submit_button(
            "发送导航目标",
            type="primary",
            use_container_width=True,
        )

    if navigate_submitted:
        queue_prompt(f"去 map 坐标 x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}")

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
            on_click=queue_prompt,
            args=(command,),
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

status_columns = st.columns(4)
status_columns[0].metric("Agent", "READY")
status_columns[1].metric("坐标系", "map")
status_columns[2].metric("检测来源", "Socket")
status_columns[3].metric("Action", "Nav2")

st.divider()

chat_tab, activity_tab = st.tabs(["对话", "执行记录"])
with chat_tab:
    render_chat()
with activity_tab:
    render_activity()


pending_prompt = st.session_state.pop("pending_prompt", None)
typed_prompt = st.chat_input("输入指令，例如：去 map 坐标 x=0.913, y=10.206")
prompt = typed_prompt or pending_prompt

if prompt:
    with st.spinner("Agent 执行中..."):
        invoke_agent(prompt)
    st.rerun()
