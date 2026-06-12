#!/usr/bin/env python3
"""PC Agent - Streamlit 前端"""
import sys
from pathlib import Path

_workspace_root = Path(__file__).resolve().parents[2]
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from examples.pc_agent.agent import create_pc_agent

st.set_page_config(page_title="无人车控制助手", page_icon="🚗")

# ── 语音 ────────────────────────────────────────────────
@st.cache_resource
def get_asr():
    try:
        from rai_s2s.asr.models.local_whisper import LocalWhisper
        return LocalWhisper(model_name="tiny")
    except Exception:
        return None

def transcribe(audio_bytes):
    asr = get_asr()
    if asr is None:
        return None
    import io, soundfile as sf, numpy as np
    data, sr = sf.read(io.BytesIO(audio_bytes))
    if data.ndim > 1:
        data = data[:, 0]
    return asr.transcribe(data.astype(np.float32))

# ── Agent ───────────────────────────────────────────────
if "agent" not in st.session_state:
    with st.spinner("正在连接 LLM 和 ROS 2..."):
        agent, tools, connector = create_pc_agent()
        st.session_state.agent = agent
    st.session_state.messages = [
        AIMessage(content="你好！我是无人车控制助手。试试「周围有什么」「找床」「去椅子那里」「停下」。")
    ]

agent = st.session_state.agent

# ── 侧边栏 ──────────────────────────────────────────────
with st.sidebar:
    st.header("🚗 无人车助手")
    for cmd in ["周围有什么", "找床", "去椅子那里", "停下"]:
        st.button(cmd, key=f"btn_{cmd}", use_container_width=True,
                  on_click=lambda c=cmd: st.session_state.update(pending=c))
    st.divider()
    st.subheader("🎤 语音")
    audio = st.audio_input("点击录音", key="audio_side")

# ── 聊天 ────────────────────────────────────────────────
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        st.chat_message("assistant").write(msg.content)

# ── 输入 ────────────────────────────────────────────────
prompt = None
if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

# 语音识别
if audio:
    t = transcribe(audio.read())
    if t:
        prompt = t
        st.toast(f"已识别: {t}")
    else:
        st.toast("语音未就绪")

# 文字输入
typed = st.chat_input("输入指令...")
prompt = prompt or typed

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            state = {"messages": [HumanMessage(content=prompt)]}
            result = agent.invoke(state)
            result_msgs = result.get("messages", [])

            final = ""
            for m in result_msgs:
                if isinstance(m, ToolMessage):
                    print(f"  ✅ [{m.name}]: {m.content[:200]}")
                elif isinstance(m, AIMessage):
                    if m.content and not getattr(m, 'tool_calls', None):
                        final = m.content
                    for tc in getattr(m, 'tool_calls', []) or []:
                        print(f"  🔧 {tc['name']}({tc.get('args', {})})")

            print(f"  💬 {final[:200]}")

            st.session_state.messages.append(AIMessage(content=final))
        if final:
            st.write(final)

    st.rerun()
