#!/usr/bin/env python3
"""
PC Agent - Streamlit 前端

用法:
    source setup_shell.sh
    streamlit run examples/pc_agent/streamlit_app.py
"""

import json
import sys
from pathlib import Path

_workspace_root = Path(__file__).resolve().parents[2]
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage

from examples.pc_agent.agent import create_pc_agent

st.set_page_config(page_title="无人车控制助手", page_icon="🚗", layout="wide")

# ─── 初始化 Agent ────────────────────────────────────────────────────────

if "agent" not in st.session_state:
    with st.spinner("正在连接 Ollama 和 ROS 2..."):
        agent, tools, connector = create_pc_agent()
        st.session_state.agent = agent
        st.session_state.connector = connector
    st.session_state.messages = [
        AIMessage(content="你好！我是无人车控制助手。\n\n你可以说「周围有什么」「找椅子」「去桌子那里」「停下」等指令。")
    ]
    st.session_state.tool_log = []  # {tool, args, result}

agent = st.session_state.agent

# ─── 侧边栏 ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🚗 无人车控制助手")
    st.caption("LLM 决策 → Agent 执行 → Orin 动作")
    st.divider()
    st.markdown("**试试这些:**")
    for cmd in ["周围有什么", "找床", "去椅子那里", "停下"]:
        st.button(cmd, key=f"btn_{cmd}", use_container_width=True,
                  on_click=lambda c=cmd: st.session_state.update(pending=c))
    st.divider()
    st.subheader("🔧 工具调用记录")
    log_container = st.container()

# ─── 聊天界面 ────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# ─── 渲染工具日志 ────────────────────────────────────────────────────────

with log_container:
    for entry in reversed(st.session_state.tool_log):
        icon = {"get_detections": "📷", "navigate_to_coordinates": "🧭",
                "cancel_navigation": "🛑"}.get(entry["tool"], "🔧")
        with st.expander(f"{icon} {entry['tool']}", expanded=len(st.session_state.tool_log) <= 3):
            st.caption(f"参数: `{entry['args']}`")
            st.code(entry["result"][:600])

# ─── 处理输入 ────────────────────────────────────────────────────────────

# 处理侧边栏快捷按钮
if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")
else:
    prompt = st.chat_input("输入指令，例如「周围有什么」...")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.chat_message("assistant"):
        status = st.status("Agent 思考中...", expanded=False)
        state = {"messages": [HumanMessage(content=prompt)]}
        result = agent.invoke(state)
        result_msgs = result.get("messages", [])

        ai_texts = []
        last_tool_name = "?"
        last_tool_args = ""
        for m in result_msgs:
            # 检测 JSON 工具调用
            if hasattr(m, "content") and m.content and not isinstance(m, (HumanMessage, SystemMessage)):
                content = m.content.strip()
                try:
                    if content.startswith("```"):
                        json_str = content.split("```")[1]
                        if json_str.startswith("json"):
                            json_str = json_str[4:]
                        obj = json.loads(json_str)
                    else:
                        obj = json.loads(content)
                    if "tool" in obj:
                        last_tool_name = obj["tool"]
                        last_tool_args = str(obj.get("args", {}))
                        continue
                except Exception:
                    pass
                # 不是 JSON 工具调用 → 是 AI 回复
                if not content.startswith('{"tool"'):
                    ai_texts.append(content)

            elif isinstance(m, ToolMessage):
                st.session_state.tool_log.append({
                    "tool": last_tool_name,
                    "args": last_tool_args,
                    "result": m.content,
                })
                with status:
                    st.caption(f"✓ {last_tool_name} 完成")

        status.update(label="完成", state="complete", expanded=False)

        final = "\n\n".join(ai_texts) if ai_texts else "操作已完成。"
        st.session_state.messages.append(AIMessage(content=final))
        st.write(final)

    st.rerun()
