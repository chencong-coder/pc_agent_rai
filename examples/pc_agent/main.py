#!/usr/bin/env python3
# Copyright (C) 2025 Robotec.AI
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
PC Agent 主程序 - 交互式命令行界面

用法:
    python -m examples.pc_agent.main

    或直接运行:
    python examples/pc_agent/main.py

依赖:
    - ROS 2 (Jazzy/Humble) 已 source
    - RAI 框架已安装 (uv sync)
    - config.toml 已配置 LLM (Ollama/OpenAI/etc.)
    - Orin 端: VOTENET 检测发布到 /detext_bbox3d
    - Orin 端: Nav2 导航运行中

退出: 输入 'quit' 或 'exit' 或按 Ctrl+C
"""

import argparse
import logging
import signal
import sys
from typing import List

# 添加 workspace root 到 path，确保在非编辑安装模式下也能导入
import os
from pathlib import Path

# 确保 rai 包可以被导入
_workspace_root = Path(__file__).resolve().parents[2]
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

import coloredlogs
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

# RAI 导入
from rai import get_tracing_callbacks

from .agent import create_pc_agent

# ─── 日志配置 ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
coloredlogs.install(level="INFO")  # type: ignore
logger = logging.getLogger("pc_agent")


# ─── 欢迎信息 ───────────────────────────────────────────────────────────────

WELCOME_BANNER = """
╔══════════════════════════════════════════════════════════╗
║            RAI PC Agent - 无人车控制助手                  ║
║                                                          ║
║  架构: PC (LLM) ←→ ROS 2 ←→ Orin (Votenet + Nav2)       ║
║                                                          ║
║  输入自然语言指令控制小车:                                 ║
║    • "找椅子" / "去桌子那里"  — 目标导航                   ║
║    • "周围有什么"            — 查看检测结果                ║
║    • "停下" / "停止"         — 取消导航                   ║
║                                                          ║
║  输入 'quit' 或 'exit' 退出                               ║
╚══════════════════════════════════════════════════════════╝
"""


def main():
    parser = argparse.ArgumentParser(
        description="RAI PC Agent - 自然语言控制机器人",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--detection-topic",
        default="/detect_bbox3d",
        help="Votenet 检测结果话题 (默认: /detect_bbox3d)",
    )
    parser.add_argument(
        "--nav-action",
        default="navigate_to_pose",
        help="Nav2 导航 Action 名称 (默认: navigate_to_pose)",
    )
    parser.add_argument(
        "--frame-id",
        default="map",
        help="导航坐标系 (默认: map)",
    )
    parser.add_argument(
        "--target-frame",
        default="map",
        help="检测坐标 TF 变换目标坐标系 (默认: map)",
    )
    parser.add_argument(
        "--model",
        default="complex_model",
        choices=["simple_model", "complex_model"],
        help="使用的 LLM 模型 (默认: complex_model)",
    )
    parser.add_argument(
        "--vendor",
        default=None,
        help="LLM 供应商，不指定则使用 config.toml 中配置",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="检测结果等待超时 (默认: 10s)",
    )

    args = parser.parse_args()

    print(WELCOME_BANNER)

    # ─── 初始化 Agent ────────────────────────────────────────────────

    logger.info("正在初始化 PC Agent...")
    logger.info(f"  LLM Vendor: {args.vendor or 'config.toml 中配置'}")
    logger.info(f"  LLM Model: {args.model}")
    logger.info(f"  检测话题: {args.detection_topic}")
    logger.info(f"  导航 Action: {args.nav_action}")

    try:
        agent, tools, connector = create_pc_agent(
            detection_topic=args.detection_topic,
            nav_action_name=args.nav_action,
            frame_id=args.frame_id,
            target_frame=args.target_frame,
            detection_timeout=args.timeout,
            model_type=args.model,
            vendor=args.vendor,
        )
    except Exception as e:
        logger.error(f"Agent 初始化失败: {e}")
        return 1

    # 获取 tracing callbacks（如果配置了 Langfuse/LangSmith）
    langchain_callbacks = get_tracing_callbacks()

    logger.info("PC Agent 初始化完成！\n")

    # ─── 交互循环 ────────────────────────────────────────────────────

    def shutdown():
        logger.info("正在关闭...")
        try:
            connector.shutdown()
        except Exception:
            pass
        logger.info("再见！")

    # 注册信号处理
    signal.signal(signal.SIGINT, lambda s, f: (shutdown(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda s, f: (shutdown(), sys.exit(0)))

    try:
        while True:
            try:
                user_input = input("\n🧑 你: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                break

            print("🤖 Agent 思考中...", end="", flush=True)

            try:
                # 构建 Agent 输入状态
                state = {"messages": [HumanMessage(content=user_input)]}

                # 调用 Agent
                result = agent.invoke(
                    state,
                    config=RunnableConfig(callbacks=langchain_callbacks),
                )

                # 提取最后一条 AI 回复
                messages = result.get("messages", [])
                ai_response = None
                for msg in reversed(messages):
                    if hasattr(msg, "content"):
                        content = msg.content
                        # 跳过工具调用 JSON 和 ToolMessage
                        if content and not content.strip().startswith('{"tool"'):
                            ai_response = content
                            break

                print("\r" + " " * 20 + "\r", end="")  # 清除"思考中"

                if ai_response:
                    print(f"🤖 Agent: {ai_response}")
                else:
                    print("🤖 Agent: (无文本回复，操作已执行)")

            except Exception as e:
                print("\r" + " " * 20 + "\r", end="")
                logger.error(f"Agent 执行出错: {e}")
                print(f"❌ 错误: {e}")

    finally:
        shutdown()


if __name__ == "__main__":
    main()
