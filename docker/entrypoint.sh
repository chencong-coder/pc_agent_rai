#!/bin/bash
if [[ -z "${OPENAI_API_KEY:-}" && -n "${DEEPSEEK_API_KEY:-}" ]]; then
    export OPENAI_API_KEY="${DEEPSEEK_API_KEY}"
fi

source /opt/ros/humble/setup.bash
source /rai/install/setup.bash
exec streamlit run /rai/examples/pc_agent/streamlit_app.py --server.address=0.0.0.0
