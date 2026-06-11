#!/bin/bash
source /opt/ros/humble/setup.bash
source /rai/install/setup.bash
exec streamlit run /rai/examples/pc_agent/streamlit_app.py --server.address=0.0.0.0
