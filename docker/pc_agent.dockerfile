FROM ubuntu:22.04

# ── 系统依赖 ─────────────────────────────────────────────
RUN apt update && apt install -y --no-install-recommends \
    python3.10 python3-pip curl git locales \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 Humble
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | \
    apt-key add - \
    && echo "deb http://packages.ros.org/ros2/ubuntu jammy main" > \
    /etc/apt/sources.list.d/ros2.list \
    && apt update && apt install -y --no-install-recommends \
    ros-humble-ros-base \
    ros-humble-vision-msgs \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# ── Python 依赖 ──────────────────────────────────────────
COPY pyproject.toml /rai/pyproject.toml
RUN pip3 install uv && cd /rai && uv sync --no-dev

# ── RAI 源码 + PC Agent ─────────────────────────────────
COPY src/      /rai/src/
COPY examples/pc_agent/ /rai/examples/pc_agent/
COPY config.toml /rai/config.toml
COPY setup_shell.sh /rai/setup_shell.sh

# ── 构建 ─────────────────────────────────────────────────
WORKDIR /rai
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install"

# ── 启动 ─────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8501
ENTRYPOINT ["/entrypoint.sh"]
