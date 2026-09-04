FROM ubuntu:22.04

# ── 系统基础 ─────────────────────────────────────────────
RUN apt update && apt install -y --no-install-recommends \
    python3 python3-pip curl git \
    software-properties-common gnupg \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 Humble
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | \
    gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" > \
    /etc/apt/sources.list.d/ros2.list \
    && apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    ros-humble-ros-base \
    ros-humble-vision-msgs \
    ros-humble-nav2-msgs \
    python3-colcon-common-extensions \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

# ── 克隆 RAI ─────────────────────────────────────────────
RUN git clone --depth=1 https://github.com/RobotecAI/rai.git /rai

# ── Python 依赖 ──────────────────────────────────────────
RUN pip3 install uv
WORKDIR /rai
RUN uv sync --no-dev

# ── ros_deps ─────────────────────────────────────────────
RUN cd /rai && vcs import src < ros_deps.repos || true

# ── PC Agent ─────────────────────────────────────────────
COPY examples/pc_agent/ /rai/examples/pc_agent/
COPY config.toml /rai/config.toml

# ── Streamlit ────────────────────────────────────────────
RUN pip3 install streamlit

# ── 构建 ─────────────────────────────────────────────────
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install"

# ── 启动 ─────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8501
ENTRYPOINT ["/entrypoint.sh"]
