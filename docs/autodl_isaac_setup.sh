#!/usr/bin/env bash
# Isaac Sim 5.1.0 on an AutoDL container (Ubuntu 22.04, no Docker). Learned on 2026-11-25.
#
#   scp -P <port> autodl_isaac_setup.sh ../validation/isaac_smoke.py root@<host>:/root/autodl-tmp/isaac/
#   ssh -p <port> root@<host> 'cd /root/autodl-tmp/isaac && setsid nohup bash autodl_isaac_setup.sh > setup.log 2>&1 < /dev/null &'
#
# Hard requirement checked first: NVIDIA driver 580.x. On 595.71.05 (RTX PRO 6000
# Blackwell Server) Isaac Sim 5.1.0 segfaults in librtx.scenedb.plugin at RTX startup,
# reproduced twice, with and without the X/GL libraries below.
set -euxo pipefail
cd /root/autodl-tmp/isaac

driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
case "$driver" in
  580.*) echo "driver $driver ok" ;;
  *) echo "STOP: driver $driver is not 580.x (595.x crashes in rtx.scenedb)"; exit 2 ;;
esac

# The container ships the NVIDIA ICD but none of the X/GL libs it and Isaac Sim load.
# Without libx11/libxext the Vulkan loader only sees llvmpipe; without libglu/libxt
# MaterialX and Iray fail to load.
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq vulkan-tools libvulkan1 \
  libx11-6 libxext6 libegl1 libgl1 libglvnd0 libglu1-mesa libxt6 libxrandr2 \
  libxinerama1 libxcursor1 libxi6 libsm6 libice6 >/dev/null
vulkaninfo --summary 2>/dev/null | grep -q "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU" \
  || { echo "STOP: Vulkan does not see the GPU"; exit 3; }

export UV_CACHE_DIR=/root/autodl-tmp/uv-cache UV_PYTHON_INSTALL_DIR=/root/autodl-tmp/uv-python
export PATH=$HOME/.local/bin:$PATH
# The AutoDL academic proxy helps GitHub only; it makes pip sources slower. Use it for uv itself.
command -v uv >/dev/null || (source /etc/network_turbo; curl -LsSf https://astral.sh/uv/install.sh | sh)
(source /etc/network_turbo; uv python install 3.11)
[ -d venv ] || uv venv --python 3.11 venv
. venv/bin/activate

# NVIDIA wheels direct (fast without proxy), everything else from the Aliyun mirror.
export UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=8
uv pip install "isaacsim[all,extscache]==5.1.0" \
  --index-url https://mirrors.aliyun.com/pypi/simple \
  --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match

# Importing isaacsim prompts for the EULA; a background job must accept it via env.
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
timeout 1500 python isaac_smoke.py > smoke.log 2>&1 && echo SMOKE_PASS || echo "SMOKE_FAIL $?"
grep SMOKE_RESULT smoke.log || true
