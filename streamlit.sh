#!/bin/bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
  libgl1 \
  libglib2.0-0 \
  libxcb1

exec python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0 --server.headless true
