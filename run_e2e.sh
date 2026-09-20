#!/bin/bash
export PYTHONPATH=/data/ajhou/repos/vllm-tapid/tapid/python:$PYTHONPATH
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_LOGGING_LEVEL=INFO
cd /data/ajhou/repos/vllm-tapid/vllm
exec stdbuf -o0 -e0 timeout -s KILL 2400 ./.venv/bin/python -u /data/ajhou/repos/vllm-tapid/run_tapid_vllm.py "$@"
