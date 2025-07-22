#!/usr/bin/env bash
# run_1.sh
# script.py 가 종료되면 자동으로 재실행 → 끌 때까지 계속 반복
INDEX=$1
PYTHON="/home/harry/PycharmProjects/PeetsFEATest/.venv/bin/python"
SCRIPT="/home/harry/PycharmProjects/PeetsFEATest/script.py"

while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting script.py…"
  "$PYTHON" "$SCRIPT" "$INDEX"
  sleep 10
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] script.py exited (code=$?). Restarting…"
done