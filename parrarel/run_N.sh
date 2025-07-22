#!/usr/bin/env bash
# run_N.sh
# run_1.sh 을 10개 병렬 실행하며, 출력을 인스턴스별로 분리

N=2
FOREVER="./run_1.sh"

for (( i=0; i<N; i++ )); do
  # 각 인스턴스의 stdout/stderr를 별도 파일로 리다이렉트
  "$FOREVER" "$i"\
    > "./logs/${i}.out.log" \
    2> "./logs/${i}.err.log" &
  echo "Launched instance $i (PID=$!)"
done

# 모든 백그라운드 작업이 올라간 후엔 wait 로 대기
wait
echo "All $N instances of run_1.sh have been launched."