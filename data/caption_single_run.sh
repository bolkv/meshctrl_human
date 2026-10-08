#!/bin/bash
# bedlam_single 크롭 캡션 - N GPU 병렬 (GPU당 샤드 1개), Qwen2.5-VL
# 사용: bash caption_single_run.sh    (추출이 다 끝난 뒤 실행)
cd /workspace/shapectrl/data
NGPU=${NGPU:-4}
export PYTHONPATH=/workspace/shapectrl/data/.qwen_libs   # transformers 4.49 shadow (학습 env 무손상)

echo "captioner=qwen ngpu=$NGPU root=${SINGLE_ROOT:-bedlam_single}"
for i in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$i NSHARD=$NGPU SHARD=$i \
    python3 -u caption_single_qwen.py > caption_single_shard${i}.log 2>&1 &
  echo "  launched shard $i on GPU $i (PID $!)"
done
echo "진행: tail -f caption_single_shard0.log"
wait
echo "ALL SHARDS DONE"
