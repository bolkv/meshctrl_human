#!/bin/bash
cd /workspace/shapectrl/data
NGPU=${NGPU:-4}
export PYTHONPATH=/workspace/shapectrl/data/.qwen_libs

echo "captioner=qwen ngpu=$NGPU root=${SINGLE_ROOT:-bedlam_single}"
for i in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$i NSHARD=$NGPU SHARD=$i \
    python3 -u caption_single_qwen.py > caption_single_shard${i}.log 2>&1 &
  echo "  launched shard $i on GPU $i (PID $!)"
done
echo "progress: tail -f caption_single_shard0.log"
wait
echo "ALL SHARDS DONE"
