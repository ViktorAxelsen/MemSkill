#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


python main.py \
    --disable-flash-attn \
    --alfworld-eval-query-source objective \
    --alfworld-pair-a-min 40 \
    --alfworld-pair-a-max 60 \
    --alfworld-pair-b-size 5 \
    --alfworld-pair-b-workers 80 \
    --alfworld-pair-same-type-prob 1.0 \
    --dataset alfworld \
    --alfworld-offline-data "[YOUR_OFFLINE_DATA_PATH]" \
    --model "[YOUR_MODEL_NAME]" \
    --api \
    --api-base "[YOUR_API_BASE]" \
    --api-key "YOUR_API_KEY_1" "YOUR_API_KEY_2" \
    --retriever contriever \
    --designer-freq 1 \
    --inner-epochs 100 \
    --outer-epochs 10 \
    --batch-size 16 \
    --encode-batch-size 8 \
    --session-mode fixed-length \
    --ppo-epochs 2 \
    --action-top-k 3 \
    --new-action-bias-steps 25 \
    --stage-reward-fraction 0.25 \
    --designer-reflection-cycles 3 \
    --mem-top-k 20 \
    --mem-top-k-eval 20 \
    --designer-max-changes 3 \
    --designer-failure-window-epochs 100 \
    --designer-failure-pool-size 2000 \
    --reward-metric llm_judge \
    --designer-new-skill-hint \
    --device cuda \
    --enable-designer \
    --skip-load-operation-bank \
    --skip-load-snapshot-manager \
    --wandb-run-name alf-train \
    --save-dir ./checkpoints/alf_with_designer \
    --out-file ./results/alf_with_designer.json
