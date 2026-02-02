#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


python main.py \
    --dataset locomo \
    --data-file "[YOUR_DATA_PATH]" \
    --model "[YOUR_MODEL_NAME]" \
    --api \
    --api-base "[YOUR_API_BASE]" \
    --api-key "YOUR_API_KEY_1" "YOUR_API_KEY_2" \
    --retriever contriever \
    --designer-freq 1 \
    --inner-epochs 100 \
    --outer-epochs 10 \
    --batch-size 4 \
    --encode-batch-size 64 \
    --session-mode full-session \
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
    --wandb-run-name locomo-train \
    --save-dir ./checkpoints/locomo_with_designer \
    --out-file ./results/locomo_with_designer.json
