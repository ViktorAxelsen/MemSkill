#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


python main.py \
    --memory-cache-suffix "hp_eval" \
    --disable-flash-attn \
    --eval-only \
    --action-top-k 7 \
    --mem-top-k-eval 20 \
    --session-mode fixed-length \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --load-checkpoint '[YOUR_CHECKPOINT_PATH]' \
    --dataset hotpotqa \
    --hotpotqa-eval-file "[YOUR_EVAL_FILE_PATH]" \
    --data-file "[YOUR_DATA_PATH]" \
    --model "[YOUR_MODEL_NAME]" \
    --api \
    --api-base "[YOUR_API_BASE]" \
    --api-key "YOUR_API_KEY_1" "YOUR_API_KEY_2" \
    --retriever contriever \
    --designer-freq 1 \
    --inner-epochs 100 \
    --outer-epochs 10 \
    --batch-size 32 \
    --encode-batch-size 4 \
    --ppo-epochs 2 \
    --new-action-bias-steps 25 \
    --stage-reward-fraction 0.25 \
    --designer-reflection-cycles 3 \
    --mem-top-k 20 \
    --designer-max-changes 2 \
    --designer-failure-window-epochs 100 \
    --designer-failure-pool-size 2000 \
    --reward-metric llm_judge \
    --designer-new-skill-hint \
    --device cuda \
    --enable-designer \
    --skip-load-snapshot-manager \
    --wandb-run-name eval \
    --save-dir ./checkpoints/hp_with_designer \
    --out-file ./results/hp_with_designer.json

