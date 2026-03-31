<div align="center">
  <img src="assets/logo.png" alt="LLMRouter Logo" width="300">
</div>



<h1 align="center">MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents</h1>




<div align="center">
  <p>
    <a href='https://viktoraxelsen.github.io/MemSkill/'><img src='https://img.shields.io/badge/Project-Page-00d9ff?style=for-the-badge&logo=github&logoColor=white'></a>
    <a href='https://arxiv.org/abs/2602.02474'><img src='https://img.shields.io/badge/arXiv-2602.02474-ff6b6b?style=for-the-badge&logo=arxiv&logoColor=white'></a>
    <a href="https://huggingface.co/papers/2602.02474"><img src="https://img.shields.io/badge/HuggingFace-Paper-FFD21E?style=for-the-badge&logo=huggingface&logoColor=FFD21E" alt="HuggingFace Paper"></a>
    <a href="https://huggingface.co/collections/XaiverZ/memskill"><img src="https://img.shields.io/badge/HuggingFace-Collection-FFD21E?style=for-the-badge&logo=huggingface&logoColor=FFD21E" alt="HuggingFace Collection"></a>
    <br>
    <a href="https://github.com/ViktorAxelsen/MemSkill/stargazers"><img src='https://img.shields.io/github/stars/ViktorAxelsen/MemSkill?color=f1e05a&style=for-the-badge&logo=star&logoColor=white' /></a>
    <a href="https://github.com/ViktorAxelsen/MemSkill/forks"><img src='https://img.shields.io/github/forks/ViktorAxelsen/MemSkill?color=2ea44f&style=for-the-badge&logo=git&logoColor=white' /></a>
    <a href="https://github.com/ViktorAxelsen/MemSkill/issues"><img src='https://img.shields.io/github/issues/ViktorAxelsen/MemSkill?color=d73a49&style=for-the-badge&logo=github&logoColor=white' /></a>
    <a href="https://deepwiki.com/ViktorAxelsen/MemSkill"><img src="https://img.shields.io/badge/DeepWiki-MemSkill-6B4FBB?style=for-the-badge&logo=readthedocs&logoColor=white" alt="DeepWiki"></a>
    <!-- <a href="https://www.python.org/downloads/release/python-3109/"><img src="https://img.shields.io/badge/PYTHON-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a> -->
    <!-- <a href="x" style="text-decoration:none;"><img src="https://img.shields.io/badge/TWITTER-ANNOUNCEMENTS-1DA1F2?style=for-the-badge&logo=x&logoColor=white" alt="Twitter"></a> -->
    <a href="LICENSE"><img src="https://img.shields.io/badge/LICENSE-Apache-2EA44F?style=for-the-badge" alt="License"></a>
  </p>
</div>




## 🧩 Overview

**MemSkill** is a framework for learning and evolving **memory skills** for long-horizon agents. It replaces static, hand-designed memory operations with a data-driven loop where skills are **learned, refined, and reused** from task feedback, enabling more adaptive memory construction across settings.

❗The skills evolved by **MemSkill** are **NOT** experiential/procedural memory/insights themselves. Rather, they are a form of **meta-memory** that focuses on **what kinds of memory to extract, how to remember and where to focus, and what to preserve or forget**. This is also why we call them **memory skills**: they capture the way or skill of remembering, rather than the remembered content itself.

**Highlights**

- **Skill-conditioned memory construction**: Compose a small set of relevant skills for each span and construct memories in one pass.

- **Skill evolution from hard cases**: Periodically mine challenging examples to refine existing skills and propose new ones.

- **Reusable skill bank**: Maintain a shared, evolving skill bank that supports transfer across datasets and base models.

- **High-throughput evaluation**: Multi-API-key round-robin for stable, parallel calls.

- **Scalable training and runs**: Multi-threading and multi-processing for large-scale training and evaluation.






<div align="center">
  <img src="./assets/model.png" width="800" alt="MemSkill">
</div>



## 📰 News

- 🚀 **[2026-03]**: `--locomo-train-query-sampling-ratio` is now available for training-time stratified test-query sampling on LoCoMo. It significantly reduces evaluation cost during training by sampling LoCoMo test queries by category, while leaving the full-evaluation protocol unchanged for `eval-only` and formal testing. For more details, please refer to [Commonly Used Configs](#️-commonly-used-configs).

- 🛠️ **[2026-03]**: We have added support for interrupted training recovery. You can now resume training in the `train_*.sh` scripts by passing `--load-checkpoint`, which restores key training state such as the controller/optimizer, operation bank, designer state (for example, the rolling failure-case pool), and other resume-critical metadata. At the moment, recovery is supported only from checkpoints saved at **outer-epoch boundaries**. By default, resumed runs continue logging to the original W&B run; if you prefer a fresh run for logging, use `--resume-new-wandb-run` instead. For more details, please refer to [Commonly Used Configs](#️-commonly-used-configs).

- 🚀 **[2026-03]**: We have improved the parallel memory extraction pipeline for evaluation and cache building, making MemSkill noticeably faster in large-scale runs. We also added clearer controls for concurrency with `--inference-workers` at the sample level and `--inference-session-workers` within each sample at the chunk/span level, which together can significantly accelerate memory extraction. For more details, please refer to [Commonly Used Configs](#️-commonly-used-configs).

- ⭐ **[2026-03]**: We have released the MemSkill controller weights in our [Hugging Face collection](https://huggingface.co/collections/XaiverZ/memskill), which can now be used directly for inference on suitable datasets. Please note that differences in experimental environments and settings may require some adaptation; when necessary, we recommend retraining and tuning key hyperparameters on a held-out validation split, especially `chunk_size` and the number of selected skills during inference (`action_top_k`), to ensure reliable performance. We hope these resources help advance self-evolving agent memory systems, and we'd be glad to hear from the community.

- 🔥 **[2026-02]**: We are honored to be featured in the 🤗 HuggingFace [#3 Paper of the day](https://huggingface.co/papers/2602.02474)

- 🚀 **[2026-02]**: **MemSkill** is officially released — a new paradigm for agent memory that learns reusable skills 🔁 and evolves them from data over time 🧠, improving memory quality and generalization across long, open-ended interactions ✨.





## 🔗 Links

- [Overview](#-overview)
- [News](#-news)
- [Get Started](#-get-started)
- [Installation](#installation)
- [Preparing Training Data](#-preparing-training-data)
- [Experiments](#-experiments)
- [Extending to New Datasets and Evaluation Protocol](#-extending-to-new-datasets-and-evaluation-protocol)
- [Commonly Used Configs](#️-commonly-used-configs)
- [Citation](#-citation)






## 🚀 Get Started

### Installation

```bash
# Clone the repository
git clone https://github.com/ViktorAxelsen/MemSkill
cd MemSkill

# Create and activate virtual environment
conda create -n memskill python=3.10
conda activate memskill

# vllm
pip install vllm==0.6.3
# PyTorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# Flash-Attn (or you can specify `--disable-flash-attn` in the .sh scripts to disable it)
pip install flash-attn --no-build-isolation
# Others
pip install -r requirements.txt
```




### 📊 Preparing Training Data

MemSkill builds training and evaluation data from the datasets below. Please download data from the official sources and place them under `data/`. Unless otherwise noted, splits are already configured in our codebase.

#### **1) LoCoMo**
- Download LoCoMo from the official repo: [LoCoMo](https://github.com/snap-research/locomo)  
- **Splits**: LoCoMo splits are **already configured in `main.py`** (no extra split file needed).  
- Put the downloaded files under:
  - `data/locomo10.json`

#### **2) LongMemEval**
- We use **LongMemEval-S** from: [LongMemEval](https://github.com/xiaowu0162/LongMemEval)  
- **Important**: LongMemEval-S is used for **transfer evaluation only**. That is, skills trained on LoCoMo are **directly evaluated** on LongMemEval-S without additional training.
- Put the downloaded files under:
  - `data/longmemeval_s_cleaned.json`
- Use our split file:
  - `data/longmemeval_s_splits.json` (**We use test split only**)



#### **3) HotpotQA**
- Download HotpotQA from: [HotpotQA-Modified](https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/tree/main) (Source: [HotpotQA](https://hotpotqa.github.io/))
- We evaluate on three test files:
  - `data/eval_50.json`
  - `data/eval_100.json`
  - `data/eval_200.json`

These correspond to **increasing context length**, where each query context is constructed by concatenating **50 / 100 / 200 documents** (following the long-context evaluation protocol we adopt in our experiments).



#### **4) ALFWorld**
Please follow the official instructions to install dependencies and download assets: [ALFWorld](https://github.com/alfworld/alfworld)

We use **offline expert trajectories** as the interaction corpus for memory construction. We provide a one-command script to collect and save trajectories:

```bash
# Collect expert trajectories for train / seen / unseen splits
python alfworld_replay.py --split train --output ./data/alfworld_train_offline.json
python alfworld_replay.py --split eval_in_distribution --output ./data/alfworld_expert_eval_in_distribution.json
python alfworld_replay.py --split eval_out_of_distribution --output ./data/alfworld_expert_eval_out_of_distribution.json
```

Note that:

- We collect `seen` and `unseen` expert plans **only to keep data formats consistent** and make evaluation easier. They are not used for training.

- The saved trajectories will be saved under `data/` by default.


> **ALFWorld Training Data Preparation Workflow**

We separate data into **two batches with different roles**.

**Batch A: Offline expert trajectories (memory construction batch)**  
We first collect expert rollouts (the JSON files above). During training, we sample a batch of trajectories and:
- split each trajectory trace into **contiguous spans** (processed sequentially span by span)
- build an **episode specific memory bank** by running MemSkill on these spans
- record the controller’s Top-K skill selections (and the associated policy info needed for RL updates)

This batch is used to teach the controller how to **compose skills** for memory construction from realistic interaction traces, without requiring environment interaction.

**Batch B: Environment evaluation episodes (reward batch)**  
To obtain task-level feedback, we sample a batch of ALFWorld tasks and:
- run the agent in the environment using the memory bank produced by MemSkill in the Batch A
- compute the **task signal** (e.g., success rate) as the reward feedback
- log difficult failures as hard cases for the designer

This batch provides the supervision signal that tells the controller whether its skill composition actually helps long-horizon execution.

**How they work together**
- Batch A provides the behavior data (skill choices on spans) needed to optimize the controller policy.
- Batch B provides the downstream task feedback that makes the optimization meaningful.
- The designer then mines hard cases from Batch B outcomes to refine existing skills and propose new ones.

In short, ALFWorld uses **offline traces for scalable memory construction training (Batch A)** and **environment rollouts for task feedback and skill evolution (Batch B)**.








❗For integrating more datasets, our framework is designed to be flexible and easy to extend to new settings (different interaction formats, query styles, and evaluation protocols). See [Extending to New Datasets and Evaluation Protocol](#-extending-to-new-datasets-and-evaluation-protocol) for step-by-step instructions.





## 🧪 Experiments

Before running, please check the parameter configuration in the `.sh` scripts.

> [!IMPORTANT]
>
> **Before running, please review **[Commonly Used Configs](#️-commonly-used-configs)** and update the `.sh` scripts with your dataset paths, API settings, and model choices.**


### 🖥️ Training

Choose the training script based on the dataset you want to use. Make sure `--data-file`, `--model`, and API settings are set correctly.

```bash
bash train_locomo.sh
# or
bash train_alfworld.sh
```


### 🧭 Evaluation

Use the matching evaluation script after training. Set --load-checkpoint and dataset paths before running.

```bash
bash eval_locomo.sh
# or
bash eval_alfworld.sh
# or
bash eval_hp.sh
# or
bash eval_longmemeval.sh
```



## 🔧 Extending to New Datasets and Evaluation Protocol

MemSkill is designed to be extensible across diverse datasets and evaluation protocols. To add a new dataset or evaluation protocol:

- Implement a data processor in `src/data_processing/` (parse raw data, build sessions/spans, and expose QA items if applicable).
- Implement an evaluator in `src/eval/` (prompt construction, answer extraction, metric computation).
- Register the new processor/evaluator in `src/data_processing/__init__.py` and `src/eval/__init__.py`.
- Add a run script (train/eval) with the dataset name, data paths, and retrieval settings.

In practice, you only need to define how to segment interaction history into spans, how to format prompts for memory construction and QA, and how to score outputs. The controller–executor–designer loop and skill‑bank evolution remain unchanged.

Note: `src/trainer.py` usually does not need changes when adding a new dataset; only interactive environments (e.g., ALFWorld‑style) or custom training loops require trainer modifications.



## ⚙️ Commonly Used Configs

These are the parameters most frequently used in the training/eval `.sh` scripts:

**Core run settings**
- `--dataset`: dataset name (`locomo`, `longmemeval`, `hotpotqa`, `alfworld`)
- `--data-file`: path to the main dataset file
- `--model`: base LLM name
- `--designer-model`: base LLM name for designer (default: the same as --model)
- `--api`: use API-based inference (without this flag, the model is loaded locally with vLLM for inference)
- `--api-base`: API endpoint
- `--api-key`: one or more API keys
- `--disable-flash-attn`: Flash Attention is enabled by default; add this flag to disable it.

> [!NOTE]
>
> To reduce the training cost introduced by open-ended skill evolution, one practical option is to use a smaller `--model` together with a larger `--designer-model`. In our experiments, we can still observe an increasing reward curve even when both `--model` and `--designer-model` are small. Generally speaking, however, a larger `--designer-model` makes skill evolution more effective while adding almost no extra training cost, because the designer contributes only a negligible number of LLM calls and token comsumption compared to the controller.
>
> Another option is to reduce the number of retrieved memories during training, namely `--mem-top-k`, since the dominant training cost usually comes from evaluating test queries. We also provide `--locomo-train-query-sampling-ratio`, a query-sampling option that can stratify test queries to reduce the evaluation burden during training while preserving stability and effectiveness. We hope these practical insights and experiences help the community follow, reproduce, and adapt MemSkill more easily.


**Retrieval & memory**
- `--retriever`: retriever type (`contriever`, `dpr`, `dragon`)
- `--mem-top-k`: top‑K memories for training
- `--mem-top-k-eval`: top‑K memories for evaluation
- `--locomo-train-query-sampling-ratio`: LoCoMo-only training-time query sampling ratio for stratified test-query evaluation (`1.0` keeps full evaluation)
- `--session-mode`: span granularity (`turn`, `turn-pair`, `full-session`, `fixed-length`)
- `--chunk-size`, `--chunk-overlap`: fixed-length chunking size/overlap
- `--inference-workers`: number of parallel workers for sample-level memory inference(`1` = serial)
- `--inference-session-workers`: number of parallel workers for session/span-level memory inference within each sample (`1` = serial; **values > 1 may degrade performance**)

> [!WARNING]
>
> `--inference-workers` increases sample-level concurrency and can usually be set higher as long as your API rate limit can sustain it, without affecting model performance. `--inference-session-workers` increases within-sample concurrency for sequential memory extraction and can speed up runs, but values greater than `1` may hurt task performance. Together with `--chunk-size`, these knobs can significantly accelerate memory extraction, so tune them carefully against quality.

**Training**
- `--batch-size`: episodes per PPO update
- `--encode-batch-size`: batch size for embedding encoder
- `--inner-epochs`, `--outer-epochs`: training schedule
- `--ppo-epochs`: PPO update epochs per batch
- `--action-top-k`: number of skills selected per step
- `--reward-metric`: `f1` or `llm_judge`
- `--llm-judge-model`: model used for LLM-judge scoring during training and text-dataset evaluation

**Designer / evolution**
- `--enable-designer`: enable skill evolution
- `--designer-freq`: evolve every N outer epochs
- `--designer-max-changes`: max changes per evolution cycle
- `--designer-new-skill-hint`: encourage adding new skills

**Eval / checkpoints**
- `--eval-only`: run evaluation only
- `--load-checkpoint`: checkpoint path (Training checkpoints are saved at outer-epoch boundaries; resuming with `--load-checkpoint` continues from the next outer epoch instead of restarting from epoch 0)
- `--resume-new-wandb-run`: resume training state from a checkpoint but log to a fresh W&B run instead of the saved run ID.
- `--skip-load-operation-bank` / `--skip-load-snapshot-manager`: partial-load options for special cases; do not use them for full resume training.
- `--save-dir`: where to save checkpoints
- `--out-file`: output result file
- `--overwrite`: force rebuilding memory caches under the run's `memories/` directory instead of reusing existing ones
- `--device`: `cuda` or `cpu`

**ALFWorld‑specific**
- `--alfworld-offline-data`: offline trajectories
- `--alfworld-eval-file`: eval split
- `--alfworld-eval-query-source`: `objective` or `first_observation`


**W&B (logging)**
- `--wandb-project`: W&B project name.
- `--wandb-run-name`: W&B run name.
- `--wandb-key`: W&B API key (optional; you can also set `WANDB_API_KEY` in the environment).



❗For the full list of configuration options, see `src/config.py`.




## 🙏 Acknowledgments


We thank the authors and maintainers of **[LoCoMo](https://github.com/snap-research/locomo)**, **[LongMemEval](https://github.com/xiaowu0162/LongMemEval)**, **[HotpotQA-Modified](https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/tree/main) (Source: [HotpotQA](https://hotpotqa.github.io/))**, and **[ALFWorld](https://github.com/alfworld/alfworld)** for releasing their datasets, evaluation protocols, and supporting code. Their efforts in building and open-sourcing high-quality benchmarks make it possible to develop, evaluate, and reproduce research on agent memory. We also appreciate the broader open-source LLM/Agent memory community for tooling that supports reliable evaluation and large-scale experimentation.






## 📚 Citation

```bibtex
@article{zhang2026memskill,
  title={MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents},
  author={Zhang, Haozhen and Long, Quanyu and Bao, Jianzhu and Feng, Tao and Zhang, Weizhi and Yue, Haodong and Wang, Wenya},
  journal={arXiv preprint arXiv:2602.02474},
  year={2026}
}
```

