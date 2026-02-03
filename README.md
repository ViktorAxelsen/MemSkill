<div align="center">
  <img src="assets/logo.png" alt="LLMRouter Logo" width="300">
</div>



<h1 align="center">MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents</h1>




<div align="center">
  <p>
    <a href='https://viktoraxelsen.github.io/MemSkill/'><img src='https://img.shields.io/badge/Project-Page-00d9ff?style=for-the-badge&logo=github&logoColor=white'></a>
    <a href='https://arxiv.org/abs/2602.02474'><img src='https://img.shields.io/badge/arXiv-2602.02474-ff6b6b?style=for-the-badge&logo=arxiv&logoColor=white'></a>
    <br>
    <a href="https://github.com/ViktorAxelsen/MemSkill/stargazers"><img src='https://img.shields.io/github/stars/ViktorAxelsen/MemSkill?color=f1e05a&style=for-the-badge&logo=star&logoColor=white' /></a>
    <a href="https://github.com/ViktorAxelsen/MemSkill/forks"><img src='https://img.shields.io/github/forks/ViktorAxelsen/MemSkill?color=2ea44f&style=for-the-badge&logo=git&logoColor=white' /></a>
    <a href="https://github.com/ViktorAxelsen/MemSkill/issues"><img src='https://img.shields.io/github/issues/ViktorAxelsen/MemSkill?color=d73a49&style=for-the-badge&logo=github&logoColor=white' /></a>
    <a href="https://www.python.org/downloads/release/python-3109/"><img src="https://img.shields.io/badge/PYTHON-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
    <!-- <a href="x" style="text-decoration:none;"><img src="https://img.shields.io/badge/TWITTER-ANNOUNCEMENTS-1DA1F2?style=for-the-badge&logo=x&logoColor=white" alt="Twitter"></a> -->
    <a href="LICENSE"><img src="https://img.shields.io/badge/LICENSE-Apache-2EA44F?style=for-the-badge" alt="License"></a>
  </p>
</div>




## 🧩 Overview

**MemSkill** is a framework for learning and evolving **memory skills** for long-horizon agents. It replaces static, hand-designed memory operations with a data-driven loop where skills are **learned, refined, and reused** from task feedback, enabling more adaptive memory construction across settings.

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


- 🚀 **[2026-02]**: **MemSkill** is officially released — a new paradigm for agent memory that learns reusable skills 🔁 and evolves them from data over time 🧠, improving memory quality and generalization across long, open-ended interactions ✨. **Stay tuned! More detailed instruction updates coming soon.**





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

# pip install
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
- `--api`: use API-based inference
- `--api-base`: API endpoint
- `--api-key`: one or more API keys

**Retrieval & memory**
- `--retriever`: retriever type (`contriever`, `dpr`, `dragon`)
- `--mem-top-k`: top‑K memories for training
- `--mem-top-k-eval`: top‑K memories for evaluation
- `--session-mode`: span granularity (`turn`, `turn-pair`, `full-session`, `fixed-length`)
- `--chunk-size`, `--chunk-overlap`: fixed-length chunking size/overlap

**Training**
- `--batch-size`: episodes per PPO update
- `--encode-batch-size`: batch size for embedding encoder
- `--inner-epochs`, `--outer-epochs`: training schedule
- `--ppo-epochs`: PPO update epochs per batch
- `--action-top-k`: number of skills selected per step
- `--reward-metric`: `f1` or `llm_judge`

**Designer / evolution**
- `--enable-designer`: enable skill evolution
- `--designer-freq`: evolve every N outer epochs
- `--designer-max-changes`: max changes per evolution cycle
- `--designer-new-skill-hint`: encourage adding new skills

**Eval / checkpoints**
- `--eval-only`: run evaluation only
- `--load-checkpoint`: checkpoint path
- `--save-dir`: where to save checkpoints
- `--out-file`: output result file
- `--device`: `cuda` or `cpu`

**ALFWorld‑specific**
- `--alfworld-offline-data`: offline trajectories
- `--alfworld-eval-file`: eval split
- `--alfworld-eval-query-source`: `objective` or `first_observation`




## 🙏 Acknowledgments


We thank the authors and maintainers of **[LoCoMo](https://github.com/snap-research/locomo)**, **[LongMemEval](https://github.com/xiaowu0162/LongMemEval)**, **[HotpotQA-Modified](https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/tree/main) (Source: [HotpotQA](https://hotpotqa.github.io/))**, and **[ALFWorld](https://github.com/alfworld/alfworld)** for releasing their datasets, evaluation protocols, and supporting code. Their efforts in building and open-sourcing high-quality benchmarks make it possible to develop, evaluate, and reproduce research on agent memory. We also appreciate the broader open-source LLM/Agent memory community for tooling that supports reliable evaluation and large-scale experimentation.






## 📚 Citation

```bibtex
@article{MemSkill,
  title={MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents},
  author={Haozhen Zhang and Quanyu Long and Jianzhu Bao and Tao Feng and Weizhi Zhang and Haodong Yue and Wenya Wang},
  journal={arXiv preprint arXiv:2602.02474},
  year={2026}
}
```
