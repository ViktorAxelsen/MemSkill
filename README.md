<div align="center">
  <img src="assets/logo.png" alt="LLMRouter Logo" width="300">
</div>



<h1 align="center">MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents</h1>




<div align="center">
  <p>
    <a href='x'><img src='https://img.shields.io/badge/Project-Page-00d9ff?style=for-the-badge&logo=github&logoColor=white'></a>
    <a href='x'><img src='https://img.shields.io/badge/arXiv-xxxx.xxxxx-ff6b6b?style=for-the-badge&logo=arxiv&logoColor=white'></a>
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

**MemSkill** is a framework for evolving memory skills in long‑horizon, self‑improving agents. It brings a data‑driven feedback loop into the memory module so that memory skills can be learned, evolved, and reused—improving stability, generalization, and maintainability in long, open‑ended interactions.

**Framework Features**:
- Evolve your own memory skills by iteratively discovering better memory operations from data.
- Upgrade memory from static rules to an evolvable capability, strengthening long‑term reasoning and transfer.
- Modular, reusable design that adapts across datasets and task settings.
- Multi‑API‑key round‑robin to improve throughput and stability.
- Multi‑threading / multi‑processing acceleration for training and evaluation at scale.
- ......







<div align="center">
  <img src="./assets/model.png" width="800" alt="MemSkill">
</div>



## 📰 News


- 🚀 **[2026-02]**: **MemSkill** is officially released — a new paradigm for agent memory that learns reusable skills 🔁 and evolves them from data over time 🧠, improving memory quality and generalization across long, open-ended interactions ✨. **Stay tuned! More detailed instruction updates coming soon.**





<!-- ## 🔗 Links

- [TBD](#-TBD) -->







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

We build training data from the following datasets. Please follow the linked sources and keep the same splits where specified.

After downloading, place the data under the `data/` folder.

- [Locomo](https://github.com/snap-research/locomo)
- [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval): use the split file `data/longmemeval_s_splits.json`.
- [HotpotQA](https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/tree/main)
- [ALFWorld](https://github.com/alfworld/alfworld)




## 🧪 Experiments

Before running, please check the parameter configuration in the `.sh` scripts.

> [!IMPORTANT]
>
> **Make sure to set your own API base and API key in the `.sh` scripts before running.**

### 🖥️ Training

Run a training script depending on the dataset you want to use:

```bash
bash train_locomo.sh
# or
bash train_alfworld.sh
```

### 🧭 Evaluation

Run the evaluation script for the corresponding dataset:

```bash
bash eval_locomo.sh
bash eval_alfworld.sh
bash eval_hp.sh
bash eval_longmemeval.sh
```

**Stay tuned! More detailed instruction updates coming soon.**



## 📚 Citation

```bibtex
@article{MemSkill,
  title={MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents},
  author={Haozhen Zhang and Quanyu Long and Jianzhu Bao and Tao Feng and Weizhi Zhang and Haodong Yue and Wenya Wang},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
```
