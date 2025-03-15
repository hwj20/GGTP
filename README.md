# Graphormer-Guided Task Planning (GGTP)

🚀 **Graphormer-Guided Task Planning** is an AI-driven task planning framework that combines LLM-based decision-making with Graphormer-enhanced risk assessment.
The overview video is avaliable as:
https://www.bilibili.com/video/BV1HDQ7YWEYN

## 📦 Motivation

<video width="600" controls>
  <source src="./docs/video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


Robots may be great at moving around, but can they *think* before they act?  
We introduce Graphormer-based task planning with LLM-driven safety perception,  
helping domestic robots navigate complex environments without, well… accidentally microwaving the cat.
We aim to enhance task planning in domestic robots by utilzing Graphormer-based and LLM-driven safety perception to dynamically adapt to complex environments.



## 🛠 Installation
Set 'OPENAI_API_KEY' as an environment variable with your own key.

Clone the repo and install dependencies:
```bash
git clone https://github.com/hwj20/GGTP.git
cd GGTP
conda env create -f environment.yml
conda activate GGTP
python main.py
```
This will run a sample task planning on ai2-thor floor1.

## 📺 Advertisement Break!
💡 **Tired of fixing dependencies?** Try our VSCode extension [Auto Config Env](https://marketplace.visualstudio.com/items?itemName=WanjingHuang.auto-config-env) to automatically install missing packages and debug your Python environment like a pro! 🚀


## 📌 Reproduce Expeirments
See as 'experiments/README_EXP.md'

## 📖 Citation
If you find our work helpful, please consider citing our paper:  

```bibtex
@misc{huang2025graphormerguidedtaskplanningstatic,
      title={Graphormer-Guided Task Planning: Beyond Static Rules with LLM Safety Perception}, 
      author={Wanjing Huang and Tongjie Pan and Yalan Ye},
      year={2025},
      eprint={2503.06866},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2503.06866}, 
}
```

## 📦 Dataset & Code Availability
Our full dataset and experimental results are **available in this repository** for reproducibility.  
However, due to the **large file size**, please ensure you have **sufficient storage and bandwidth** before cloning (300 MB).  

💡 **For a quick overview**, check out our [Paper on Arxiv](https://arxiv.org/abs/2503.06866).