# Graphormer-Guided Task Planning (GGTP)

🚀 **Graphormer-Guided Task Planning** is an AI-driven task planning framework that combines LLM-based decision-making with Graphormer-enhanced risk assessment.

## 📦 Motivation
<video width="600" controls>
  <source src="./docs/video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
Our video is avaliable as:
https://www.bilibili.com/video/BV1HDQ7YWEYN

Robots may be great at moving around, but can they *think* before they act?  
We introduce Graphormer-based task planning with LLM-driven safety perception,  
helping domestic robots navigate complex environments without, well… accidentally microwaving the cat.
We aim to enhance task planning in domestic robots by utilzing Graphormer-based and LLM-driven safety perception to dynamically adapt to complex environments.



## 🛠 Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/hwj20/GGTP.git
cd GGTP
conda env create -f environment.yml
conda activate GGTP
python main.py
```
This will run a sample task planning on ai2-thor floor1.

## Reproduce Expeirments
See as 'experiments/README_EXP.md'

