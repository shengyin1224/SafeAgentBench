# SafeAgentBench: A Benchmark for Safe Task Planning of Embodied LLM Agents

With the integration of large language models (LLMs), embodied agents have strong capabilities to execute complicated instructions in natural language, paving a way for the potential deployment of embodied robots. However, a foreseeable issue is that those embodied agents can also flawlessly execute some hazardous tasks, potentially causing damages in real world. To study this issue, we present **SafeAgentBench** —- a new benchmark for safety-aware task planning of embodied LLM agents. SafeAgentBench includes: (1) a new dataset with 750 tasks, covering 10 potential hazards and 3 task types; (2) SafeAgentEnv, a universal embodied environment with a low-level controller, supporting multi-agent execution with 17 high-level actions for 8 state-of-the-art baselines; and (3) reliable evaluation methods from both execution and semantic perspectives. Experimental results show that the best-performing baseline gets 69% success rate for safe tasks, but only 5% rejection rate for hazardous tasks, indicating significant safety risks.

For the latest updates, see: [**our website**](https://safeagentbench.github.io)

![](figure/safeagentbench_show.jpg)

## Quickstart

Clone repo:

```bash
$ git clone https://github.com/shengyin/safeagentbench.git 
$ cd safeagentbench
```

Install requirements:

```bash
$ pip install -r requirements.txt
```

## More Info 

- [**Dataset**](dataset/): Safe detailed tasks(300 samples), unsafe detailed tasks(300 samples), abstract tasks(100 samples) and long-horizon tasks(50 samples).
- [**Evaluators**](evaluator/): Evaluation metrics for each type of task, including success rate, rejection rate, and other metrics.
- [**low-level controller**](low_level_controller/): A low-level controller for SafeAgentEnv, which takes in the high-level action and map them to low-level actions supported by AI2-THOR for the agent to execute. You can choose multi-agent version or single-agent version. 
- [**Methods**](methods/): Implementation of the proposed methods.

## SOTA Embodied LLM Agents

Because each agent has different code structure, we can not provide all the implementation codes. You can refer to these works' papers and codes to implement your own agent.  

<b> LoTa-Bench: Benchmarking Language-oriented Task Planners for Embodied Agents </b>
<br>
Jae-Woo Choi, Youngwoo Yoon, Hyobin Ong, Jaehong Kim, Minsu Jang
<br>
<a href="https://arxiv.org/abs/2402.08178"> Paper</a>, <a href="https://choi-jaewoo.github.io/LoTa-Bench/"> Code </a> 

<b> Building Cooperative Embodied Agents Modularly with Large Language Models </b>
<br>
Hongxin Zhang*, Weihua Du*, Jiaming Shan, Qinhong Zhou, Yilun Du, Joshua B. Tenenbaum, Tianmin Shu, Chuang Gan_: Building Cooperative Embodied Agents Modularly with Large Language Models
<br>
<a href="https://arxiv.org/abs/2307.02485"> Paper</a>, <a href="https://vis-www.cs.umass.edu/Co-LLM-Agents/"> Code </a> 

<b> ProgPrompt: Generating Situated Robot Task Plans using Large Language Models </b>
<br>
Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, Animesh Garg
<br>
<a href="https://arxiv.org/abs/2209.11302"> Paper</a>, <a href="https://github.com/NVlabs/progprompt-vh"> Code </a> 

<b> MLDT: Multi-Level Decomposition for Complex Long-Horizon Robotic Task Planning with Open-Source Large Language Model </b>
<br>
Yike Wu, Jiatao Zhang, Nan Hu, LanLing Tang, Guilin Qi, Jun Shao, Jie Ren, Wei Song
<br>
<a href="https://arxiv.org/abs/2403.18760.pdf"> Paper</a>, <a href="https://github.com/wuyike2000/MLDT"> Code </a>

<b> PCA-Bench: Evaluating Multimodal Large Language Models in Perception-Cognition-Action Chain </b>
<br>
Liang Chen, Yichi Zhang, Shuhuai Ren, Haozhe Zhao, Zefan Cai, Yuchi Wang, Peiyi Wang, Xiangdi Meng, Tianyu Liu, Baobao Chang
<br>
<a href="https://arxiv.org/abs/2402.15527.pdf"> Paper</a>, <a href="https://github.com/pkunlp-icler/PCA-EVAL"> Code </a>

<b> ReAct: Synergizing Reasoning and Acting in Language Models </b>
<br>
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao
<br>
<a href="https://arxiv.org/abs/2210.03629/"> Paper</a>, <a href="https://github.com/ysymyth/ReAct"> Code </a>

<b> LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models </b>
<br>
Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M. Sadler, Wei-Lun Chao, Yu Su,
<br>
<a href="https://arxiv.org/abs/2212.04088.pdf"> Paper</a>, <a href="https://github.com/OSU-NLP-Group/LLM-Planner/"> Code </a>

<b> Multi-agent Planning using Visual Language Models </b>
<br>
Michele Brienza, Francesco Argenziano, Vincenzo Suriani, Domenico D. Bloisi, Daniele Nardi
<br>
<a href="https://arxiv.org/abs/2408.05478"> Paper</a>, <a href="https://github.com/Lab-RoCoCo-Sapienza/map-vlm/"> Code </a>  

## Hardware 

The same as AI2-THOR.

## Citation

If you find the dataset or code useful, please cite:

```
TBD
```

## License

MIT License


## Contact

Questions or issues? Contact [yin.sheng011224@sjtu.edu.cn](yin.sheng011224@sjtu.edu.cn)