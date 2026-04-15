## 说明

- **来源**：原论文 Markdown 中标题为「2 Method」的小节（对应标准章节：Methods (方法)）。
- **解析模板**：`prompts/paper/section-methods.md`
- **提示摘要**：你是学术论文解读助手。请针对下列 **Methods（方法）** 原文进行中文解析。

---

{
    "方法管线概览": {
        "输入": "任务指令、环境观测序列 $s_{t-H}, a_{t-H}, ..., s_t$ 以及当前动作 $a_t$",
        "处理": "1. LLM策略 $\\pi_\\theta$ 基于历史交互和当前动作生成推理标记和下一状态预测 $(\\text{reason}, \\hat{s}_{t+1})$。\n2. 使用预训练嵌入模型 $E(\\cdot)$ 计算预测状态 $\\hat{s}_{t+1}$ 与真实状态 $s_{t+1}$ 的余弦相似度距离 $d(\\hat{s}_{t+1}, s_{t+1})$。\n3. 基于距离阈值 $\\tau_d$ 生成二元奖励 $r^{\\text{WM}}(\\hat{s}_{t+1}, s_{t+1})$。\n4. 使用GRPO（Group Relative Policy Optimization）优化策略，目标函数为 $\\mathbb{E}_{\\pi_{\\theta_{\\text{old}}}}\\left[\\min{\\left(\\rho_\\theta A,\\text{clip}(\\rho_\\theta,1\\pm\\epsilon)A\\right)}-\\beta D_{\\text{KL}}(\\pi_\\theta||\\pi_{\\theta_{\\text{ref}}})\\right]$，其中 $A=[r^{\\text{WM}}-\\text{mean}(r^{\\text{WM}})]/{\\text{std}(r^{\\text{WM}})}$。",
        "输出": "训练后的LLM世界模型 $\\pi_\\theta$，能够准确预测环境动态，为后续基于任务成功奖励的RL微调奠定基础。"
    },
    "关键模块、符号与假设": {
        "关键模块": [
            "LLM策略/世界模型 $\\pi_\\theta$：接收历史交互和动作，生成推理和下一状态预测。",
            "奖励函数 $r^{\\text{WM}}$：基于预测状态与真实状态的相似度提供二元奖励。",
            "距离计算 $d(\\hat{s}_{t+1}, s_{t+1})$：使用预训练嵌入模型 $E(\\cdot)$ 和余弦相似度。",
            "优化器GRPO：使用组相对优势（group-relative advantage）进行策略优化。",
            "数据筛选机制：基于辅助模型 $\\pi'_\\theta$ 和阈值 $\\tau_{\\text{easy}}$ 过滤简单样本，优先训练困难样本。"
        ],
        "关键符号": [
            "$\\mathcal{S}, \\mathcal{A}, \\mathcal{T}, \\mathcal{R}, \\gamma$：MDP（马尔可夫决策过程）的状态、动作、转移函数、奖励函数和折扣因子。",
            "$s_t, a_t$：时间步 $t$ 的真实状态和动作。",
            "$\\hat{s}_t$：模型生成的状态（与世界模型相关）。",
            "$H$：历史交互的回合数。",
            "$\\tau_d$：奖励函数中的距离阈值。",
            "$\\tau_{\\text{easy}}$：数据筛选中的容易样本阈值。",
            "$\\beta, \\epsilon$：GRPO中的KL正则化系数和裁剪范围。"
        ],
        "关键假设": [
            "环境交互可建模为MDP（或部分可观测MDP，POMDP），但为简化使用 $s_t$ 表示观测。",
            "二元奖励比连续奖励更鲁棒，不易被“黑客攻击”（hacking）。",
            "通过过滤简单样本（使用辅助模型 $\\pi'_\\theta$ 和阈值 $\\tau_{\\text{easy}}$），可以提升世界模型学习非平凡知识的效果。"
        ],
        "关键公式": [
            "状态预测：$(\\text{reason},\\hat{s}_{t+1})\\sim\\pi_\\theta(\\cdot|s_{\\leq t},a_t); s_{\\leq t}\\equiv\\left\\langle s_{t-H},a_{t-H},...,s_t\\right\\rangle$",
            "奖励函数：$r^{\\text{WM}}(\\hat{s}_{t+1},s_{t+1})=\\begin{cases}1.0,&\\text{if }d(\\hat{s}_{t+1},s_{t+1})<\\tau_d,\\\\ 0.0,&\\text{otherwise.}\\end{cases}$",
            "距离计算：$d(\\hat{s}_{t+1},s_{t+1})=1-\\cos(E(\\hat{s}_{t+1}),E(s_{t+1}))$",
            "GRPO目标函数：$\\mathbb{E}_{\\pi_{\\theta_{\\text{old}}}}\\left[\\min{\\left(\\rho_\\theta A,\\text{clip}(\\rho_\\theta,1\\pm\\epsilon)A\\right)}-\\beta D_{\\text{KL}}(\\pi_\\theta||\\pi_{\\theta_{\\text{ref}}})\\right]$，其中 $\\rho_\\theta=\\pi_\\theta(y|x)/\\pi_{\\theta_{\\text{ref}}}(y|x)$，$A=[r^{\\text{WM}}-\\text{mean}(r^{\\text{WM}})]/{\\text{std}(r^{\\text{WM}})}$",
            "数据筛选条件：$\\frac{1}{K}\\sum\\limits_{K}r^{\\text{WM}}(\\hat{s}_{t+1},s_{t+1})\\geq\\tau_{\\text{easy}}$，其中 $K=10$"
        ]
    },
    "与baseline或常见做法的差异": {
        "主要差异": "RWML（Reinforcement World Model Learning）是一种自监督的、可扩展的世界模型学习方法，旨在解决传统RL方法在复杂任务中依赖稀疏、专家设计的任务成功奖励（task-success rewards）的扩展性挑战。",
        "具体对比": [
            "**奖励设计**：常见RL方法直接使用任务成功奖励（稀疏、需专家设计），而RWML使用基于状态预测相似度的二元奖励 $r^{\\text{WM}}$（密集、自监督），无需任务成功信号。",
            "**训练目标**：传统方法直接优化任务完成，RWML先训练LLM作为世界模型（学习环境动态 $\\mathcal{T}$），再微调任务奖励，分阶段提升可扩展性。",
            "**数据需求**：RWML不依赖专家数据、更强LLM或任务奖励信号，仅通过环境交互收集数据，降低了数据获取成本。",
            "**数据筛选**：引入基于辅助模型 $\\pi'_\\theta$ 和阈值 $\\tau_{\\text{easy}}$ 的筛选机制，过滤简单样本，优先学习困难的世界模型知识，提升效率。"
        ]
    },
    "实现或复现需注意的细节": {
        "数据收集": "使用目标模型 $\\pi_\\theta$ 在环境中收集轨迹 $(s_0,a_0,s_1,a_1,...,s_T)$，并转换为三元组 $\\left\\langle s_{\\leq t},a_t,s_{t+1}\\right\\rangle$ 用于训练。每个训练任务进行 $N>1$ 次轨迹收集以增加覆盖和多样性。",
        "数据筛选步骤": [
            "1. 使用10%的全数据集通过SFT（监督微调）训练一个辅助LLM $\\pi'_\\theta$，用于预测 $\\hat{s}_{t+1}$。",
            "2. 用 $\\pi'_\\theta$ 在剩余90%数据（训练集）上生成预测，对每个样本进行 $K=10$ 次尝试，计算平均奖励 $\\frac{1}{K}\\sum_{K}r^{\\text{WM}}(\\hat{s}_{t+1},s_{t+1})$。",
            "3. 若平均奖励 $\\geq \\tau_{\\text{easy}}$，则视为“简单样本”，仅以概率 $p=0.1$ 纳入最终训练集，否则全部纳入。优先保留困难样本，同时保持多样性。"
        ],
        "超参数设置": [
            "距离阈值 $\\tau_d$：用于二元奖励的判定。",
            "容易样本阈值 $\\tau_{\\text{easy}}$：用于数据筛选。",
            "KL正则化系数 $\\beta$ 和裁剪范围 $\\epsilon$：在GRPO优化中使用。",
            "尝试次数 $K=10$ 和简单样本保留概率 $p=0.1$：在数据筛选中固定。"
        ],
        "技术依赖": [
            "使用现成的嵌入模型 $E(\\cdot)$（如基于Karpukhin et al., 2020; Zhang et al., 2025b）计算余弦相似度。",
            "优化方法采用GRPO（Shao et al., 2024; DeepSeek-AI et al., 2025），需实现组相对优势计算。"
        ],
        "环境适配": "方法适用于多种环境，如ALFWorld（动作如“go to sidetable 1”，状态为自然语言描述）和 $\\tau^2$ Bench（动作包括工具调用或用户响应，状态为JSON或自然语言）。具体细节需参考附录B和C。"
    }
}