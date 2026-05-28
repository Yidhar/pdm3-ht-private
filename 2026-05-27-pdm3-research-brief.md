# PDM-3 三层级解耦扩散模型 · 研究任务书

> **文档性质**:这是一份交给研究型 AI agent 的自包含研究任务书。阅读者无法访问本项目的本地文件或对话上下文,所有必要背景信息都已包含在本文档中。
>
> **研究主题**:PAE × DDT × MeanFlow 三层级解耦扩散模型 (PDM-3) 的实现路径与隐藏风险
>
> **使用方式**:把这份文档的完整内容粘贴给具备深度研究能力的 AI(Claude Deep Research / o1 Pro / Gemini Deep Research / 同类产品),让其按第四部分指定的格式输出报告。

---

## 第一部分:项目全景

### 1.1 项目目标

我们要训练一个**主干参数 < 1B、生成质量强劲的扩散图像生成模型**,这是一个硕士/博士开题级别的项目。

具体量化目标:

| 评估项 | 目标值 | 参考 SOTA |
|--------|--------|-----------|
| ImageNet 256 class-cond, FID (no CFG) | ≤ 1.8 | REPA-E: 1.83, DDT: 1.31 |
| ImageNet 256 class-cond, FID (w/ CFG + guidance interval) | ≤ 1.3 | REPA-E: 1.26, PAE: 1.03 |
| ImageNet 256, 1-NFE FID | ≤ 3.0 | MeanFlow: 3.43 |
| 主干参数量 | < 1B | DiT-XL/2: 675M, SiT-XL/2: 675M |

**为什么是 ImageNet 256 class-cond 而非 T2I**:我们的算力(Pro 6000 × 1 常驻 + B300 × 1 短时 burst, 折合约 90 H100-day 总预算)无法支撑 T2I 从零训练。ImageNet 256 是几乎所有相关 SOTA 论文(REPA / VA-VAE / LightningDiT / REPA-E / DDT / MeanFlow / PAE / JiT / PixelDiT)的对比擂台,在这里发声影响力远高于半成品 T2I。

**项目预期 contributions**:

1. **Empirical**:在 <1B 参数尺度上对 ImageNet 256 class-cond 取得新 SOTA(FID ≤ 1.3 w/ CFG)
2. **Theoretical**:证明 Proposition 1 —— PAE 的 Manifold Continuity Regularization (MCR) 损失对 MeanFlow 训练中 JVP 项的数值稳定性提供严格保证
3. **Architectural**:首次将 MeanFlow 的 average-velocity 训练目标移植到 DDT 的 encoder-decoder 解耦架构上,实证验证两者的因子分解天然对齐

**投稿目标**:NeurIPS 2026 (截止 ~5月) 或 ICLR 2027 (截止 ~9月)

### 1.2 技术架构详解

PDM-3 是三个独立工作的**正交组合**,作用在系统的不同抽象层:

```
原始图像 x
   │
   ↓ [PAE Encoder]              ← 第一层解耦 (latent)
                                  PAE = Prior-Aligned AutoEncoder
                                  用 frozen VFM (DINOv2) 作语义参照
                                  + DAM (Detail-Aware Modulator) 注入细节
                                  + 三个对齐损失 (SSR/MCR/SCR) 显式重塑流形
   ↓
Latent z ∈ R^(32×32×32)         (用 PAE-f16d32 默认配置)
   │
   ↓ 加噪 z_t = (1-t)z + tε      (Flow Matching linear interpolant)
   ↓
[DDT Network in MeanFlow form]   ← 第二+第三层解耦 (architecture + objective)
  ┌──────────────────────────────┐
  │  Encoder f_e(z_t, t, c)      │  DDT condition encoder
  │  → semantic feature s_t      │  只看 t,不看 r (新设计)
  │  典型配置:DDT-L 16En/8De    │
  │  (在 PAE latent 下,可能从     │
  │   原 DDT 的 20En/4De 偏向     │
  │   更平衡的比例)               │
  ├──────────────────────────────┤
  │  Decoder f_d(z_t, s_t, r, t) │  DDT velocity decoder
  │  → average velocity u        │  看 (r,t),输出 MeanFlow 平均速度
  └──────────────────────────────┘
   ↓
训练目标:MeanFlow Identity
  u_tgt = v - (t-r)·[v·∂_z u_θ + ∂_t u_θ]
  loss = ||u_θ - sg(u_tgt)||²
   ↓
推理: x = ε - u_θ(ε, r=0, t=1)   (1-NFE 单步生成)
                                   (亦可多步:z_r = z_t - (t-r)·u_θ(z_t, r, t))
```

**三个组件各自的技术核心**:

**PAE (Prior-Aligned AutoEncoder)** ([arXiv: 2605.07915](https://arxiv.org/abs/2605.07915), GitHub: `ZhengrongYue/PAE`):
- 论文核心论点:重建保真度 rFID 与生成质量 gFID 相关性弱;真正决定 gFID 的是 latent manifold 的三个几何性质:
  - **SSC (Spatial Structure Coherence)**:邻近 patch 的 latent 空间结构一致
  - **LPC (Local Manifold Continuity)**:latent 平滑,小扰动 → 小变化
  - **GSQ (Global Manifold Semantics)**:latent 按语义聚类
- 用 frozen VFM 作"语义参照",DAM 模块注入像素细节防止 latent 全被 VFM 主导
- 三个对齐损失:SSR (空间结构正则)、MCR (流形连续性正则)、SCR (语义一致性正则)
- 报告结果:ImageNet 256 gFID **1.03** (LightningDiT-XL/1, SOTA),比 RAE 收敛快 13×

**DDT (Decoupled Diffusion Transformer)** ([arXiv: 2504.05741](https://arxiv.org/abs/2504.05741), CVPR 2026 highlight, GitHub: `MCG-NJU/DDT`):
- 论点:DiT 同一组层同时干"抽语义"(低频)和"出细节"(高频),目标冲突。应该解耦成两个网络。
- Condition Encoder 从 noisy 输入抽 semantic self-condition,Velocity Decoder 用 self-condition + noisy 输入解码 velocity
- 最优 Encoder/Decoder 比例随模型规模变化:
  - DDT-B/2: 8En/4De
  - DDT-L/2: 20En/4De (反直觉地 encoder-heavy)
  - DDT-XL/2: 22En/6De
- 报告结果:ImageNet 256 FID **1.31**,比 vanilla DiT 训练快 4×
- 附带优势:相邻 denoising step 间 encoder 输出可复用(动态规划找最优共享 schedule),推理加速 1.5-2×

**MeanFlow** ([arXiv: 2505.13447](https://arxiv.org/abs/2505.13447), NeurIPS 2025 Oral, Kaiming He / J. Zico Kolter):
- 论点:不要学瞬时速度然后数值积分,直接学**平均速度**:
  $$u(z_t, r, t) := \frac{1}{t-r}\int_r^t v(z_\tau, \tau)\,d\tau$$
- 微分 MeanFlow Identity 把训练目标变得可计算:
  $$u(z_t, r, t) = v(z_t, t) - (t-r) \cdot \frac{du}{dt}$$
- 用 JVP(`torch.func.jvp` / `jax.jvp`)计算 `du/dt`,只多一次 backward(~16% overhead)
- 25% 样本采 r≠t,75% 样本采 r=t(此时退化为 Flow Matching)
- 推理:1-NFE 单步,$x = \epsilon - u_\theta(\epsilon, 0, 1)$
- 报告结果:ImageNet 256 1-NFE FID **3.43**,从头训,无蒸馏无 curriculum

**架构上的关键创新点(我们做的)**:把 DDT 的 encoder-decoder 切分塞进 MeanFlow 训练目标里。具体地,encoder $f_e(z_t, t, c) \to s_t$ 只依赖时刻 $t$(语义内容与"位移区间" $(r,t)$ 无关),decoder $f_d(z_t, s_t, r, t) \to u$ 接收 encoder 输出 + 时间区间。这个因子分解是**自然的**——DDT 的"语义/细节"切分恰好对应 MeanFlow 的"内容/位移"切分。

### 1.3 走过的路(为什么走到 PDM-3 这条路径)

整个调研由"主干 <1B 强生成质量"这个具体约束驱动,经历了三轮筛选。

**第一轮:大格局的范式选择**

我们系统调研了 2024-2026 年扩散图像生成的主要方向:

| 范式 | 代表工作 | <1B 适用性 |
|------|---------|----------|
| Latent Diffusion + DiT | SD3, FLUX | ⚠️ 单卡训 12B 不现实 |
| Pixel-space DiT | JiT, PixelDiT, DiP | ✅ 新方向,空间大但风险高 |
| 表征对齐加速训练 | REPA, REPA-E, VA-VAE, PAE | ✅ <1B 模型最受益 |
| 自回归 (VAR/MAR) | VAR, MAR, MaskGIL | ✅ 蓝海但生态不成熟 |
| 单步生成 | MeanFlow, Consistency Models | ✅ 训练目标改进可加速 |
| 架构改进 | DDT, MMDiT, S3-DiT, SANA Linear-DiT | ✅ 多个独立工作可叠加 |

**第二轮:确定主路径**

在"算力够、SOTA 还在快速演进、有可叠加空间"三个条件下,我们排除了:
- 纯 T2I 从零训练(数据 + 算力双重不可行)
- 大模型 (>2B) (单卡不可达)
- VAR/MAR 路线(可能 promising,但要从基础设施开始搭,周期不可控)

选定方向:**Latent Diffusion + DiT 主流框架内的多组件叠加**,这条线 SOTA 数字最透明、可比性最强、生态最成熟。

**第三轮:挑选具体组件并验证可叠加性**

我们列出了 ImageNet 256 class-cond 上的几个 SOTA 配方:

| 配方 | FID (w/ CFG) | 时间 | 我们的判断 |
|------|------|------|----------|
| LightningDiT + VA-VAE | 1.35 | 2025.01 | 工程基线,已成熟 |
| REPA | 1.42 | 2024.10 (ICLR'25 Oral) | DiT 浅层对齐 DINOv2 |
| REPA-E | 1.26 | 2025.04 (ICCV'25) | VAE + DiT 联合训练 |
| DDT-XL/2 | 1.31 | 2025.04 (CVPR'26) | 编解耦合架构 |
| MeanFlow-XL/2 (2-NFE+) | 2.20 | 2025.05 (NeurIPS'25 Oral) | 单/双步生成的训练目标 |
| **PAE (frozen LightningDiT)** | **1.03** | **2026.05** | **三属性显式重塑流形 (最新 SOTA)** |

我们注意到:
- PAE × DDT 没人组合过
- PAE × MeanFlow 没人组合过
- DDT × MeanFlow 没人组合过
- **三者全部组合**(PDM-3)从未有人系统验证

理由分析后,**这种组合的潜在协同性较高**:
- PAE 在 latent 端解耦(让 latent 几何性质好)
- DDT 在网络结构端解耦(让语义/细节不互相干扰)
- MeanFlow 在训练目标端"解耦"(把多步 ODE 积分变成单步 average velocity 预测)
- 三者作用层不同,理论上正交;实际上有数学协同(见 Proposition 1)

**Proposition 1 (核心理论 nugget)**:
PAE 的 MCR 损失约束 decoder 的 Lipschitz 常数 $L_D$ → 隐式约束 latent 分布 $p_z$ 的几何条件数 $\kappa(p_z)$ → score 函数 $\nabla \log p_t(z_t)$ 的 Lipschitz 性 → velocity field $v(z_t, t)$ 的 Lipschitz 性 → MeanFlow 训练中 JVP 项 $\|du/dt\|$ 有界 → 训练目标方差有界 → SGD 数值稳定。

形式上:存在常数 $C(T, L_D, p_z)$ 使得对 $t \in [t_0, t_1] \subset (0,1)$ 紧致区间:
$$\mathrm{Var}_{(z_t, r, t)}[u_{\text{tgt}}(z_t, r, t)] \leq C \cdot (1 + \mathbb{E}\|v\|_2^2)$$
其中 $C \to \infty$ 当 $L_D \to \infty$ 或 $\kappa(p_z) \to \infty$。

这个命题给出了三个**可证伪的实验预测**:
- **推论 1**:同一 MeanFlow 训练下,PAE tokenizer 的 JVP norm 分布显著低于 SD-VAE / VA-VAE
- **推论 2**:同一 MeanFlow 训练下,PAE 配置中 NaN 出现频率显著低于 SD-VAE / VA-VAE(尤其在 bf16 下)
- **推论 3**:PAE 下允许的最大稳定学习率高于 SD-VAE / VA-VAE

如果**三个推论都验证**,Proposition 1 落地;**至少有 2 个验证**,论文有完整的理论-实验闭环;**全失败**,理论故事崩塌,需要回退到工程性 framing 或者改投 DDT × MeanFlow 的小论文。

### 1.4 计算预算与时间线

**算力**:
- **Pro 6000 (Blackwell, 96GB)** × 1,常驻,bf16 持续训练算力对标 H100 的 ~70-80%
- **B300 (288GB HBM3e)** × 1,**仅可短时借用**,fp8 算力对标 H100 的 ~5×
- 折合总预算约 **87 H100-day**

**时间线**(15 周):

| Phase | 周次 | 算力 | 关键目标 |
|-------|------|------|----------|
| 0: 环境 + Baseline | 1-2 | Pro 6000 | 4 个 baseline 复现 |
| 1: 三组件独立集成 | 3-5 | Pro 6000 | 每个组件单独 work |
| 2: 两两组合验证 | 6-7 | Pro 6000 | JVP norm 数据采集 |
| 3: 三联合主训 | 8-10 | **B300 burst** | 冲 SOTA FID |
| 4: 理论 ablation | 11-12 | Pro 6000 | 验证 Prop 1 三推论 |
| 5: 论文写作 | 13-15 | — | 投稿 |

**Phase 3 是 B300 的唯一使用窗口**,必须提前锁定排期。

### 1.5 PDM-3 三组件的预设角色(我们目前的理解)

我们对三个组件**协同后的行为**只有 high-level 直觉,具体技术细节是黑箱:

| 阶段 | PAE 的作用 | DDT 的作用 | MeanFlow 的作用 | 我们不了解的部分 |
|------|----------|----------|----------|------------|
| **数据→latent** | 用 VFM 重塑流形,SSR/MCR/SCR 损失训练 | — | — | DAM 模块在 MeanFlow 训练下还需要吗?MCR 的扰动幅度是否需要重调? |
| **latent→网络输入** | latent 是 PAE 的 32 维 | — | — | PAE-f16d32 vs PAE-f16d16 vs PAE-f16d64 在 MeanFlow 下的偏好 |
| **网络前向** | — | encoder 抽 semantic,decoder 出 velocity;关键:encoder 只看 t | — | DDT 的 encoder/decoder 比例在 PAE latent 下应该怎么调?(我们预测 encoder 可以变浅) |
| **训练目标** | 三个对齐损失已经训好,frozen | — | 平均速度建模,JVP 计算 du/dt;25% r≠t | PAE 是否需要和 DiT 联合 fine-tune (类 REPA-E 路线)?MeanFlow 的 r/t 采样在 DDT 因子分解下行为如何? |
| **CFG 整合** | — | 标准 CFG | MeanFlow 自带 CFG 内置 (u^cfg 直接建模),无需 sampling-time 加权 | DDT × MeanFlow CFG 整合的最优 ω 和 guidance interval 在 PAE 下应该多少? |
| **采样** | — | encoder-decoder 共享可加速多步采样 | 1-NFE 单步采样 | 在 MeanFlow 1-NFE 下,DDT 的 encoder 共享 trick 完全失效——是否还有替代的推理加速? |

**关键不确定性的总结**(我们需要外部研究帮我们看清的部分):

1. JVP 在 DDT 因子分解下,bf16 训练数值稳定性的实际表现
2. PAE 的三个 alignment 损失在 MeanFlow 训练循环中是否需要重调权重 / schedule
3. DDT 的 encoder/decoder 比例在 PAE latent 下的偏移方向与幅度
4. MeanFlow CFG (built-in) 与 DDT 架构整合时的细节
5. **Proposition 1 的实证基础是否扎实**——三个推论中如果有的不成立,如何调整 framing

---

## 第二部分:探索任务

> **致研究 AI 的说明**:
>
> 以下不是一系列需要你"回答"的问题。这是一系列**探索区域**——每个区域描述了我们遇到的一个具体困境,以及我们目前有限的理解。
>
> 你的任务是:
>
> 1. **先深入理解**每个困境的本质——不是表面的"怎么用 [组件]",而是底层的技术挑战是什么
> 2. **然后自由探索**所有可能的技术路径——**尤其是我们完全没想到的**
> 3. **特别关注"跨领域借鉴"**:神经 ODE、常微分方程数值积分、最优控制、normalizing flows、score-based 生成模型、自动微分 (AD/JVP)、隐式微分、随机优化稳定性理论——这些邻近领域有没有解决过类似问题?它们的方案能不能搬过来?
> 4. 对每条发现的路径,**给出具体到可以开始编码的实现方案**
>
> 我们是扩散模型的应用研究者,不是数值分析或优化理论专家。我们能想到的方案大概率是最表面的几种。请像一个**同时精通扩散模型、神经 ODE、自动微分系统和数值优化稳定性**的资深专家那样思考——**你觉得理所当然的东西,对我们来说可能是全新的发现。**

### 探索区域 1: DDT 因子分解下 MeanFlow JVP 的数值稳定性

**困境**:

MeanFlow 训练的核心是 JVP 计算:

```python
# 标准 DiT 上的 MeanFlow JVP
u, du_dt = jvp(net_dit, (z, r, t), (v, 0, 1))
```

但 DDT 把 `net_dit` 拆成 `encoder + decoder`:

```python
# DDT × MeanFlow
s = encoder(z, t, c)              # encoder 只看 t
u = decoder(z, s, r, t)           # decoder 看 (r, t)
# JVP 要走 chain rule 穿过两层
```

`du/dt` 通过链式法则展开为:
$$\frac{du}{dt} = \frac{\partial f_d}{\partial t}\Big|_{\text{direct}} + \frac{\partial f_d}{\partial s} \cdot \frac{\partial f_e}{\partial t}\Big|_{\text{via encoder}}$$

理论上,`torch.func.jvp` 通过 autograd 系统自动处理链式法则;**但**:

- MeanFlow 论文是 JAX + TPU + fp32 训出来的;我们在 PyTorch + Blackwell GPU + bf16 下跑
- DDT 增加了网络深度(encoder 22 层 + decoder 6 层 = 28 层 vs 标准 DiT 28 层),JVP 经过的层数相同,**但路径分叉**
- bf16 下 JVP 在 small-t 区域 ($\sigma_t \to 0$ 或 $1-t \to 0$) 数值精度损失严重,early-training NaN 风险高

**我们这个外行能想到的做法**:

> 1. 前 50K iter 用 fp32 训练,稳定后切 bf16
> 2. 在 JVP 计算路径上用 `torch.amp.autocast(enabled=False)` 强制 fp32
> 3. 加大 gradient clipping (e.g., max_norm=0.5)
> 4. 跳过 `t < 0.05` 或 `t > 0.95` 区域的样本

**但我们怀疑这只是冰山一角。**

请自由探索:

- 神经 ODE 社区(neural ODE,Chen et al. 2018 起)长期处理类似问题——他们用过哪些技巧?adjoint method?implicit differentiation?
- 自动微分系统的研究(JAX, Enzyme, Diffrax)有没有针对 chained JVP 的稳定性技巧?
- **Stop-gradient 的位置可以变吗?**MeanFlow 论文在 target 上用 sg,但在 DDT 下可能 encoder 和 decoder 的 sg 策略不同
- **能不能 JVP-free?** 有没有论文绕过 JVP 计算 `du/dt`,比如用差分近似 + variance reduction?(SplitMeanFlow [arXiv: 2507.16884] 用了 algebraic identity 避免 JVP——这条路应该深入看一下)
- **第二阶**:如果允许部分 fp32 部分 bf16,**最优精度配置**是什么?(input fp32 + matmul bf16 + JVP path fp32?)
- 跨领域:**量子化学** 里的能量梯度计算也有类似 JVP 稳定性问题,他们的 "stochastic gradient" 方法有借鉴价值吗?

---

### 探索区域 2: PAE 三个 alignment 损失在 MeanFlow 下的调度

**困境**:

PAE 的三个损失 (SSR + MCR + SCR) 是在**普通 Flow Matching 训练**下调优的——具体地,LightningDiT 用 v-prediction + Flow Matching 训练时,PAE 的损失权重和扰动幅度已经收敛到一个 sweet spot。

但 MeanFlow 训练有几个**根本不同**:
1. 训练时 25% 样本采 r≠t,75% 样本采 r=t——网络要同时学瞬时速度 (r=t 时) 和平均速度 (r≠t 时)
2. JVP 计算要求 latent space 的 manifold 性质比普通 FM 更严格(见 Proposition 1)
3. MeanFlow 关心的是 velocity field **在轨迹上的积分**,而非单点 velocity

**PAE 的三个损失实际控制了三个不同的几何属性**:
- SSR (Spatial Structure Regularization):每个 latent 与对应 VFM feature 对齐 → 控制空间结构
- MCR (Manifold Continuity Regularization):扰动 latent 再 decode,感知一致 → 控制局部连续
- SCR (Semantic Consistency Regularization):latent 全局 pool 与 VFM 全局对齐 → 控制全局语义

**MeanFlow 对 PAE 的真正需要是什么?** 我们直觉是 MCR (manifold continuity) 最关键,因为它直接影响 JVP 稳定性。但这只是直觉,没有实证。

**我们这个外行能想到的做法**:

> 1. 直接照搬 PAE 默认权重训(快但可能 suboptimal)
> 2. 简单 grid search SSR/MCR/SCR 权重比(贵但 brute-force)
> 3. 训完 PAE 后冻结,完全用作 frozen tokenizer

**但我们怀疑这只是冰山一角。**

请自由探索:

- **MeanFlow-aware PAE 损失**:能不能设计一个**为 average velocity 量身定制的 alignment 损失**?比如"对齐 latent 在 [r,t] 区间的 average direction"?
- **Curriculum schedule**:训练早期用强 MCR(稳定 JVP),后期降 MCR(允许网络捕捉细节)。有没有现成的 dynamic loss weighting 方法适合这种情形?(GradNorm? PCGrad? Multi-Objective Optimization?)
- **PAE × REPA-E 双层端到端**:REPA-E 把 VAE 和 DiT 联合训。**MeanFlow 下能不能做 PAE × MeanFlow 端到端**?stop-gradient 该放哪?
- 跨领域借鉴:**Optimal transport** 里的 Wasserstein gradient flow 也有"轨迹平均"的概念,他们如何在 latent 端约束?
- **理论层**:Proposition 1 只用了 MCR。**SSR 和 SCR 对 MeanFlow 有没有数学上可证明的作用?**(比如 SSR 影响 v 的空间平滑度,SCR 影响轨迹的长程一致性)
- 反向问题:**有没有 alignment 损失 + average velocity 训练会冲突的情形?**(比如 alignment 强行把 latent 拉到 VFM 流形上,但 VFM 流形与 MeanFlow 想要的 "smooth trajectory" 流形不一致)

---

### 探索区域 3: DDT encoder/decoder 比例在 PAE latent 下的重平衡

**困境**:

DDT 论文给出的 ratio 是在 SD-VAE latent 上调出来的:
- DDT-B/2: **8En/4De** (encoder 占 2/3)
- DDT-L/2: **20En/4De** (encoder 占 5/6,极度 encoder-heavy)
- DDT-XL/2: **22En/6De** (encoder 占 ~22/28)

**底层逻辑**:模型越大,Encoder 越深,因为"抽语义"的能力上限越高,模型越大就越有余地把容量放在 encoder。

**但 PAE 已经在 latent 里塞了语义!** 如果 latent 已经天然是"半成品语义表征" (DINOv2-aligned),DDT 的 encoder 还需要做多少工作?

我们的预测:**在 PAE latent 下,DDT 的最优 ratio 会向 decoder 偏移**。比如 DDT-L 在 PAE 上的最优可能是 16En/8De 或更平衡。**但具体偏移多少,理论上没有公式。**

**我们这个外行能想到的做法**:

> 1. 在 PAE latent 上做 ratio sweep:{24En/0De, 20En/4De, 16En/8De, 12En/12De} × 80 epoch
> 2. 比较收敛速度和最终 FID
> 3. 选最优配置

**但我们怀疑这只是冰山一角。**

请自由探索:

- **理论指导**:有没有数学方法预测 PAE latent 下的最优 encoder 容量?(信息论?表征论?互信息分析 latent 与图像之间剩余的"semantic gap"?)
- **自适应 ratio**:能不能让模型在训练中**自己学**最优容量分配?(类似 Mixture of Experts 但在深度而非宽度上分配)
- **Neural Architecture Search**:有没有适合 DiT 类模型的 NAS 框架,可以搜 encoder/decoder 比例 + 各层宽度?
- **Encoder/decoder 不平衡的极端**:如果 PAE 已经把语义完全塞进 latent,**DDT 退化到只有 decoder 的情况是否更好?**(即 0En/28De,等价于普通 DiT)
- 跨领域:**Scaling laws** 文献(Chinchilla, etc.)有没有针对"语义已经预编码"的输入做过 capacity allocation 分析?
- **Encoder 共享推理 trick**:在 PAE latent 下,encoder 输出的相邻时间步相似度可能比 SD-VAE 下更高(因为 latent 更平滑)→ encoder 共享 schedule 是否可以更激进?

---

### 探索区域 4: 1-NFE 推理在 <1B 模型上的 CFG 集成与质量提升

**困境**:

MeanFlow 的 CFG 是**内置在训练目标里**的,不需要 sampling 时做 2-NFE:

$$v^{\text{cfg}}(z_t, t \mid c) = \omega v(z_t, t \mid c) + (1-\omega) v(z_t, t)$$

训练时把 $v^{\text{cfg}}$ 当 target 直接学,得到 $u^{\text{cfg}}_\theta$,推理时一步生成:$x = \epsilon - u^{\text{cfg}}_\theta(\epsilon, 0, 1 \mid c)$。

**但有两个我们不确定的点**:

1. **MeanFlow 论文的 CFG 是在标准 DiT 架构上调的**。DDT 因子分解下,encoder 看 t,decoder 看 (r,t)——CFG 这种"两种 condition 混合"在 DDT 下应该放哪里?(在 encoder 上做 CFG?在 decoder 上做?两者都做?)
2. **MeanFlow 论文报告最优 CFG 在 ω≈3.0 (small B/4 model)**,大模型 (XL/2) 用了 ω≈2.0;但**用了 guidance interval 后**(REPA 的 trick),最优 ω 又变了。**PDM-3 下的最优 CFG 配置完全没人测过**。

**我们这个外行能想到的做法**:

> 1. CFG 完全照搬 MeanFlow XL/2 的 ω≈2.0,在 DDT 上 encoder 和 decoder 都加 class condition
> 2. 加 guidance interval [0, 0.7] (REPA 的 SOTA 配置)
> 3. 对 ω 和 interval 各做一个 sweep

**但我们怀疑这只是冰山一课。**

请自由探索:

- **CFG-free 替代**:近年有很多 CFG-free 但效果接近的工作(Self-Attention Guidance, Perturbed Attention Guidance, etc.)——这些方法在 DDT × MeanFlow 上能用吗?
- **轨迹 CFG**:MeanFlow 的 CFG 是对**整段轨迹**的"内置"——能不能在不同 (r,t) 区间用不同 CFG 强度?(类似 guidance interval 但更细粒度)
- **架构感知 CFG**:DDT 的 encoder 学语义,decoder 学细节。**只在 encoder 加 CFG**(强化语义)、**只在 decoder 加 CFG**(强化细节)、**两者权重分别调**——哪个更好?
- **<1B 模型的 CFG 病理**:小模型在高 ω 下容易过饱和(over-saturation),有没有"对小模型友好"的 CFG schedule?
- 跨领域:**Conditional generation in NLP** (类似 PPO with reward model) 有"too strong steering 反而降质量"的现象,他们有 robust steering 方法吗?
- **1-NFE + few-NFE 混合**:能不能训练时学到一个**"可调"的 NFE 模型**——用户根据需要选 1, 2, 4, 8 步推理?(Shortcut models, IMM 走过类似路线)

---

### 探索区域 5: Proposition 1 的可证伪性、反例情况与 fallback framing

**困境**:

Proposition 1 (PAE-MCR ⇒ MeanFlow JVP 稳定性) 是我们整篇论文的理论支柱,但它的实证基础**只有三个推论**:

| 推论 | 实验 | 如果失败怎么办? |
|------|------|----------|
| 推论 1: PAE 下 JVP norm 显著低于 SD-VAE/VA-VAE | 收集 100K JVP 样本,画分布 | **理论崩塌,需要回退** |
| 推论 2: PAE 下 NaN 频率显著低 (bf16) | 多种子统计 | 理论部分崩塌,可改 framing |
| 推论 3: PAE 下允许的最大 lr 更高 | lr sweep 找临界点 | 次要证据,失败影响小 |

**最坏情况**:三个推论全部 inconclusive(差异在统计噪声内)——这时论文的理论部分要么改成纯 architectural story (Proposition 2: DDT 的 encoder 低秩归纳偏置),要么砍掉理论部分变成纯 empirical paper。

**但我们对"什么情况下推论会失败"几乎没有理解**——这是我们的盲区。

**我们这个外行能想到的做法**:

> 1. 实验设计上多保险:大 batch size 收集 JVP 样本(减小统计噪声)
> 2. 对比 3-4 个 tokenizer 而非 2 个(看曲线趋势而非两点对比)
> 3. 准备 Plan B framing: "DDT × MeanFlow 是 main contribution, PAE 是 cherry on top"

**但我们怀疑这只是冰山一角。**

请自由探索:

- **数值分析视角**:JVP norm 真的是衡量 MeanFlow 训练稳定性的最佳指标吗?有没有更精确的稳定性指标?(condition number of Jacobian? Lyapunov exponent? Gradient signal-to-noise ratio?)
- **理论加固**:Proposition 1 假设了 LPIPS ≈ L2(感知度量与 L2 等价),这个 Bi-Lipschitz 常数实际多大?能否给出严格 bound?
- **反例构造**:能否构造一个**"满足 MCR 但 JVP 不稳定"** 的反例?(比如某种病理 latent 分布)。如果能构造,Proposition 1 需要加更多假设。
- **更强的 PAE 改造**:如果当前 PAE 不能拯救 JVP 稳定性,**能不能设计一个针对 MeanFlow 的新 tokenizer**?(MeanFlow-aware PAE? Trajectory-smoothing VAE?)
- **跨领域**:**控制理论**里的 system stability 分析(Lyapunov, robust control)对此有借鉴吗?
- **元问题**:有没有可能,**Proposition 1 是对的,但 ImageNet 256 这个 benchmark 本身不够 challenging,看不出差异**?需要在 ImageNet 512 或 LAION 子集上验证?
- **如果三推论全失败**,你建议我们怎么 reframe 整个故事?是不是应该把理论 contribution 完全砍掉,转向纯实证 + 工程论文?

---

### 探索区域 6: 我们没想到的方向

以上 5 个区域是我们这些外行能划定的范围。**但几乎可以肯定,还有我们完全不知道的方向存在。**

请特别关注:

- **最近 6 个月发表的相关新论文、新项目**(2026.01-2026.05 区间)——特别是 MeanFlow 的 follow-ups (SplitMeanFlow, Re-Meanflow, Pixel MeanFlow, Improved Mean Flows, etc.)、PAE 的同期工作(RAE: Representation Autoencoder, ReDi, FAE, AlignTok, RePack)、DDT 之后的架构创新
- **其他领域解决类似问题的方法**:
  - **神经 ODE / Diffrax / Neural Stochastic Differential Equations**——adjoint method, implicit differentiation, stochastic optimal control
  - **隐式生成模型**(GAN, Normalizing Flows)——有没有"learn the integral directly"的更老想法值得回看?
  - **Reinforcement learning** 的 policy gradient 也有类似"沿轨迹积分"的目标,有 variance reduction 经验
- **三个组件中,有没有哪个其实**不应该用**?**比如 PAE 也许过度工程化,简单的 VA-VAE 就够;或者 DDT 在 MeanFlow 下其实是负贡献
- **完全跳出当前思维框架的方案**:
  - 能不能完全不用 PAE,而用 frozen DINOv2 features 直接做 token? (RAE 路线)
  - 能不能不用 DDT,而用 mixture-of-experts (DiT-MoE, DiffMoE) 替代解耦?
  - 能不能不用 MeanFlow,而用 Consistency Flow Matching / Inductive Moment Matching / Shortcut Models 这些"姊妹"方法?
- **整个技术路线**走偏了的可能:
  - 是否 pixel-space (JiT) + REPA-style alignment 是被我们错过的更优组合?
  - 是否应该放弃 latent diffusion,转向 autoregressive (VAR / MaskGIL / Markov-VAR)?
  - 是否 <1B 模型本质上不可能在 ImageNet 256 上达到 FID 1.0,我们的目标设定本身有误?

---

## 第三部分:我们已知的信息(请勿重复研究这些)

### 已读论文(按推荐度排序)

**理论核心(必读)**:
- **REPA** (Yu et al., ICLR'25 Oral, [arXiv: 2410.06940](https://arxiv.org/abs/2410.06940)) — DiT 浅层对齐 DINOv2,17.5× 加速收敛。**核心配置**:λ=0.5, alignment depth=8 (28% 深度), DINOv2-B, cosine similarity, NT-Xent 早期略优但最终持平
- **VA-VAE + LightningDiT** (Yao et al., CVPR'25, [arXiv: 2501.01423](https://arxiv.org/abs/2501.01423)) — VAE 训练时用 VF Loss 对齐 DINOv2 解决高维 latent 优化困境,FID 1.35
- **REPA-E** (Leng et al., ICCV'25, [arXiv: 2504.10483](https://arxiv.org/abs/2504.10483)) — VAE + DiT 端到端联合训练,REPA 作锚,FID 1.26
- **DDT** (Wang et al., CVPR'26 highlight, [arXiv: 2504.05741](https://arxiv.org/abs/2504.05741)) — Encoder-Decoder 解耦,FID 1.31 / 1.28 (256/512),encoder 共享推理加速
- **MeanFlow** (Geng et al., NeurIPS'25 Oral, [arXiv: 2505.13447](https://arxiv.org/abs/2505.13447)) — Average velocity training, 1-NFE FID 3.43,从头训。**关键超参**:25% r≠t, lognorm(-0.4, 1.0) time sampler, p=1.0 adaptive weight, (t, t-r) positional embedding
- **PAE** (Yue et al., 2026.05, [arXiv: 2605.07915](https://arxiv.org/abs/2605.07915)) — Prior-aligned AutoEncoder,三个 alignment 损失,gFID **1.03**

**相关参考**:
- **DiT** (Peebles & Xie, ICCV'23) — DiT 原文
- **SiT** (Ma et al., ECCV'24) — DiT + Flow Matching
- **SD3** (Esser et al., 2024) — MMDiT, logit-normal time sampling, rectified flow
- **SANA** (Xie et al., ICLR'25 Oral, [arXiv: 2410.10629](https://arxiv.org/abs/2410.10629)) — 0.6B + DC-AE + Linear Attention, 4K T2I
- **Rectified Diffusion** (Wang et al., ICLR'25, [arXiv](https://arxiv.org/abs/2410.07303)) — "straightness is not your need"
- **SplitMeanFlow** (ByteDance, [arXiv: 2507.16884](https://arxiv.org/abs/2507.16884)) — JVP-free MeanFlow via Interval Splitting Consistency
- **Understanding MeanFlow Training** ([arXiv: 2511.19065](https://arxiv.org/abs/2511.19065)) — 训练阶段分析,小间隔 → 大间隔 curriculum
- **JiT** (Li & He, 2025.11, [arXiv: 2511.13720](https://arxiv.org/abs/2511.13720)) — Pure ViT on pixels with x-prediction
- **PixelDiT** (NVlabs, CVPR'26 Oral, [arXiv: 2511.20645](https://arxiv.org/abs/2511.20645)) — Dual-level pixel-space DiT
- **DiP** (NJU + Tencent, [arXiv: 2511.18822](https://arxiv.org/abs/2511.18822)) — DiT + Patch Detailer Head, FID 1.79
- **VAR** (Tian et al., NeurIPS'24 Oral, [arXiv: 2404.02905](https://arxiv.org/abs/2404.02905)) — Next-scale prediction
- **MaskGIL** ([arXiv: 2507.13032](https://arxiv.org/abs/2507.13032)) — 改进的 MAR

**已经评估过但暂不采用**:
- **Z-Image (S3-DiT)** (Alibaba, 2025.11, [arXiv: 2511.22699](https://arxiv.org/abs/2511.22699)) — 6B 单流 DiT,主要为 T2I 设计,与我们的 <1B 目标不符
- **DiT-MoE / DiffMoE** — 稀疏 MoE,可叠加但增加复杂度
- **DC-AE** — 32× 深压缩,主要服务于 SANA 的 T2I,我们用 PAE-f16d32 更合适

### 已确定的工程配置

| 项 | 我们的选择 | 来源 |
|---|---|---|
| 数据集 | ImageNet-1K, 256×256 (主), 512×512 (次) | ADM evaluation protocol |
| Latent | PAE-f16d32 (32 × 32 × 32) | PAE 默认 |
| Backbone | DDT-L variant, ~459M params (估计) | DDT-L (20En/4De) 改造,具体配比待定 |
| Training objective | MeanFlow with 25% r≠t | MeanFlow 论文默认 |
| Optimizer | Adam, lr 1e-4, (β1, β2)=(0.9, 0.95), wd=0 | MeanFlow 默认 |
| EMA decay | 0.9999 | MeanFlow 默认 |
| Batch size | 256 (global) | MeanFlow 默认 |
| Precision | fp32 warmup 50K iter → bf16,JVP path 强制 fp32 | 我们的稳定性保险 |
| Eval | FID/sFID/IS/Pre/Rec via ADM 50K samples | 标准 |
| CFG | guidance interval + ω sweep | REPA-style scheduling |

### 已确认的算力

- 1× RTX Pro 6000 Blackwell (96GB GDDR7) 常驻
- 1× B300 (288GB HBM3e),Phase 3 (Week 8-10) 借用窗口
- 预计 87 H100-day equivalent 总预算
- 15 周时间

### 已规划的实验框架

- Phase 0-1: 4 个 baseline 复现 + 三组件独立验证
- Phase 2: 三对 pairwise 组合 (DDT×MeanFlow / PAE×MeanFlow / PAE×DDT)
- Phase 3: 三联合主训 + CFG sweep
- Phase 4: Proposition 1 三推论的实证 + DDT ratio 在 PAE latent 下的偏移测试
- Phase 5: 论文写作

---

## 第四部分:期望的输出格式

请对**每个探索区域**(区域 1-5)按以下结构输出。区域 6 (我们没想到的方向) 单独章节输出。

### 探索区域 X: [名称]

#### 发现的方案全景

列出所有路径——已知的(我们已经列出的)和新发现的(你研究后补充的),编号,一句话描述。例如:
- 方案 X.1: ... (我们已知)
- 方案 X.2: ... (我们已知)
- 方案 X.3: ... (新发现)
- 方案 X.4: ... (新发现,来自跨领域)

#### 方案详情

对每个方案(尤其新发现的),按以下结构展开:

##### 方案 X.N: [名称]

**一句话原理**:核心机制是什么?

**灵感来源**:来自哪个领域/论文/项目/代码库?(给具体引用 + arXiv ID + GitHub 链接)

**具体实现步骤**:
1. [步骤1:具体到可以开始编码,包含关键代码片段或伪代码]
2. [步骤2:...]
3. ...

**涉及组件**:PAE / DDT / MeanFlow / 训练循环 / 推理流程 中的哪些?

**与现有 PDM-3 系统的对接方式**:
- 需要修改哪些模块?
- 输入格式怎么变?
- 输出格式怎么变?
- 训练超参怎么变?

**难度评估**:[低/中/高] + 具体原因(实现复杂度、debug 难度、对现有 codebase 的侵入性)

**预期效果**:能解决问题到什么程度?有什么局限?如果可能,给出量化预期(预计 FID 改善幅度、训练加速倍数等)

**先例/参考实现**:
- 论文链接
- GitHub repo
- 博客文章
- 任何 reproduction notes

**风险与不确定性**:这个方案最可能在哪里翻车?需要先做什么 mini-experiment 验证?

### 技术顾问建议(每个探索区域结束时)

如果你是这个项目的技术顾问:
1. **推荐的 MVP 路径**(最快出结果)是什么?为什么?
2. **推荐的终极路径**(效果最好)是什么?为什么?
3. 建议的**实施顺序**是什么?(哪些方案应该先尝试,哪些应该作为 fallback)
4. 这个区域是否有**值得做但不该在这篇论文做**的方向?(给未来工作留 ticket)

### 区域 6 输出格式:意外发现

在研究过程中发现的、不属于上述 5 个探索区域的新方向。结构同上,但**特别强调**:

- 如果发现了**完全超出探索区域**的全新方向——类似于"不加大扇叶,而是重新设计风道"的灵感——请务必单独列出
- 如果发现整个技术路线的**潜在问题**(比如"PDM-3 这个方向本身可能是死胡同,因为 [某个我们没考虑到的根本性原因]"),请直接说
- 如果发现 PDM-3 与某个**完全不同的新范式**(比如 score-based perceptual models, energy-based generative models, neural compression-as-generation)之间有出人意料的联系,请详细展开

这类发现对我们来说价值最高。

### 最终整体建议

文档末尾请给出:

1. **基于本次研究,你对 PDM-3 整体方向的判断**:
   - 信心评级 (高/中/低) + 原因
   - 最大的 1-2 个潜在 dealbreaker
   - 你认为 reviewer 最可能 challenge 的点

2. **如果你只能给我们三条建议**,它们是什么?

3. **如果三个推论(Proposition 1 的实验验证)全部失败,你建议的 Plan B / Plan C 是什么?**

---

*本文档生成日期:2026-05-27*
*项目阶段:开题(Pre-Phase 0,环境尚未搭建)*
*预计开始执行:Week 1 (~2026-05-29)*
