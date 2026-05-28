# PDM-3-HT 三层级解耦扩散模型 · 研究任务书 (v2)

> **文档性质**:这是一份交给研究型 AI agent 的自包含研究任务书。阅读者无法访问本项目的本地文件或对话上下文,所有必要背景信息都已包含在本文档中。
>
> **研究主题**:PAE × Heterogeneous Timesteps × MeanFlow 三层级解耦扩散模型 (PDM-3-HT) 的实现路径与隐藏风险,**以及与 FD-loss post-training 的整合**
>
> **版本说明**:
> - **v1 (废弃)**:PDM-3 = PAE × **DDT** × MeanFlow。因 DDT 与 HT 设计冲突而废弃
> - **v2 (上一版)**:PDM-3-HT = PAE × **HT (Patch Forcing)** × MeanFlow。HT 实证收益更高(SiT-XL/2 上 FID 17.2 → 9.8, -43%)
> - **v2.1 (本版)**:在 v2 基础上增加新探索区域 6:**FD-loss (Representation Fréchet Loss, arXiv: 2604.28190, 2026.04) 与 PDM-3-HT 的整合策略**。FD-loss 把 ImageNet 256 1-NFE FID 推到 0.72,**显著重设了我们的目标 FID 基准**
>
> **使用方式**:把这份文档的完整内容粘贴给具备深度研究能力的 AI(Claude Deep Research / o1 Pro / Gemini Deep Research / 同类产品),让其按第四部分指定的格式输出报告。

---

## 第一部分:项目全景

### 1.1 项目目标

我们要训练一个**主干参数 < 1B、生成质量强劲的扩散图像生成模型**,这是一个硕士/博士开题级别的项目。

具体量化目标(已根据 FD-loss SOTA 调整):

| 评估项 | 目标值 | 当前最佳 SOTA(已知) |
|--------|--------|----------------------|
| ImageNet 256 FID (base model, no CFG) | ≤ 1.8 | REPA-E: 1.83 |
| ImageNet 256 FID (base model, w/ CFG + guidance interval) | ≤ 1.2 | PAE: 1.03 |
| ImageNet 256, 1-NFE FID (base model, no post-training) | ≤ 3.0 | MeanFlow-XL/2: 3.43 |
| **ImageNet 256 FID (1-NFE, with FD-loss post-training)** | **≤ 0.7** | **FD-loss: 0.72 (pMF-H post-trained)** |
| **FDr⁶ (multi-representation FID ratio, 新指标)** | **≤ 1.9** | **FD-loss: 1.89, validation ref: 1.0** |
| 主干参数量 | < 1B | SiT-XL/2: 675M, DiT-XL/2: 675M |

**关于 FDr⁶**:FD-loss 论文 (arXiv: 2604.28190) 指出单一 Inception-FID 不再可靠——SOTA 模型已经在 FID 上"超过" validation set,但视觉质量差距明显。FDr⁶ 是 6 个 representation 空间(Inception, SigLIP2, MAE, DINOv2, CLIP, RADIO)的标准化 Fréchet 距离平均,validation reference = 1.0,SOTA 仍在 1.89-3.0 区间。**本项目从一开始就将 FDr⁶ 作为主要评价指标之一**,以避免 reviewer 用"FID 已饱和"的理由 reject。

**为什么是 ImageNet 256 class-cond 而非 T2I**:我们的算力(Pro 6000 × 1 常驻 + B300 × 1 短时 burst, 约 90 H100-day 总预算)无法支撑 T2I 从零训练。ImageNet 256 是所有相关 SOTA 论文(REPA / VA-VAE / LightningDiT / REPA-E / DDT / MeanFlow / PAE / Patch Forcing / JiT / PixelDiT)的对比擂台,在这里发声影响力远高于半成品 T2I。

**项目预期 contributions**:

1. **Empirical**:在 <1B 参数尺度上对 ImageNet 256 class-cond 取得新 SOTA
   - Base model (no post-training):FID ≤ 1.2 w/ CFG
   - With FD-loss post-training:**1-NFE FID ≤ 0.7,FDr⁶ ≤ 1.9** (突破 FD-loss 论文的 0.72/1.89 基准)
2. **Theoretical**:
   - 推导并证明 **Per-Patch MeanFlow Identity** 及其训练目标的良定性
   - 证明 Proposition 1 —— PAE 的 Manifold Continuity Regularization (MCR) 损失对 MeanFlow JVP 数值稳定性的保证,推广到 per-patch setting
   - **可能的新 Proposition**:PAE-aligned latent + DINOv2-based FD-loss 的 representation 一致性(代价是双重 representation overfitting 的 risk)
3. **Methodological**:首次将 **四层级解耦** 完整组合
   - Latent geometry (PAE)
   - Spatial-time decoupling (HT/Patch Forcing)
   - Trajectory objective (MeanFlow)
   - Distribution-level post-training (FD-loss)
   - **实证验证四层级解耦的相互独立性与协同性**

**投稿目标**:NeurIPS 2026 (截止 ~5月) 或 ICLR 2027 (截止 ~9月)

### 1.2 技术架构详解

PDM-3-HT 由三个独立组件构成,作用在系统的**三个不同抽象层**:

```
原始图像 x ∈ R^(3×256×256)
   │
   ↓ [PAE Encoder]                ← 第一层解耦 (latent geometry)
                                    PAE = Prior-Aligned AutoEncoder
                                    用 frozen VFM (DINOv2) 作语义参照
                                    + DAM (Detail-Aware Modulator) 注入细节
                                    + 三个对齐损失 (SSR/MCR/SCR) 显式重塑流形
   ↓
Latent z₀ ∈ R^(32×32×32)          (PAE-f16d32 默认配置)
   │
   ↓ 加噪 (heterogeneous!)        ← 第二层解耦 (spatial-time granularity)
   ↓ 对每个 patch i 独立采样 tᵢ
   ↓ zₜᵢ^(i) = (1-tᵢ)·z₀^(i) + tᵢ·εᵢ
   ↓ tᵢ 服从 LTG sampler:
   ↓   1. 先采 t_max ~ LogitNormal(μ, σ)
   ↓   2. tᵢ ~ TruncGaussian(center=t_max, support=[0, t_max])
   ↓
[LightningDiT with Per-Token AdaLN]
   │  改造极简(基于 Patch Forcing 的 ~15 行 diff):
   │  - timestep embedding 从 (B, D) 变为 (B, N, D)
   │  - AdaLN 的 scale/shift/gate 从 (B, D) 变为 (B, N, D)
   │  - 每个 token 独立 conditioning on 它的 tᵢ
   │  - 额外输出 per-patch log-variance σᵢ² (difficulty head)
   ↓
训练目标:Per-Patch MeanFlow            ← 第三层解耦 (trajectory)
   │  对每个 patch i 独立:
   │  uᵢ_tgt = vᵢ - (tᵢ-rᵢ)·[vᵢ·∂_z uᵢ_θ + ∂_t uᵢ_θ]
   │  loss = ||uᵢ_θ - sg(uᵢ_tgt)||² + λ_var·GaussianNLL(vᵢ, σᵢ²)
   │  
   │  ⚠️ 关键 unknown: per-patch JVP 怎么算最高效?
   │  ⚠️ 关键 unknown: (rᵢ, tᵢ) 的联合采样策略?
   ↓
推理: 多种策略
   - 1-NFE 全局: 所有 patch 一步, xᵢ = εᵢ - uᵢ_θ(εᵢ, 0, 1)
   - Adaptive: 易 patch 1-NFE,难 patch 2-3 NFE
   - Look-ahead: 用 confident patches 作为 context refine uncertain patches
```

**三个组件各自的技术核心**:

**PAE (Prior-Aligned AutoEncoder)** ([arXiv: 2605.07915](https://arxiv.org/abs/2605.07915), GitHub: `ZhengrongYue/PAE`):
- 重建保真度 rFID 与生成质量 gFID 相关性弱;真正决定 gFID 的是 latent manifold 的三个几何性质:
  - **SSC (Spatial Structure Coherence)**:邻近 patch 的 latent 空间结构一致
  - **LPC (Local Manifold Continuity)**:latent 平滑,小扰动 → 小变化
  - **GSQ (Global Manifold Semantics)**:latent 按语义聚类
- 三个对齐损失:SSR (spatial structure regularization)、MCR (manifold continuity regularization)、SCR (semantic consistency regularization)
- 报告结果:ImageNet 256 gFID **1.03** (LightningDiT-XL/1, SOTA)

**Patch Forcing / Heterogeneous Timesteps** ([arXiv: 2604.19141](https://arxiv.org/abs/2604.19141), CVPR 2026, GitHub: `CompVis/patch-forcing`, CompVis @ LMU Munich, Björn Ommer 组):
- 核心:训练时每个 patch 独立的 timestep,推理时 adaptive
- **LTG sampler (Logit-Normal Truncated Gaussian)**:控制每个样本的 maximum information per sample(而不是 average information,SRM 的失败教训),避免 training-test distribution mismatch
- **Per-patch difficulty head**:用 Gaussian NLL 训出 per-patch log-variance σᵢ²,作为 uncertainty proxy
- **Adaptive sampling**:Dual-loop(交替推进 confident 与 refine uncertain)、Look-ahead(用 confident 作为 context)
- 报告结果:SiT-XL/2 baseline FID 17.2 → Patch Forcing + Look-ahead 9.8 (-43%)
- 关键架构改造极小,只是 AdaLN 从 broadcast 改成 per-token

**MeanFlow** ([arXiv: 2505.13447](https://arxiv.org/abs/2505.13447), NeurIPS 2025 Oral, Kaiming He / J. Zico Kolter):
- 核心:学**平均速度** $u(z_t, r, t) := \frac{1}{t-r}\int_r^t v(z_\tau, \tau)d\tau$ 而非瞬时速度
- 训练目标(MeanFlow Identity):
  $$u_{tgt} = v(z_t, t) - (t-r) \cdot \frac{du}{dt}$$
- 用 JVP (`torch.func.jvp`) 计算 `du/dt`,只多一次 backward(~16% overhead)
- 25% 样本采 r≠t,75% 样本采 r=t (退化为 Flow Matching)
- 推理:1-NFE 单步,$x = \epsilon - u_\theta(\epsilon, 0, 1)$
- 报告结果:ImageNet 256 1-NFE FID **3.43**,从头训,无蒸馏无 curriculum

**架构上的关键创新点(我们做的)**:

把上述三者完整组合。关键工程/数学创新:

1. **Per-Patch MeanFlow Identity 的推广**(数学):
   - 当前 MeanFlow:$(r, t)$ 是 scalar,定义一段全空间共享的轨迹
   - PDM-3-HT:$(r_i, t_i)$ 是 per-patch 张量,每个 patch 沿自己的轨迹
   - Identity 推广: $u_i(z_t, r_i, t_i) = v_i(z_t, t_i) - (t_i - r_i) \cdot \partial_t u_i$
   - 但注意 $u_i$ 通过 attention 依赖**所有其他 patches** 的 $z^{(j)}$,**链式法则比 global 情况复杂**

2. **JVP-friendly batch organization**(工程):
   - per-patch JVP 不能简单地"对每个 patch 单独 JVP"——成本会乘 N (token 数)
   - 需要 batched JVP:把 N 个 patch 的 (r, t, v) 打包,一次 JVP 同时给出所有 N 个 $du_i/dt_i$
   - PyTorch `torch.func.jvp` 原生支持 batched JVP,但数值精度在 bf16 下需要验证

3. **(r, t) per-patch 联合采样**(算法):
   - 当前 MeanFlow:(r, t) ~ lognorm(-0.4, 1.0),global
   - PDM-3-HT:t_i 来自 Patch Forcing 的 LTG,r_i 应该怎么采?
   - 候选:r_i = α·t_i (α ~ U(0, 1)),保持 r_i < t_i 的约束
   - 候选:r_i 独立采 LTG,但需要保证 r_i ≤ t_i

### 1.3 走过的路(为什么选 Path C / PDM-3-HT)

**第一轮迭代:PDM-3 (v1)**:
- 原 v1 方案 = PAE × **DDT** × MeanFlow
- DDT (Decoupled Diffusion Transformer, CVPR'26 highlight) 把 DiT 拆成 encoder/decoder
- 故事:三层级解耦 (latent / architecture / objective)
- 风险:DDT × MeanFlow 的 encoder/decoder 重设计、JVP 链式法则复杂度

**第二轮迭代:发现 Patch Forcing 的实证优势**:
- Patch Forcing (CVPR'26 CompVis @ LMU) 在 SiT-XL/2 上拿到 -43% FID(相对 vanilla SiT baseline)
- 同等 settings 下,DDT 大约 -30%
- HT 的**实证收益更高**,且**架构改造更小**(per-token AdaLN 改 15 行 vs DDT 的 encoder/decoder 拆分)

**第三轮迭代:DDT 与 HT 的设计冲突**:
- DDT:encoder 抽**全局语义**(low-freq, semantic),decoder 出**局部细节**(high-freq, velocity)
- HT:每个 patch 有自己的 t,自己的 denoising trajectory
- 合并后:encoder 应该看 global 还是 per-patch 的 condition? 这是个**没人解决过**的设计冲突
- **结论**:DDT 和 HT 在同一份模型里"打架";HT 已经提供了"按需处理"的空间维度,DDT 的深度维度处理 marginal value 大幅降低

**第四轮迭代:转向 Path C / PDM-3-HT**:
- 把 DDT 砍掉,改用 LightningDiT(成熟、生态完整) + 加 per-token AdaLN
- 故事重写为:**"三层级解耦"——latent (PAE), spatial-time (HT), trajectory (MeanFlow)**
- 工程难度大幅降低,理论 contribution 仍然丰富(per-patch MeanFlow Identity + Proposition 1)

**Proposition 1 (核心理论 nugget,从 v1 继承并推广)**:

原版 Proposition 1:PAE 的 MCR 损失约束 decoder 的 Lipschitz 常数 $L_D$ → 隐式约束 latent 分布 $p_z$ 的几何条件数 $\kappa(p_z)$ → score 函数 $\nabla \log p_t(z_t)$ 的 Lipschitz 性 → velocity field $v(z_t, t)$ 的 Lipschitz 性 → MeanFlow 训练中 JVP 项 $\|du/dt\|$ 有界 → 训练目标方差有界 → SGD 数值稳定。

**Per-patch 推广 (待研究)**:在 HT setting 下,$z_t$ 的不同 patch 在不同 noise level,velocity field $v_i(z_t, t_i)$ 同时依赖 $t_i$ 和所有其他 patch 的 noise levels(通过 attention)。是否仍然存在 Lipschitz 上界?这是开放问题。

**三个可证伪的实验预测(从 v1 继承)**:
- **推论 1**:同一 MeanFlow 训练下,PAE tokenizer 的 JVP norm 分布显著低于 SD-VAE / VA-VAE
- **推论 2**:同一 MeanFlow 训练下,PAE 配置中 NaN 出现频率显著低于 SD-VAE / VA-VAE(尤其在 bf16 下)
- **推论 3**:PAE 下允许的最大稳定学习率高于 SD-VAE / VA-VAE

如果**三个推论都验证**,Proposition 1 落地;**至少有 2 个验证**,论文有完整的理论-实验闭环。

### 1.4 计算预算与时间线

**算力**:
- **Pro 6000 (Blackwell, 96GB)** × 1,常驻,bf16 训练对标 H100 的 ~70-80%
- **B300 (288GB HBM3e)** × 1,**仅可短时借用**,fp8 算力对标 H100 的 ~5×
- 折合总预算约 **87-90 H100-day**

**时间线**(16-18 周,加入 Phase 3.5 FD-loss post-training):

| Phase | 周次 | 算力 | 关键目标 |
|-------|------|------|----------|
| 0: 环境 + Baseline | 1-2 | Pro 6000 | 复现 6 个 baseline (SiT, LightningDiT+PAE, MeanFlow, Patch Forcing, LightningDiT+PAE+REPA-E, **FD-loss post-training**) |
| 1: 三组件独立集成 | 3-5 | Pro 6000 | PAE / HT / MeanFlow 各自验证 |
| 2: Per-Patch MeanFlow 推导与实现 | 6-7 | Pro 6000 | per-patch MeanFlow Identity 公式 + 工程实现 |
| 3: 三联合主训 | 8-11 | **B300 burst** | 冲 base FID ≤ 1.2 |
| **3.5: FD-loss post-training (新增)** | **12-13** | **Pro 6000** | **冲 1-NFE FID ≤ 0.7, FDr⁶ ≤ 1.9** |
| 4: 理论 ablation | 14-15 | Pro 6000 | Proposition 1 三推论 + per-patch JVP 稳定性 + FD-loss 整合 ablation |
| 5: 论文写作 | 16-18 | — | 投稿 |

**Phase 3 是 B300 的主要使用窗口,Phase 3.5 在 Pro 6000 上即可完成**(FD-loss post-training 不需要从头训,只 fine-tune ~10K-50K iter)。

### 1.5 PDM-3-HT 三组件的预设角色与未知点

| 阶段 | PAE 的作用 | HT (Patch Forcing) 的作用 | MeanFlow 的作用 | 我们不了解的部分 |
|------|----------|----------|----------|------------|
| **数据→latent** | 用 VFM 重塑流形,SSR/MCR/SCR 训练 | — | — | PAE-f16d32 vs f16d16 vs f16d64 在 HT × MeanFlow 下的偏好 |
| **加噪** | — | 每个 patch 独立 t,LTG sampler | — | LTG 在 PAE latent 上的最优 (μ, σ) 参数?在 latent 而非 RGB 上 LTG 的"最大信息控制"还成立吗? |
| **网络前向** | — | Per-token AdaLN,patch_i 的 hidden 看自己的 t_i | — | Per-token AdaLN 的额外参数量(可忽略)、attention pattern 在 spatially-heterogeneous noise 下的行为 |
| **训练目标** | 三个对齐损失,frozen | difficulty head: GaussianNLL with σᵢ² | Per-patch MeanFlow Identity | (rᵢ, tᵢ) 联合采样策略?λ_var 权重?difficulty head 是否要 fade-in? |
| **JVP 计算** | — | per-patch | per-patch | Batched JVP 在 bf16 下的数值稳定性 |
| **推理** | — | Dual-loop / Look-ahead | 1-NFE / few-NFE | 1-NFE × spatially heterogeneous 这个组合的最优策略 |

**关键不确定性总结(我们需要外部研究帮我们看清的部分)**:

1. **Per-Patch MeanFlow Identity 的精确数学形式与算法实现**
2. **JVP 在 per-patch + bf16 + PAE-MCR 下的实证稳定性**
3. **LTG sampler 在 PAE latent space 的最优参数化**
4. **MeanFlow 的 (r,t) 联合采样,在 per-patch 下的最优策略**
5. **1-NFE 推理 vs Look-ahead/Dual-loop 多步推理在 HT 下的对比**
6. **Proposition 1 的 per-patch 扩展是否成立,以及如何严格陈述**

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
> 3. **特别关注"跨领域借鉴"**:神经 ODE / Diffrax、最优控制、隐式微分、partial differential equation (PDE) 数值求解、计算流体力学 (CFD) 的非均匀网格方法、自适应有限元、attention sparsity、稀疏正则化、多任务学习的 gradient surgery、视频/序列扩散模型——这些邻近领域有没有解决过类似问题?
> 4. 对每条发现的路径,**给出具体到可以开始编码的实现方案**
>
> 我们是扩散模型的应用研究者,不是数值分析、最优控制或几何深度学习专家。我们能想到的方案大概率是最表面的几种。请像一个**同时精通扩散模型、神经 ODE、自动微分、PDE 数值方法的资深专家**那样思考——**你觉得理所当然的东西,对我们来说可能是全新的发现。**

### 探索区域 1: Per-Patch MeanFlow Identity 的精确推导与算法实现

**困境**:

标准 MeanFlow 的核心方程是:
$$u(z_t, r, t) := \frac{1}{t-r}\int_r^t v(z_\tau, \tau)\,d\tau$$

由微分得到 MeanFlow Identity:
$$u(z_t, r, t) = v(z_t, t) - (t-r) \cdot \frac{du}{dt}$$

但这一切都假设 $(r, t)$ 是 **scalar**,定义**一段全空间共享的轨迹**。

**在 Patch Forcing × MeanFlow 联合 setting 下**,每个 patch i 有自己的 $(r_i, t_i)$。这意味着:
1. 整个 latent $z_t$ 的每个 patch 处于不同 noise level
2. 每个 patch 沿**自己的轨迹**演化:$dz_t^{(i)}/dt_i = v_i(z_t, t_i)$,**注意 $v_i$ 依赖整个 $z_t$,因为 attention 让 patches 互相耦合**
3. 整体不是一条轨迹,而是 **N 条相互纠缠的轨迹的束 (bundle)**

**关键数学问题**:Per-patch average velocity 应该怎么定义?

候选 1:每个 patch 独立做 1D 积分
$$u_i(z_t, r_i, t_i) := \frac{1}{t_i - r_i}\int_{r_i}^{t_i} v_i(z_\tau, \tau_i)\,d\tau_i$$
**问题**:在积分中 $z_\tau$ 是什么?其他 patch 也在演化,$z_\tau$ 是**整个束**在某个时刻的状态——但每个 patch 的时刻不同步,"$\tau$" 是哪个 patch 的 $\tau$?

候选 2:用 "global 进度参数" $s \in [0, 1]$ 重新参数化
$$t_i(s) = r_i + s \cdot (t_i - r_i), \quad u_i(z_{t(s)}, 0, 1) := \int_0^1 v_i\,ds$$
**问题**:这种参数化下,所有 patches 同时从 $r_i$ 走到 $t_i$,但它们各自的速度不一样——是否还满足 ODE flow 的性质?

**我们这个外行能想到的做法**:

> 1. **最朴素**:沿用候选 1,$z_\tau$ 在积分时假设其他 patch 也"等比例"演化
> 2. **最朴素**:把 per-patch MeanFlow Identity 直接写成 $u_i = v_i - (t_i - r_i) \cdot du_i/dt_i$,只对 $t_i$ 求偏导,忽略与其他 patch 的耦合
> 3. **最朴素**:每个 patch 独立 JVP,共 N 次 JVP(太贵)

**但我们怀疑这只是冰山一角。**

请自由探索:

- **数学严格性**:Per-Patch MeanFlow Identity 的**严格**形式应该是什么?有没有可能 Identity 本身就需要包含 cross-patch 项(因为 attention 耦合)?如果是,新的 Identity 长什么样?
- **数值积分**:神经 ODE 社区对"耦合的 N 个 ODE"的处理方法,有借鉴价值吗?(adjoint method? checkpointing? coupled implicit solvers?)
- **物理类比**:**多体问题** (N-body problem) 数值积分有非常成熟的方法 (symplectic integrators, Yoshida 高阶分解, Hamiltonian splitting),这些技术能不能搬来?
- **PDE 数值类比**:CFD 里**非均匀时间步长** (asynchronous time stepping) 有成熟方法,这条线对吗?
- **JVP 的 batched 实现**:`torch.func.jvp` 能不能一次给出所有 N 个 patches 的 $du_i/dt_i$?具体怎么写?bf16 下数值稳定性如何?
- **简化假设**:如果 cross-patch 耦合在数学上太难,有没有合理的**simplification** 让我们能 close form 推导?比如 "忽略 attention 的二阶项" 这种近似?
- **替代目标**:有没有完全不需要 MeanFlow Identity 但能达到同样 1-NFE 效果的训练目标?(IMM, Shortcut, Consistency Flow Matching 的 per-patch 推广?)

---

### 探索区域 2: LTG Sampler 在 PAE Latent Space 的最优形式

**困境**:

Patch Forcing 的 LTG (Logit-Normal Truncated Gaussian) sampler 是这样工作的:
1. 每个样本采一个 $t_{\max} \sim \text{LogitNormal}(\mu, \sigma)$
2. 每个 patch 的 $t_i \sim \text{TruncGaussian}(\text{center}=t_{\max}, \text{support}=[0, t_{\max}])$
3. 关键论点:控制 **maximum information per sample**(而不是 average)避免 training-test mismatch

但 Patch Forcing 论文的实验在 **SD-VAE latent + ImageNet 256/512** 上调出来的 (μ, σ) 是否就是 PAE latent 上的最优?

**PAE latent 和 SD-VAE latent 有几个本质不同**:
1. PAE latent 是 32 维通道 (vs SD-VAE 4 维),信息密度高得多
2. PAE latent 有显式的 manifold 几何约束 (MCR),平滑性更好
3. PAE latent 的"语义信息密度"分布可能更均匀(因为 SSR 强迫空间结构一致)

这些差异意味着 **LTG 的最优参数大概率不一样**——可能 PAE latent 上 max-controlled 不再是最优,可能 mean-controlled 反而更好(因为 PAE 已经"平均化"了信息)。

**我们这个外行能想到的做法**:

> 1. 直接照搬 Patch Forcing 的 (μ, σ) 默认值
> 2. 在 PAE 上做一个简单的 grid search,sweep (μ, σ) 的几个组合
> 3. 完全替换成 logit-normal,放弃 LTG 的 truncation

**但我们怀疑这只是冰山一角。**

请自由探索:

- **理论分析**:LTG 在 latent space 的"信息控制"应该是什么?**latent space 的"information per sample" 怎么严格定义**?(是 Shannon entropy of denoised distribution? Fisher information? KL to prior?)
- **PAE-aware sampler**:能不能基于 PAE 的三个 manifold 性质 (SSC/LPC/GSQ) 设计**新的 sampler**?比如根据 latent 的 local manifold curvature 决定 t_i?
- **跨领域**:**Quasi-Monte Carlo** 方法里有非常成熟的"如何在高维空间均匀采样"理论(Sobol, Halton sequences)。能不能借鉴到 (t_1, ..., t_N) 联合采样?
- **跨领域**:**Adaptive importance sampling** 在 MCMC 里的工作,能不能借来设计 online-adapted (μ, σ)?
- **联合优化**:Sampler 参数 (μ, σ) 是否应该作为可学习参数,和网络一起训?有没有可能 sampler 本身也有"梯度回传"?
- **新统计量**:Max 和 Average 之外,还有什么 statistic 可以控制?比如 **median** (鲁棒)、**quantile** (灵活)、甚至 **moment matching** (强约束)?

---

### 探索区域 3: Per-patch Difficulty Head 与 PAE 语义先验的协同

**困境**:

Patch Forcing 引入的 difficulty head 是这样工作的:
- 模型额外输出 per-patch log-variance $\sigma_i^2$
- 用 Gaussian NLL 训:$\mathcal{L} = \|v_{GT} - v_\theta\|^2 - \lambda \log \mathcal{N}(v_{GT} | \text{sg}(v_\theta), \sigma_\theta^2 I)$
- 直觉:误差大的 patch → 高 variance → 推理时多分配 compute

但 **PAE 的 latent 已经隐含了 patch-level "difficulty" 信号**:
- PAE 的 SSR (Spatial Structure Regularization) 损失对每个 patch 计算
- 高 SSR 残差的 patch = 与 VFM 对齐困难 = 语义复杂 = 可能就是难 patch
- 我们已经免费拥有一个"PAE 视角的 difficulty proxy",**但 Patch Forcing 没用上**

**关键问题**:能不能用 PAE 的语义信号 **bootstrap** difficulty head?或者完全替代 difficulty head?

**我们这个外行能想到的做法**:

> 1. 直接用 PAE 的 per-patch SSR 残差作为初始 difficulty 估计
> 2. 加 difficulty head 作为 SSR 之外的 refinement
> 3. 完全跳过 difficulty head,用 SSR 残差直接做推理调度

**但我们怀疑这只是冰山一角。**

请自由探索:

- **多模态 difficulty 估计**:可能有**多个**互补的 difficulty signals:PAE SSR 残差、PAE MCR 残差、image 频谱能量、attention entropy、velocity norm。**哪些组合最 informative**?是否可以用 ensemble?
- **训练时 vs 推理时**:Difficulty head 在训练时和推理时的角色是不同的吗?训练时它影响 loss,推理时它影响 schedule——能不能**两阶段用不同的 difficulty estimator**?
- **跨领域 (active learning)**:Active learning 里有大量关于"哪个 sample 难"的研究。**Coreset 选择、不确定性估计 (BALD, ALC) **的方法能不能转化到 patch-level?
- **跨领域 (uncertainty quantification)**:**Bayesian deep learning** 里 deep ensembles, MC dropout, evidential deep learning ——这些 UQ 方法在 diffusion patch difficulty 估计上的应用?
- **零成本 difficulty**:能不能完全**不训** difficulty head,直接用现成信号(比如 PAE residual)?这能省训练 compute,但损失多少质量?
- **基于物理的 difficulty**:某些 patch 难,是因为它们的 latent **更接近 manifold boundary** (PAE 的 LPC 残差);某些是因为它们 **跨多个语义类别** (PAE 的 SCR 残差)。难度有不同种类,**用单个 σ² 表达是否过度简化**?
- **Cross-patch difficulty**:一个 patch 的难度可能依赖于**其相邻 patch 的状态**——上下文越复杂,该 patch 越难。这种 cross-patch 信号怎么建模?

---

### 探索区域 4: 1-NFE 推理与 Patch Forcing Adaptive Sampling 的整合

**困境**:

Patch Forcing 和 MeanFlow 各自的推理策略**截然不同**:

| 方法 | 推理策略 | 总 NFE |
|------|---------|--------|
| Patch Forcing + Euler | Parallel Euler 每 patch | ~50 |
| Patch Forcing + Dual-loop | 交替推进 confident / refine uncertain | ~30-50 |
| Patch Forcing + Look-ahead | confident patches 作为 context | ~30-50 |
| MeanFlow | 1-NFE 全部 patches 一起 | **1** |

合并后,**最优策略应该是什么**?

候选 1: **完全 1-NFE**——所有 patch 一起一步生成
- 优点:速度最快,符合 MeanFlow 主卖点
- 缺点:放弃了 Patch Forcing 的 adaptive context propagation 优势

候选 2: **Adaptive few-NFE**——简单 patch 1-NFE,难 patch 2-3 NFE
- 优点:利用了 difficulty head,平衡速度和质量
- 缺点:NFE 不再统一,batch 处理变复杂

候选 3: **Look-ahead in 1-NFE**——所有 patch 一步生成,但简单 patch 的 1-NFE 结果作为"context"喂给难 patch 的 1-NFE 重算
- 优点:仍然是"1-NFE per patch",但 confidence propagation 利用了
- 缺点:技术细节(怎么 propagate? 用什么作为 confidence?)

**我们这个外行能想到的做法**:

> 1. 默认 1-NFE,作为 baseline
> 2. 加一个 "difficulty-conditional NFE" rule (e.g., σ_i > τ → 2 NFE, else 1 NFE)
> 3. 单独评估 1-NFE / 2-NFE / 4-NFE / multi-NFE 的 FID,看 quality-speed Pareto

**但我们怀疑这只是冰山一角。**

请自由探索:

- **MeanFlow 多步推理**:虽然 MeanFlow 主推 1-NFE,但它也支持多步 ($z_r = z_t - (t-r)·u_\theta(z_t, r, t)$)。多步采样可以让每个 patch 独立选择何时"停止" (early-exit)——这种 per-patch early-exit 在 MeanFlow 框架下的最优策略?
- **混合 NFE**:某些 patch 1-NFE 直接到 $t=0$,某些 patch 走 ($t \to t' \to 0$) 两步——这种"异步推理"和 Look-ahead 怎么结合?
- **跨领域 (computational fluid dynamics)**:CFD 里有 **adaptive mesh refinement** 和 **non-uniform time stepping**——根据局部 PDE error 决定哪里需要更多 compute。这条线对吗?
- **跨领域 (rendering)**:**Real-time rendering** 里 LOD (level-of-detail) 系统 —— 远景低 detail,近景高 detail。**等价于 patch 的"detail budget" 不均匀**。能不能转化为推理时 patch-NFE 分配?
- **学习一个 NFE controller**:能不能用 RL 训一个 controller,根据当前 $\sigma_i^2$ 和 schedule budget 动态分配 NFE per patch?(类似 TPDM 的思路)
- **1-NFE 本质上是 ODE Euler 的近似**:1-NFE per patch 是个**很糟糕的近似** for 困难 patch。能不能在 1-NFE 框架内增加 patch-specific "correction term" (Richardson extrapolation? Adams-Bashforth 1.5-NFE)?
- **Distillation 视角**:能不能从 Patch Forcing + multi-step 模型,**蒸馏**到 1-NFE 模型?(类似 Diffusion ODE distillation)

---

### 探索区域 5: Proposition 1 的 Per-Patch 推广与 JVP 稳定性

**困境**:

原 Proposition 1 (PAE-MCR ⇒ MeanFlow JVP 数值稳定性) 的论证链是:
1. MCR 损失 ⇒ decoder $D$ 是 $L_D$-Lipschitz
2. ⇒ latent 分布 $p_z$ 的几何条件数 $\kappa(p_z)$ 有界
3. ⇒ score function $\nabla \log p_t(z_t)$ Lipschitz
4. ⇒ velocity field $v(z_t, t)$ Lipschitz
5. ⇒ MeanFlow JVP $\|du/dt\|$ 有界
6. ⇒ 训练目标方差有界

**在 per-patch setting 下,这条链的每一环都需要重新审视**:

- 第 3 环:score function 现在是 $\nabla \log p_{\{t_i\}}(z_t)$,**联合密度** $p_{\{t_i\}}$ 是 multi-time 边际,Lipschitz 性更复杂
- 第 4 环:velocity field $v_i(z_t, t_i)$ 依赖**整个束** $z_t$ 和所有 $t_j$——cross-patch interaction 是否破坏 Lipschitz?
- 第 5 环:JVP 是 per-patch 的,$\|du_i/dt_i\|$ 有界吗?

**最大的不确定性**:**当 patches 通过 attention 强耦合时,某个 patch 的 noise level 突变会传播到所有 patches 的 velocity 估计**——这种"非局部"性质会不会破坏 Lipschitz 上界?

**我们这个外行能想到的做法**:

> 1. 直接套用 v1 的 Proposition 1 证明,假设 per-patch 完全独立(忽略 attention 耦合)
> 2. 加一个 "attention contribution" 项,argue 它在 expectation 下是 bounded
> 3. 数值实验验证:不证 theorem,直接测 JVP norm 在 PDM-3-HT 下的统计量

**但我们怀疑这只是冰山一角。**

请自由探索:

- **严格数学**:per-patch Proposition 1 的**严格陈述**应该是什么?需要哪些额外假设?(比如 "attention 是 Lipschitz 算子")
- **反例构造**:能不能构造一个**满足 MCR 但 per-patch JVP 不稳定**的反例?(比如某种病理的 cross-patch attention pattern)
- **跨领域 (PDE)**:**PDE 数值稳定性**理论里有"non-local term ⇒ instability"的丰富研究 (e.g., Cahn-Hilliard, fractional Laplacian)。这条线能借鉴吗?
- **跨领域 (Random matrix theory)**:Attention 的 Jacobian 是个 random matrix,**它的 spectral radius** 有什么已知 bound?(self-attention 的 Lipschitz 性已经有研究——Kim et al. 2021)
- **数值实验设计**:验证 "per-patch JVP 在 PDM-3-HT 下稳定"需要测什么具体的量?**怎么知道我们的实证是否能 generalize**?
- **如果三推论失败**:从 v1 继承的三个推论(JVP norm / NaN frequency / max LR)在 per-patch 下应该怎么调整?有没有更适合 per-patch 的诊断指标?
- **Fallback 路径**:如果 per-patch JVP 不可救药地不稳定(很有可能,attention 耦合非常 strong),整个 PDM-3-HT 的可行性受影响吗?有没有 "稳健 fallback" (比如 stop-gradient cross-patch attention 在 JVP 计算时)?

---

### 探索区域 6: FD-loss (Representation Fréchet Loss) 与 PDM-3-HT 的整合策略

**困境**:

2026.04 出现了一篇可能颠覆我们路线的论文:**Representation Fréchet Loss for Visual Generation** (Yang et al., arXiv: 2604.28190, USC + CMU + CUHK + OpenAI)。

**这篇论文的核心 idea(简单到反直觉)**:
- FID 一直以来只是评估指标,因为算 FID 需要 50k 大 population,无法用作训练 loss
- **关键洞察**:**解耦 population size 和 batch size** —— population 用 queue 或 EMA 维护(50k),gradient 只通过当前 batch (1024)
- 真实数据只用一次,offline 算 reference statistics(均值 + 协方差),训练时不需要真实图像

**惊人的实证结果**:
1. **Post-training 任意 base generator**:用 FD-loss fine-tune,ImageNet 256 一步生成 **FID 0.72**(目前所有公开数字中最低)
2. **能把 multi-step generator 转成 1-step generator,无需 distillation/adversarial/per-sample target**:JiT-H 强行 1-NFE 推理 base FID ~300,FD-loss post-train 后直接到 0.72
3. **暴露 FID 本身的可靠性问题**:不同 representation backbone (Inception/SigLIP2/MAE/DINOv2) 下 FID 排序和人眼视觉排序不一致,论文提出新指标 **FDr⁶**(6 个 representation 标准化平均)

**对 PDM-3-HT 的根本性影响**:

| 维度 | v2 原计划 | 受 FD-loss 影响后 |
|------|----------|--------------------|
| 目标 FID | 1.2 | **0.7**(否则不够 ambitious) |
| Per-patch MeanFlow Identity 的必要性 | 核心理论贡献 | **可能可以绕过**(用 multi-step + FD-loss repurpose) |
| MeanFlow JVP 的复杂性 | 需要严格处理 | **可能完全规避** |
| 主要评价指标 | FID (single) | **FID + FDr⁶ (multi-rep)** |
| 工程难度 | 中-高 | 取决于整合策略,可能大幅降低 |

**三种可能的整合策略**(我们能想到的):

> **策略 A — 简单叠加(保守)**:
> - Phase 0-3:仍走 PDM-3-HT (PAE × HT × MeanFlow),训出 1-NFE base model
> - Phase 3.5 新增:用 FD-loss post-train ~10K iter,把 1-NFE FID 从 1.2 推到 0.7
> - 几乎零额外风险,只多 1-2 周
>
> **策略 B — 绕过 MeanFlow(中度激进)**:
> - Phase 0-3:PAE × HT × 标准 Flow Matching (multi-step) → multi-NFE base model (FID 1.5)
> - Phase 3.5:FD-loss post-train repurpose 到 1-NFE (FID 0.7)
> - 砍掉 per-patch MeanFlow Identity 的所有数学复杂性
> - 失去一个理论 contribution,获得工程简洁性
>
> **策略 C — 双轨制(完整版)**:
> - Phase 3a:PAE × HT × MeanFlow → 1-NFE generator A
> - Phase 3b:PAE × HT × Flow Matching → multi-NFE generator B
> - Phase 4:两条都用 FD-loss post-train,实证对比
> - 工作量最大,但 contribution 最丰富,适合冲 oral

**但我们怀疑这只是冰山一角。**

请自由探索:

- **FD-loss 的失败模式**:FD-loss 论文 Appendix A 提到 "representation-coupled reward hacking" —— 模型可能 game 特定 representation 的 FD 而损害其他 representation 下的视觉质量。这对 PAE 设置下的影响?如果 PAE 用 DINOv2-aligned latent,FD-loss 也用 DINOv2 representation,**是否双重 representation 锁定导致严重 overfit**?
- **Queue vs EMA estimator 的优劣**:论文同时给出 queue-based 和 EMA-based 两种实现。在 PDM-3-HT 下哪种更合适?queue 的 memory 占用 (50k × feature_dim,大约 50M 浮点) 在我们的 96GB Pro 6000 上可以接受,但 EMA 可能数值上更稳定。
- **Representation 选择的优化问题**:FD-loss 论文用了 6 个 representation 做 FDr⁶。**最优的 representation 组合是什么?**是否应该和 PAE 的 VFM 保持一致,还是有意"错开"以避免 representation 锁定?
- **Curriculum:何时启用 FD-loss?**
  - 选项 1:训完再 post-train(论文做法)
  - 选项 2:训练中后期 joint(类似 GAN 的逐步加入 adversarial loss)
  - 选项 3:从训练开始就 joint(可能不稳定,因为早期 generator 输出离 reference 分布太远,FD gradient 过大)
- **Per-patch FD-loss?** Patch Forcing 让每个 patch 有自己的 trajectory。**能否对每个 patch 算单独的 FD?** 即把 patches 当 "small images" 算 patch-level FD?这种 per-patch distributional matching 对 spatially heterogeneous 信号是否更适合?
- **FD-loss × Difficulty Head 的协同**:Patch Forcing 的 difficulty head 输出 per-patch σ²。FD-loss 提供 per-batch distributional signal。**两个 uncertainty signal 能否融合**?(比如把 σ² 大的 patch 在 FD-loss 中加权更高)
- **跨领域 (GAN training)**:GAN 训练里类似"distributional matching"的工作非常多 (W-GAN, MMD-GAN, sliced Wasserstein)。**这些工作的失败模式 (mode collapse, instability) 在 FD-loss 中是否会重现**?如何防范?
- **跨领域 (Implicit Maximum Likelihood Estimation, IMLE)**:IMLE 系列工作也是 distribution matching 但 batch-wise。**IMLE 的 nearest-neighbor 匹配思想能否补充 FD-loss 的 moment-matching**?
- **跨领域 (Optimal Transport)**:**Sliced OT distance 是否比 Fréchet distance 更鲁棒**?(已有工作用 sliced Wasserstein 训 GAN)
- **元问题:FD-loss 真的"安全"吗?**
  - 如果整个学术界都开始用 FD-loss,**会不会出现"集体 game representation"的现象**?
  - 即 generation 看起来 FID/FDr⁶ 漂亮,但视觉质量本质上没进步,只是更善于 fool 这些 representation 模型?
  - 这是不是一种新形式的 "representation collapse"?
- **元问题:对评测的反噬**
  - 我们用 FD-loss 训出 FID 0.7 的模型,**reviewer 怎么知道这不是"过拟合 evaluation"**?
  - 是否需要在论文中报告**完全 hold-out 的 representation**(未参与 FD-loss 训练) 下的 FID,作为 sanity check?

---

### 探索区域 7: 我们没想到的方向

以上 6 个区域是我们这些外行能划定的范围。**但几乎可以肯定,还有我们完全不知道的方向存在。**

请特别关注:

- **最近 6 个月发表的相关新论文** (2026 上半年):
  - Patch Forcing 是 2026 CVPR(已知),Self-Flow (arXiv: 2603.06507) 是 concurrent
  - MeanFlow 的 follow-up:SplitMeanFlow (arXiv: 2507.16884)、Understanding MeanFlow Training (arXiv: 2511.19065)、Improved MeanFlow
  - PAE 的同期工作:RAE (Representation Autoencoder)、FAE、AlignTok、RePack
  - Patch Forcing × T2I 已经验证 (Patch Forcing 论文 Sec 5.2)
- **其他领域解决类似问题**:
  - **Video diffusion**: 视频 diffusion 天生就是 "heterogeneous t per frame",有大量经验 — Diffusion Forcing, FIFO-Diffusion, Loong, etc.
  - **3D / point cloud diffusion**: 点云每个点有自己的 noise level (e.g., Point-Voxel Diffusion, LION),有借鉴价值吗?
  - **Sequence diffusion** (text, code): tokens have independent noise levels by design (e.g., MDLM, Discrete Diffusion)
  - **PDE solvers**: adaptive mesh refinement (AMR), domain decomposition, multigrid
  - **Computational geometry**: variable-density sampling, Riemannian optimization
  - **Numerical analysis**: stiff ODE solvers (BDF, Rosenbrock), exponential integrators
- **三个组件中,是否有哪个不应该用**?
  - 也许 PAE 在 HT × MeanFlow 下边际收益变小,简单 VA-VAE 就够了
  - 也许 MeanFlow 在 HT 下不再是最优,Consistency Flow Matching 或 Shortcut Models 反而更好
  - 也许 HT 在 PAE × MeanFlow 下边际收益变小(因为 PAE 已经"语义对齐",空间不均匀的好处变小)
- **完全跳出当前思维框架的方案**:
  - 是否应该放弃 latent diffusion,转向 **pixel diffusion + HT**?(JiT × Patch Forcing 还没人做)
  - 是否 HT 的"per-patch"还不够细,应该 **per-pixel**?(MuLAN 在 CIFAR 上做到了 per-pixel,搬到 DiT 上是空白)
  - 是否应该完全放弃 Patch Forcing 的 LTG,**重新设计 sampler**?
  - 是否 1-NFE 这个目标本身是错的——也许 PDM-3-HT 的最优定位是 **few-NFE high-quality** 而不是 1-NFE?
- **整个技术路线**走偏的可能:
  - 是否 <1B 模型本质上不可能 ImageNet 256 FID < 1.0,我们的目标设定有误?
  - 是否 ImageNet 256 已经不是合适的 benchmark (FID < 1.5 时,FID 本身已经接近 noise floor)?KDD (Kernel Density Distance) 或 sFID 是否更合适?
  - 是否我们花太多力气在"训练 efficiency"上,而真正应该投入的是"推理 quality"?

---

## 第三部分:我们已知的信息(请勿重复研究这些)

### 已读论文(按推荐度排序)

**核心组件(必读)**:
- **PAE** (Yue et al., 2026.05, [arXiv: 2605.07915](https://arxiv.org/abs/2605.07915)) — Prior-aligned AutoEncoder,三个 alignment 损失 (SSR/MCR/SCR) + DAM 模块,gFID **1.03**
- **Patch Forcing** (Schusterbauer et al., CVPR'26, [arXiv: 2604.19141](https://arxiv.org/abs/2604.19141)) — Per-patch heterogeneous timesteps + LTG sampler + difficulty head + adaptive sampling,SiT-XL/2 FID 17.2 → 9.8。**关键配置**:LTG (μ, σ) 在 ImageNet 256 上的默认值、per-token AdaLN、Gaussian NLL 的 λ
- **MeanFlow** (Geng et al., NeurIPS'25 Oral, [arXiv: 2505.13447](https://arxiv.org/abs/2505.13447)) — Average velocity training, 1-NFE FID 3.43,从头训。**关键超参**:25% r≠t, lognorm(-0.4, 1.0) time sampler, p=1.0 adaptive weight, (t, t-r) positional embedding
- **FD-loss / Representation Fréchet Loss** (Yang et al., 2026.04, [arXiv: 2604.28190](https://arxiv.org/abs/2604.28190), [GitHub](https://github.com/Jiawei-Yang/FD-loss)) — **可能颠覆我们路线的工作**。解耦 population size 和 batch size,把 FID 变成可训练 loss。**关键结果**:Post-train 任意 base generator,ImageNet 256 1-NFE FID 0.72。能把 multi-step generator 直接 repurpose 到 1-step,无需 distillation。**关键配置**:queue-based estimator with 50k population,EMA-based estimator 作 alternative,reference statistics 在 100k 真实图像上 offline 计算。提出新指标 FDr⁶(6 representations)
- **MuLAN** (Sahoo et al., NeurIPS'24 Spotlight, [arXiv: 2312.13236](https://arxiv.org/abs/2312.13236)) — Per-pixel + image-conditional noise schedule,ELBO 不变性的反驳。**警告**:UNet + CIFAR/ImageNet32,**没 scale 到 DiT**

**理论 / 方法基础**:
- **REPA** (Yu et al., ICLR'25 Oral, [arXiv: 2410.06940](https://arxiv.org/abs/2410.06940)) — DiT 浅层对齐 DINOv2,17.5× 加速。关键配置:λ=0.5, depth=8/28, DINOv2-B, cosine sim
- **VA-VAE + LightningDiT** (Yao et al., CVPR'25, [arXiv: 2501.01423](https://arxiv.org/abs/2501.01423)) — VAE 训练时用 VF Loss 对齐 DINOv2,FID 1.35
- **REPA-E** (Leng et al., ICCV'25, [arXiv: 2504.10483](https://arxiv.org/abs/2504.10483)) — VAE + DiT 端到端联合训练,REPA 作锚,FID 1.26
- **Min-SNR Weighting** (Hang et al., CVPR'24, [arXiv: 2303.09556](https://arxiv.org/abs/2303.09556)) — Diffusion 训练当作 multi-task,timestep loss reweighting

**相关参考**:
- **DiT** (Peebles & Xie, ICCV'23) — DiT 原文
- **SiT** (Ma et al., ECCV'24) — DiT + Flow Matching
- **SD3** (Esser et al., 2024) — MMDiT, logit-normal time sampling, rectified flow
- **Self-Flow** (Chefer et al., 2026, [arXiv: 2603.06507](https://arxiv.org/abs/2603.06507)) — Concurrent to Patch Forcing,dual-timestep + repr learning
- **Diffusion Forcing** (Chen et al., NeurIPS'24, [arXiv: 2407.01392](https://arxiv.org/abs/2407.01392)) — 视频上的 per-frame 异步 t,Patch Forcing 的灵感来源
- **SRM** (Wewer et al., 2025, [arXiv: 2502.21075](https://arxiv.org/abs/2502.21075)) — Spatial reasoning models,Patch Forcing 的先驱
- **SplitMeanFlow** ([arXiv: 2507.16884](https://arxiv.org/abs/2507.16884)) — JVP-free MeanFlow via Interval Splitting
- **Understanding MeanFlow Training** ([arXiv: 2511.19065](https://arxiv.org/abs/2511.19065)) — Curriculum learning insights
- **TPDM** (Time Prediction Diffusion Model, [arXiv: 2412.01243](https://arxiv.org/abs/2412.01243)) — RL-based time predictor 推理时
- **DDT** (Wang et al., CVPR'26 highlight, [arXiv: 2504.05741](https://arxiv.org/abs/2504.05741)) — Decoupled DiT,**v1 中考虑过但因与 HT 冲突在 v2 中放弃**

**已评估但暂不采用**:
- **DDT** — v1 考虑过,v2 因与 Patch Forcing 设计冲突而放弃
- **Z-Image (S3-DiT)** (Alibaba, 2025.11) — 6B 单流 DiT,主要为 T2I 设计,与我们的 <1B 目标不符
- **DC-AE / SANA** — 32× 深压缩,主要为 4K T2I,我们用 PAE-f16d32 更合适
- **DiT-MoE / DiffMoE** — 稀疏 MoE,可叠加但增加复杂度
- **JiT / PixelDiT** — Pixel-space diffusion,放在 future work

### 已确定的工程配置

| 项 | 我们的选择 | 来源 |
|---|---|---|
| 数据集 | ImageNet-1K, 256×256 (主), 512×512 (次) | ADM evaluation protocol |
| Latent | PAE-f16d32 (32 × 32 × 32) | PAE 默认 |
| Backbone | LightningDiT-XL/2, ~675M params | LightningDiT 配方,改造 per-token AdaLN |
| Time granularity | Per-patch (1024 patches at 32×32 / patch_size=1) | Patch Forcing |
| Time sampler | LTG (Logit-Normal Truncated Gaussian) | Patch Forcing 默认参数为起点 |
| Training objective | Per-Patch MeanFlow (待推导) | 新工作 |
| (r, t) sampling | 25% r≠t,r ~ U(0, t), 75% r=t | MeanFlow + per-patch 适配 |
| Difficulty head | Per-patch log-variance σᵢ² with Gaussian NLL | Patch Forcing |
| **FD-loss post-training (Phase 3.5)** | **Queue-based estimator, population 50k, batch 1024, fine-tune 10K iter** | **FD-loss 论文默认** |
| **Reference statistics** | **6 representations (Inception, SigLIP2, MAE, DINOv2, CLIP, RADIO), offline on 100k ImageNet train images** | **FDr⁶ 论文设置** |
| Optimizer | AdamW, lr 1e-4, β=(0.9, 0.95), wd=0 | MeanFlow 默认 |
| EMA decay | 0.9999 | MeanFlow 默认 |
| Batch size | 256 (global) | MeanFlow 默认 |
| Precision | fp32 warmup 50K iter → bf16,JVP path 强制 fp32 | 稳定性保险 |
| Eval | FID/sFID/IS/Pre/Rec via ADM 50K samples | 标准 |
| CFG | guidance interval + ω sweep | REPA-style scheduling |

### 已确认的算力

- 1× RTX Pro 6000 Blackwell (96GB GDDR7) 常驻
- 1× B300 (288GB HBM3e),Phase 3 (Week 8-11) 借用窗口
- 预计 87-90 H100-day equivalent 总预算
- 15-17 周时间

### 已规划的实验框架

- Phase 0-1: 5 个 baseline 复现 (SiT, LightningDiT+PAE, MeanFlow, Patch Forcing, LightningDiT+PAE+REPA-E) + 三组件独立验证
- Phase 2: Per-Patch MeanFlow Identity 推导与工程实现
- Phase 3: 三联合主训 (PAE × HT × MeanFlow) + CFG sweep + adaptive sampling 评估
- Phase 4: Proposition 1 三推论 + per-patch 扩展 + LTG 参数 sweep
- Phase 5: 论文写作

### 已确定不做的事

- T2I 从零训练 (算力不够)
- ImageNet 512 + 主干 > 1B (算力不够)
- 多分辨率 cascade 训练 (复杂度太高,放 future work)
- Pixel-space diffusion (JiT 路线放 future work)
- Autoregressive 生成 (VAR/MAR 路线生态不成熟,放 future work)

---

## 第四部分:期望的输出格式

请对**每个探索区域**(区域 1-6)按以下结构输出。区域 7 (我们没想到的方向) 单独章节输出。

**特别强调对区域 6 (FD-loss 整合策略) 的期望**:这是本任务书最新加入、对项目影响最大的区域。请在该区域给出**明确的整合策略推荐**(策略 A / B / C 中的哪一个,或全新的策略 D),并对失败模式做尽可能完整的预测。

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

**涉及组件**:PAE / Patch Forcing / MeanFlow / 训练循环 / 推理流程 / 数学推导 中的哪些?

**与现有 PDM-3-HT 系统的对接方式**:
- 需要修改哪些模块?
- 输入格式怎么变?
- 输出格式怎么变?
- 训练超参怎么变?

**难度评估**:[低/中/高] + 具体原因(实现复杂度、debug 难度、对现有 codebase 的侵入性)

**预期效果**:能解决问题到什么程度?有什么局限?如果可能,给出量化预期(预计 FID 改善幅度、训练加速倍数、JVP norm 减少幅度等)

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

### 区域 7 输出格式:意外发现

在研究过程中发现的、不属于上述 6 个探索区域的新方向。结构同上,但**特别强调**:

- 如果发现了**完全超出探索区域**的全新方向——类似于"不加大扇叶,而是重新设计风道"的灵感——请务必单独列出
- 如果发现整个技术路线的**潜在问题**(比如"PDM-3-HT 这个方向本身可能是死胡同,因为 [某个我们没考虑到的根本性原因]"),请直接说
- 如果发现 PDM-3-HT 与某个**完全不同的新范式**(比如 score-based perceptual models, energy-based generative models, neural compression-as-generation)之间有出人意料的联系,请详细展开

这类发现对我们来说价值最高。

### 最终整体建议

文档末尾请给出:

1. **基于本次研究,你对 PDM-3-HT (Path C) 整体方向的判断**:
   - 信心评级 (高/中/低) + 原因
   - 最大的 1-2 个潜在 dealbreaker
   - 你认为 reviewer 最可能 challenge 的点

2. **关于 FD-loss 整合的明确建议**(必答):
   - 推荐策略 A (简单叠加) / B (绕过 MeanFlow) / C (双轨制) 中的哪一个?为什么?
   - 是否存在我们没考虑到的策略 D?
   - FD-loss 的引入是否让 PDM-3-HT 的部分组件变得 redundant?(比如:有了 FD-loss 后,PAE 的 MCR 损失是否还必要?)

3. **比较三个版本的路线**:
   - v1 (PDM-3 with DDT)
   - v2 (PDM-3-HT)
   - v2.1 (PDM-3-HT + FD-loss)
   哪条路 risk-adjusted return 最高?是否应该考虑混合方案?

4. **如果你只能给我们三条建议**,它们是什么?

5. **如果三个推论 (Proposition 1 的实验验证) 全部失败**,你建议的 Plan B / Plan C 是什么?

6. **关于 FD-loss 的元批判**(必答):
   - FD-loss 的 SOTA 数字 (0.72) 是否代表真实的 visual quality 进步,还是只是 representation overfitting?
   - 我们在论文中应该如何 frame 这个问题,以避免 reviewer 用"FD-loss 也是 representation gaming"的理由 reject?

---

*本文档生成日期:2026-05-27*
*版本:v2.1 (v2 基础上增加 FD-loss 整合策略探索区域)*
*项目阶段:开题(Pre-Phase 0,环境尚未搭建)*
*预计开始执行:Week 1 (~2026-05-29)*
