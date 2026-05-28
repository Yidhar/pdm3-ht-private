# PDM-3-HT v2.1 研究报告：PAE × Heterogeneous Timesteps × MeanFlow × FD-loss

> 生成日期：2026-05-27  
> 输入任务书：`/workspace/PDM/2026-05-27-pdm3-ht-research-brief-v2.1.md`  
> 输出文件：`/workspace/PDM/2026-05-27-pdm3-ht-research-report-v2.1.md`  
> 报告定位：面向开题与 Phase 0/1 执行的技术路线研究、风险拆解、MVP 方案与 fallback 设计。  
> 重要声明：任务书中部分 2026 论文与指标为项目给定背景；本报告将其作为当前研究假设使用。涉及 FD-loss 代码实现的判断来自公开 GitHub 源码抽样核验；涉及 FDr⁶ representation 组成处，任务书与当前 GitHub 脚本中存在命名差异，建议以后续论文最终版 / 官方评测脚本为准。

---

## 0. 执行摘要

### 0.1 一句话结论

**PDM-3-HT 比 v1 的 PDM-3/DDT 路线有更高的 risk-adjusted return；但“Per-Patch MeanFlow Identity”不是一个简单的逐 patch 标量推广，而是一个由 attention 耦合的多时间变量 bundle 问题。**如果直接使用任务书中的朴素公式

\[
 u_i = v_i - (t_i-r_i)\,[v_i\cdot\partial_z u_i + \partial_t u_i]
\]

会漏掉 cross-patch 时间与状态耦合项，最多只能作为 diagonal approximation。严格实现应把所有 patch 的时间向量作为整体，使用一次 **full-bundle JVP** 计算沿整条异步轨迹束的方向导数。

FD-loss 的出现显著改变项目优先级：**不要把 Per-Patch MeanFlow 当成唯一主线。推荐策略 D：Flow-Matching 主线 + Per-Patch MeanFlow 风险分支 + FD-loss 前置闸门。**也就是先把更稳的 `PAE × HT × Flow Matching` 做成强 baseline，并在 Phase 0/1 就复现 / 接入 FD-loss；Per-Patch MeanFlow 继续作为高价值理论分支，但必须通过小模型 JVP 稳定性 gate 后才进入 B300 主训。

### 0.2 最重要的 6 个判断

1. **Path C / PDM-3-HT 方向值得做，信心：中高。** 砍掉 DDT 后，工程侵入性大幅下降；HT 与 PAE 在“空间结构 / 局部流形 / patch 难度”上存在自然协同。
2. **Per-Patch MeanFlow 的数学核心是“全向量时间导数”，不是逐 patch 偏导。** 对 token 之间有 attention 的 DiT，\(u_i\) 依赖所有 \(z_j,t_j\)。正确目标应包含 \(\sum_j \Delta_j\partial u_i/\partial t_j\) 和 \(\sum_j \partial u_i/\partial z_j\cdot \Delta_j v_j\)。
3. **JVP 应一次 batched/full-bundle 计算，不应 N 次逐 patch 计算。** `torch.func.jvp` 可以对整个模型输入 `(z, r_vec, t_vec)` 做一次 forward-mode JVP；关键是把 tangent 写成整条 bundle 的方向。
4. **PAE-MCR 对 JVP 稳定性只能提供条件性帮助，不能单独保证。** 还必须假设 attention operator、time embedding、AdaLN、QK softmax 温度 / spectral norm 有界。否则可构造满足 PAE 平滑但 attention 导致 JVP 爆炸的反例。
5. **FD-loss 最适合作为 post-training 与决策闸门，而不是从零训练主损失。** 它可以极大提高 1-NFE 指标，但也带来 representation gaming 风险；必须用 hold-out representation、sFID、precision/recall、最近邻记忆检查和人工网格防御 reviewer 质疑。
6. **最终论文叙事应把 FD-loss 与 non-FD base model 分开报告。** Base model 证明 PAE×HT×MeanFlow/Flow-Matching 的生成能力；FD-loss 证明 distribution-level post-training 的上限与 metric-aware 后训练收益。不要把 FD-loss 后的 FID 当作唯一质量证据。

### 0.3 建议的总路线：策略 D

**策略 D：FD-first staged dual-track**

- **主线 M0：PAE × HT × 标准 Flow Matching。** 作为稳定、可训练、可 FD-loss post-train 的工程主线。
- **分支 M1：PAE × HT × Per-Patch MeanFlow。** 只在 full-bundle JVP 公式、小模型 toy、bf16/fp32 stability gate 通过后进入大规模。
- **FD-loss 前置闸门：** Phase 0/1 先复现 FD-loss，并在小模型 / 已有 generator 上确认 queue、feature extractors、reference stats、evaluation pipeline 可用；不要等到 Week 12 才发现 FD-loss 与 PAE/LightningDiT 代码栈不兼容。
- **决策规则：** 如果 `PAE×HT×Flow Matching + FD-loss` 已接近或达到 1-NFE FID ≤ 0.7 且 hold-out FDr 稳健，则 MeanFlow 可降级为理论 / future 分支；如果 FD-loss 出现 representation overfit 或 hold-out FDr 失败，则 MeanFlow/few-NFE 质量仍是主贡献。

---

## 1. 探索区域 1：Per-Patch MeanFlow Identity 的精确推导与算法实现

### 1.1 发现的方案全景

- **方案 1.1：朴素 diagonal per-patch MeanFlow。** 每个 patch 单独套 scalar MeanFlow identity，只对 \(t_i\) 求偏导；实现简单，但忽略 attention cross terms。
- **方案 1.2：Full-bundle MeanFlow Identity（推荐理论主方案）。** 用全局进度变量 \(s\) 参数化所有 patch 的异步时间束，JVP 沿 \((z,t)\) 的整条方向计算；数学上最一致。
- **方案 1.3：Stop-context / semi-coupled JVP 近似。** forward 仍用全 attention，但 JVP tangent 对非本 patch 或远邻 stop-gradient，降低 cross term 方差。
- **方案 1.4：Operator splitting / domain decomposition。** 借鉴 PDE、CFD、N-body splitting：把 global context 与 local patch update 拆成交替子步，避免一次求全耦合导数。
- **方案 1.5：JVP-free 替代目标。** 使用 SplitMeanFlow、Consistency Flow Matching、Shortcut/Rectified Flow distillation、Self-Flow 类目标绕开 JVP。
- **方案 1.6：Finite-difference audit + toy ODE proof harness。** 不作为最终训练方案，但必须用于验证公式符号、尺度、tangent 是否正确。

### 1.2 方案详情

#### 方案 1.1：朴素 diagonal per-patch MeanFlow

**一句话原理**：把每个 patch 当作独立 scalar-time MeanFlow 样本，令

\[
 u_{i,\text{tgt}} = v_i - \Delta_i\,\frac{\partial u_i}{\partial t_i},\quad \Delta_i=t_i-r_i
\]

并忽略 \(j\neq i\) 的 cross-patch 时间与状态影响。

**灵感来源**：标准 MeanFlow identity；Patch Forcing 的 per-token timestep conditioning。

**具体实现步骤**：

1. 保持模型 forward 输入为 `z_t: [B,N,C]`、`r_vec/t_vec: [B,N]`、`y: [B]`。
2. 对 `t_vec` 使用 tangent 只在本 patch 方向为 1 或 \(\Delta_i\)，但不传播其他 patch 的 tangent。
3. 可以用 `vmap(jvp)` 逐 patch 做，也可以构造 N 个 basis tangent 后合并；但前者成本乘 N，后者显存爆炸。
4. 实际上若要高效，只能近似：一次 full JVP 后取 diagonal 近似，或设计 mask 使 cross tangent 为 0。

**涉及组件**：MeanFlow 训练目标、JVP 计算、LightningDiT per-token AdaLN。

**对接方式**：只修改 training step；模型结构不变。

**难度评估**：低到中。公式易写，debug 直观；但若真做 N 次 JVP，计算不可接受。

**预期效果**：可能作为 baseline 或 ablation 有价值；不建议作为论文中的“严格 identity”。若 attention cross-coupling 弱，diagonal 可能足够；若 cross-coupling 强，训练目标会系统性偏差。

**风险与不确定性**：

- 公式漏项，reviewer 容易指出不严谨。
- 若直接用于主训，可能学到局部一致但全局不一致的 velocity。
- N 次 JVP 不可扩展；强行近似时要报告 cross term gap。

**建议 mini-experiment**：在一个 4-token toy transformer 上，冻结随机 attention，比较 diagonal JVP 与 finite difference full endpoint derivative 的相对误差：

\[
\text{diag\_gap}=\frac{\|J_{\text{full}}-J_{\text{diag}}\|}{\|J_{\text{full}}\|+10^{-8}}
\]

若 median gap > 0.2，不能把 diagonal 当主公式。

---

#### 方案 1.2：Full-bundle MeanFlow Identity（推荐理论主方案）

**一句话原理**：把 per-patch time vector 看成一个整体 \(\boldsymbol{t}\in[0,1]^N\)，用全局进度 \(c\in[0,1]\) 缩放整条异步轨迹束，MeanFlow identity 变成沿 bundle endpoint 的方向导数。

**严格化符号**：

令

\[
\boldsymbol{\rho}=(r_1,\ldots,r_N),\quad
\boldsymbol{\tau}=(t_1,\ldots,t_N),\quad
\boldsymbol{\Delta}=\boldsymbol{\tau}-\boldsymbol{\rho}.
\]

对 rectified/linear interpolant：

\[
 z_{\boldsymbol{\tau}}^{(i)}=(1-\tau_i)z_0^{(i)}+\tau_i\epsilon^{(i)},\quad
 v_i=\epsilon^{(i)}-z_0^{(i)}.
\]

引入 endpoint scaling 参数 \(c\)：

\[
 \tau_i(c)=\rho_i+c\Delta_i,
 \qquad
 z(c)^{(i)}=(1-\tau_i(c))z_0^{(i)}+\tau_i(c)\epsilon^{(i)}.
\]

定义 patch \(i\) 的平均速度为

\[
 U_i(c)=\int_0^1 v_i\big(z(\rho+c\alpha\Delta),\,\rho+c\alpha\Delta\big)\,d\alpha.
\]

等价地，patch 位移满足

\[
 c\Delta_i U_i(c)=\int_0^c \Delta_i\,v_i\big(z(\rho+a\Delta),\rho+a\Delta\big)\,da.
\]

对 \(c\) 求导并在 \(c=1\) 处取值，得到

\[
 U_i(1)+\frac{dU_i}{dc}\bigg|_{c=1}=v_i\big(z(\tau),\tau\big).
\]

因此 full-bundle MeanFlow target 是

\[
 U_{i,\text{tgt}} = v_i - \operatorname{JVP}_c[U_i],
\]

其中 JVP 的 tangent 包含所有 patch：

\[
 \dot z^{(j)}=\Delta_j v_j,
 \qquad
 \dot t_j=\Delta_j,
 \qquad
 \dot r_j=0.
\]

展开后可见 cross terms：

\[
\frac{dU_i}{dc}
=\sum_j \left\langle \frac{\partial U_i}{\partial z^{(j)}},\Delta_j v_j\right\rangle
+\sum_j \frac{\partial U_i}{\partial t_j}\Delta_j.
\]

这正是朴素 per-patch 公式遗漏的部分。注意：**如果 tangent 已经乘了 \(\Delta\)，target 中就不要再额外乘一次 \((t_i-r_i)\)。**两种等价写法只能选一种：

- 写法 A：tangent = `(Δ*v, Δ)`，则 `u_tgt = v - du_dc`。
- 写法 B：tangent = `(v, 1)` 的 diagonal scalar derivative，则 `u_tgt = v - Δ*du_dt`，但这是近似。

**灵感来源**：MeanFlow 的 endpoint derivative；Neural ODE 中对耦合状态向量整体求导；CFD local-time stepping 把异步时间作为一个 time-vector state。

**具体实现步骤**：

1. **模型接口统一**：建议模型 forward 接收 `r_vec` 与 `t_vec`，或接收 `t_vec` 与 `dt_vec=t_vec-r_vec`。不要在不同文件中混用 scalar 和 vector。
2. **JVP fp32 island**：即便主训练 bf16，JVP forward 内部强制 `z, r, t, v` 为 fp32；输出再 cast 回训练 dtype。
3. **一次 full-bundle JVP**：

```python
from torch.func import jvp

# z_t: [B, N, C] 或 [B, C, H, W]
# r_vec, t_vec: [B, N]
# v: [B, N, C]，linear interpolant velocity eps - z0

def model_u(z, r, t, y):
    out = model(z, r_vec=r, t_vec=t, y=y)
    return out.u  # [B, N, C]

with torch.cuda.amp.autocast(enabled=False):
    z_f = z_t.float()
    r_f = r_vec.float()
    t_f = t_vec.float()
    v_f = v.float()
    delta = (t_f - r_f).clamp_min(0.0)

    z_dot = delta[..., None] * v_f
    r_dot = torch.zeros_like(r_f)
    t_dot = delta

    u_pred, du_dc = jvp(
        lambda z, r, t: model_u(z, r, t, y),
        (z_f, r_f, t_f),
        (z_dot, r_dot, t_dot),
    )

    u_tgt = v_f - du_dc
    loss_mf = ((u_pred - u_tgt.detach()) ** 2).mean()
```

若模型使用 `(t, dt=t-r)` embedding：

```python
dt_dot = delta
u_pred, du_dc = jvp(
    lambda z, t, dt: model_u_dt(z, t, dt, y),
    (z_f, t_f, delta),
    (z_dot, t_dot, dt_dot),
)
u_tgt = v_f - du_dc
```

4. **r=t 退化样本**：当 \(r=t\) 时 \(\Delta=0\)，`z_dot/t_dot/dt_dot=0`，target 退化为 `u_tgt=v`，也就是 Flow Matching。保留 MeanFlow 的 75% `r=t` 配方可显著降低早期不稳定。
5. **finite-difference audit**：每隔若干 step，对小 batch 做：

```python
eps_fd = 1e-3
u0 = model_u(z_f, r_f, t_f, y)
z1 = z_f + eps_fd * z_dot
t1 = t_f + eps_fd * t_dot
r1 = r_f
u1 = model_u(z1, r1, t1, y)
du_fd = (u1 - u0) / eps_fd
rel_err = (du_fd - du_dc).norm() / (du_fd.norm() + 1e-8)
```

6. **监控 cross-coupling**：同时计算 stop-context/diagonal 近似，记录 `cross_ratio`，决定是否有必要做 full JVP。

**涉及组件**：MeanFlow 数学推导、训练循环、JVP、LightningDiT forward signature、precision policy、telemetry。

**与现有 PDM-3-HT 对接方式**：

- `TimeEmbedding`：从 scalar `[B]` 改为 per-token `[B,N]`。
- `AdaLN`：scale/shift/gate 输出 `[B,N,D]`。
- `model.forward`：返回至少 `u` 和可选 `logvar`。
- `train_step`：新增 full-bundle JVP branch；`r=t` 样本可跳过 JVP 节省计算。

**难度评估**：中高。公式本身清晰，但 JVP 与 transformer fused kernels、activation checkpointing、bf16、DDP 组合容易出问题。

**预期效果**：

- 数学上最 defendable。
- 若 JVP 稳定，可形成论文核心理论贡献。
- 预期相对 diagonal 能改善 global coherence，尤其在 heterogeneous t contrast 大时。

**风险与不确定性**：

- full JVP 可能显存/时间成本超出预期。
- attention cross terms 可能导致 JVP 方差很大。
- 如果 fused attention 不支持 forward-mode AD，需要切换 kernel 或使用 fallback attention。

**先做 mini-experiment**：

- 8×8 latent、DiT-Tiny、N=64 tokens，比较 full vs diagonal vs finite-difference。
- 记录 JVP p50/p95/p99、NaN、loss variance、step time。
- 若 p99 JVP norm 比 Flow Matching target norm 高 >10×，必须启用稳定化或 fallback。

---

#### 方案 1.3：Stop-context / semi-coupled JVP 近似

**一句话原理**：forward 使用完整 attention 保持生成能力；JVP 计算时只让当前 patch 或局部邻域的 tangent 参与，阻断远距离 cross-patch tangent，降低数值方差。

**灵感来源**：截断反向传播、implicit differentiation 中的 stop-gradient approximation、domain decomposition。

**具体实现步骤**：

1. 将 tokens 划分为局部 windows，例如 4×4 或 8×8。
2. 构造 tangent mask `M_i`：只保留同一 window 或距离小于 R 的 token tangent。
3. 训练时随机选择 full JVP 与 masked JVP 的混合：

```python
if step < warmup_steps:
    jvp_mode = "diag_or_local"
elif step % full_jvp_every == 0:
    jvp_mode = "full"
else:
    jvp_mode = "local"
```

4. 对 masked JVP 加一个 consistency penalty，使其不要偏离 full JVP 的低频部分：

\[
 L_{\text{jvp-cons}}=\|\text{LP}(J_{\text{local}})-\text{LP}(J_{\text{full}})\|^2.
\]

**涉及组件**：JVP path、attention mask 或 tangent construction、训练 telemetry。

**难度评估**：中。实现不难，但理论上是近似，需要实验支撑。

**预期效果**：降低 JVP 方差与 NaN 率；若图像局部性强，质量损失可能很小。

**风险**：全局语义 patch（物体主体、背景大区域）可能需要远距离 cross term；local JVP 会削弱一致性。

---

#### 方案 1.4：Operator splitting / domain decomposition

**一句话原理**：把全耦合 per-patch dynamics 拆成“全局 context 更新”和“局部 patch 更新”两个算子，训练或推理时交替应用，避免一次求解完整异步耦合系统。

**灵感来源**：PDE operator splitting、Strang splitting、CFD local time stepping、N-body Hamiltonian splitting。

**具体实现步骤**：

1. 将模型输出拆成两部分：

\[
 u_i = u_i^{\text{global}} + u_i^{\text{local}}.
\]

2. `global` 分支使用 downsampled latent 或 class token，时间取统计量如 `max(t)` / `mean(t)` / quantile；`local` 分支使用 per-token t。
3. JVP 只对 local 分支做 full per-token tangent；global 分支用 scalar MeanFlow 或 stop-gradient。
4. 推理时使用 Strang-style：半步 global → 一步 local hard patches → 半步 global。

**涉及组件**：模型结构、训练目标、推理 sampler。

**难度评估**：高。会改变 LightningDiT 结构，接近重新设计 backbone。

**预期效果**：如果 full-bundle JVP 不稳定，这是比完全放弃 MeanFlow 更优雅的理论 fallback；可能改善全局一致性。

**风险**：工程侵入性接近 DDT；不建议 Phase 0/1 做。

---

#### 方案 1.5：JVP-free 替代目标

**一句话原理**：如果 full-bundle MeanFlow 的 JVP 不稳定，用无需 JVP 的 one/few-step 目标替代，例如 SplitMeanFlow、Consistency Flow Matching、shortcut models 或 teacher-free self-consistency。

**灵感来源**：SplitMeanFlow、consistency models、rectified flow distillation、self-flow / shortcut learning。

**具体实现步骤**：

1. 先训练 `PAE×HT×Flow Matching` 多步模型。
2. 对同一 `(z0, eps, t_vec)` 构造两个时间划分：single interval 与 split interval。
3. 用 interval consistency 代替 JVP：

\[
 z_r^{\text{one}} = z_t - \Delta\,u_\theta(z_t,r,t),
\]

\[
 z_m = z_t - (t-m)u_\theta(z_t,m,t),\quad
 z_r^{\text{two}} = z_m - (m-r)u_\theta(z_m,r,m),
\]

\[
 L_{\text{split}}=\|z_r^{\text{one}}-\text{sg}(z_r^{\text{two}})\|^2.
\]

4. 对 per-patch time vector，`m_i = r_i + \alpha_i(t_i-r_i)`；可让 hard patches 使用更细 split。

**涉及组件**：训练目标，不一定改模型结构。

**难度评估**：中。多一次或两次 forward，少了 JVP kernel 风险。

**预期效果**：可能牺牲一点 MeanFlow 理论纯度，但工程更稳。若 FD-loss 后训练强，可能足够达标。

**风险**：需要 teacher/stopgrad 设计避免 collapse；训练成本可能大于 JVP。

---

#### 方案 1.6：Finite-difference audit + toy ODE proof harness

**一句话原理**：用可解析或高精度有限差分的 toy 系统验证 full-bundle identity 的符号、尺度、tangent 与 stop-gradient 位置。

**具体实现步骤**：

1. 构造线性耦合 velocity：

\[
 v(z,t)=Az+Bt+c,
\]

其中 \(A\) 有 cross-token blocks。
2. 用数值积分得到 ground-truth average velocity。
3. 训练一个小 MLP/mini-transformer 拟合 \(U\)，比较：
   - full-bundle identity target
   - diagonal target
   - finite-difference target
4. 验证 \(\Delta\) 是否被重复乘、`r_dot` 是否应为 0、`dt_dot` 是否应等于 \(\Delta\)。

**难度评估**：低。建议作为 Phase 2 第一件事。

**预期效果**：极大降低后续主训公式 bug 风险。

### 1.3 技术顾问建议

1. **推荐 MVP 路径**：先做 full-bundle JVP 的 tiny model + finite-difference audit；同时保留 diagonal 作为 ablation。不要一开始就把 diagonal 写进主论文公式。
2. **推荐终极路径**：full-bundle JVP + fp32 island + cross-ratio telemetry + local/stop-context fallback。论文中可以把 diagonal 作为近似并量化误差。
3. **实施顺序**：
   1. Toy ODE / 4-token transformer 验证 identity；
   2. DiT-Tiny 8×8 latent 跑 full-vs-diag；
   3. LightningDiT-B/2 在 PAE latent 上稳定性测试；
   4. 只有 p99 JVP、NaN、LR gate 通过后，进入 XL/2 主训。
4. **不该放进本篇但值得做**：operator splitting 结构化 backbone、Hamiltonian/symplectic 类高阶积分、implicit solver。这些会把论文主线拖散，适合 future work。

---

## 2. 探索区域 2：LTG Sampler 在 PAE Latent Space 的最优形式

### 2.1 发现的方案全景

- **方案 2.1：照搬 Patch Forcing 默认 LTG。** 作为必需 baseline，风险最低。
- **方案 2.2：PAE latent 上的小网格校准。** 对 \((\mu,\sigma)\)、truncation width、`t_max` 分布做低成本 sweep。
- **方案 2.3：loss-balanced / information-equalized sampler。** 在线维护 `(patch, t-bin)` loss histogram，让采样分布朝高误差区域倾斜。
- **方案 2.4：PAE geometry-aware sampler。** 用 SSR/MCR/SCR 残差或 latent curvature 决定 patch 的 t 分布宽度。
- **方案 2.5：QMC / low-discrepancy time-vector sampler。** 用 Sobol/Halton + Gaussian copula 生成 \(t_1,
...,t_N\)，降低高维采样方差。
- **方案 2.6：quantile/moment-controlled sampler。** 不只控制 max，还控制 mean、median、p90、局部 contrast。
- **方案 2.7：joint \((r,t)\) sampler。** 在 HT 的 \(t_i\) 基础上为 MeanFlow 设计 \(r_i\)，保证 \(0\le r_i\le t_i\)。

### 2.2 方案详情

#### 方案 2.1：默认 LTG baseline

**一句话原理**：复用 Patch Forcing 的 LogitNormal `t_max` + truncated Gaussian per-patch sampling，先证明 PAE×HT 能跑通。

**具体实现步骤**：

1. `t_max ~ sigmoid(N(μ, σ²))`。
2. 对每个 patch 采样 `t_i`，分布中心靠近 `t_max`，support 限制在 `[0,t_max]`。
3. 记录每个 batch 的 `max(t)`, `mean(t)`, `p50/p90/p99(t)`, local contrast `|t_i-t_j|`。
4. 与 SD-VAE/VA-VAE latent 上的同配置对比 loss curve 与 FID。

**难度评估**：低。

**预期效果**：提供最重要 baseline；若已经显著优于 vanilla FM，说明 HT 与 PAE 兼容。

**风险**：PAE latent 信息密度高、通道数多，默认 LTG 可能让训练过度集中在高 noise 或 max-controlled 过强。

---

#### 方案 2.2：PAE latent 小网格校准

**一句话原理**：用有限算力寻找 PAE latent 的合适 `(μ, σ, trunc_width)`，不假设 SD-VAE 参数最优。

**具体实现步骤**：

1. 选择 6–9 个配置，而不是大规模 grid：
   - `μ ∈ {μ0-0.3, μ0, μ0+0.3}`
   - `σ ∈ {0.7σ0, σ0, 1.3σ0}`
2. 每个配置训练 DiT-S/B 或 10–30K iter proxy。
3. 指标：
   - train loss by t-bin
   - validation reconstruction/denoising loss by t-bin
   - per-patch difficulty calibration
   - small-sample FID / feature FD
4. 淘汰标准：高 t-bin loss 长期不降、local contrast 过大导致 JVP p99 爆炸、或 1-NFE proxy 明显差。

**难度评估**：低到中。

**预期效果**：用很小成本避免把主训押在错误 sampler 上。

**风险**：proxy 模型上最优参数未必迁移到 XL/2；需要保留 telemetry 而不是只看早期 FID。

---

#### 方案 2.3：loss-balanced / information-equalized sampler

**一句话原理**：把 timestep sampling 当作多任务采样问题，在线提高高损失 t-bin / patch-type 的采样概率，类似 Min-SNR 与 adaptive importance sampling。

**具体实现步骤**：

1. 把 \([0,1]\) 分为 K 个 bins，例如 K=32。
2. 维护 EMA loss histogram：

```python
loss_hist[bin] = beta * loss_hist[bin] + (1-beta) * batch_loss_in_bin
count_hist[bin] = beta * count_hist[bin] + (1-beta) * batch_count_in_bin
score = loss_hist / (count_hist + eps)
```

3. 更新 sampler 的 target density：

\[
 p_{\text{new}}(b) \propto (score_b + \epsilon)^\alpha p_{\text{base}}(b)^{1-\alpha}.
\]

4. 限制 KL drift，避免 training-test mismatch：

\[
 D_{KL}(p_{\text{new}}\|p_{\text{base}}) < \delta.
\]

5. 对 HT 还要维护 local contrast histogram：若 `max_neighbor_delta_t` 过大且 loss/JVP 高，则降低高 contrast 采样。

**涉及组件**：sampler、训练 loop、telemetry。

**难度评估**：中。

**预期效果**：在 PAE latent 信息分布不同的情况下，比固定 LTG 更稳。可能减少 10–20% 收敛步数，但需实测。

**风险**：过度追逐高 loss bin 会造成 sampler collapse；需要 KL/clipping 与 periodic reset。

---

#### 方案 2.4：PAE geometry-aware sampler

**一句话原理**：PAE 已提供 SSR/MCR/SCR 残差信号，可让语义复杂或流形边界 patch 采更细致/更高难度的 time distribution。

**具体实现步骤**：

1. 离线为训练集缓存 per-patch PAE geometry map：

\[
 d_i = \operatorname{norm}(w_s SSR_i + w_m MCR_i + w_c SCR_i).
\]

2. 根据 \(d_i\) 调整 patch t 的均值或方差：

```python
base_t = ltg_sample(...)
d = normalize(diff_prior)  # [B,N]
# hard patch 更常见中高噪，或更大 variance；二者需 ablation
logit_t = logit(base_t)
logit_t = logit_t + alpha * (d - d.mean(dim=1, keepdim=True))
t_vec = sigmoid(logit_t).clamp(0, t_max)
```

3. 控制每张图的 `t_max` 不变，只改变 patch 间分配，避免破坏 Patch Forcing 的 max-information 设计。
4. 训练后比较 difficulty head 与 PAE prior 的相关性。

**涉及组件**：PAE preprocessing、sampler、difficulty head。

**难度评估**：中。需要访问或重算 PAE alignment residuals。

**预期效果**：更早聚焦难 patch；可能提升 adaptive sampling 的信号质量。

**风险**：PAE residual 不一定等价于 diffusion denoising difficulty；过度使用会让模型偏向语义边界、忽略纹理。

---

#### 方案 2.5：QMC / low-discrepancy time-vector sampler

**一句话原理**：用 Sobol/Halton 等低差异序列生成高维 \(t\)-vector 的基础随机数，再经 LTG/Gaussian copula 映射，降低 batch 内 time coverage 方差。

**具体实现步骤**：

1. 使用 `torch.quasirandom.SobolEngine(dimension=N+1, scramble=True)` 生成 `[B,N+1]` uniform。
2. 第一维映射到 `t_max`，其余 N 维映射到 truncated Gaussian。
3. 为避免相邻 patch 出现棋盘式 quasi pattern，每个 step 随机 permutation 或用空间 Hilbert order。

```python
sobol = torch.quasirandom.SobolEngine(dimension=N+1, scramble=True)
u = sobol.draw(B).to(device)
t_max = sigmoid(mu + sigma * normal_icdf(u[:, 0]))
# u[:,1:] -> truncated normal CDF inverse, then clamp to [0,t_max]
```

**涉及组件**：sampler。

**难度评估**：低到中。

**预期效果**：减少 small batch 下高维 time-vector 的 Monte Carlo 噪声，尤其对 batch size 256 较有用。

**风险**：QMC 的低差异性可能与图像 spatial locality 产生伪相关；必须 randomize/scramble。

---

#### 方案 2.6：quantile/moment-controlled sampler

**一句话原理**：Patch Forcing 强调 max information，但 PAE latent 可能让 max 不再足够；同时控制 mean、p90、local contrast 等统计量。

**具体实现步骤**：

1. 采初始 `t_vec` 后，计算统计量：
   - `m=max(t)`
   - `q90=quantile(t,0.9)`
   - `avg=mean(t)`
   - `contrast=mean_abs_neighbor_diff(t)`
2. 若偏离目标区间，做 affine/logit-space correction：

```python
logits = logit(t_vec)
logits = a * logits + b
# choose a,b by matching mean and q90 approximately
```

3. 对 local contrast 加约束：

\[
L_{\text{contrast}}=\max(0, \operatorname{mean}_{(i,j)\in E}|t_i-t_j|-\gamma)^2.
\]

该项不一定作为训练 loss，也可只作为 sampler rejection criterion。

**难度评估**：中。

**预期效果**：降低异步时间过度粗糙导致的 attention/JVP 不稳。

**风险**：过多手工控制会偏离原 LTG 理论；应只作为 ablation。

---

#### 方案 2.7：MeanFlow joint \((r,t)\) sampler

**一句话原理**：先采 HT 的 \(t_i\)，再为每个 patch 采 \(r_i\le t_i\)，并保留大量 \(r_i=t_i\) 样本使目标退化为 Flow Matching。

**推荐实现**：

```python
# t_vec from LTG/PAE-aware sampler
is_mf = torch.rand(B, N, device=device) < p_mf   # e.g. 0.25
beta = torch.rand(B, N, device=device)           # U(0,1)
r_vec = torch.where(is_mf, beta * t_vec, t_vec)
# optional: beta distribution biases shorter intervals early
# beta ~ Beta(a,b), start near 1 then anneal to uniform
```

**改进版**：

- Early training：\(r_i\approx t_i\)，短 interval，降低 JVP。
- Mid training：\(r_i/t_i\sim U(0,1)\)。
- Late training：增加 long interval 与 \(r_i=0,t_i=1\) 的 1-NFE endpoint 样本。

**难度评估**：低。

**预期效果**：稳定 MeanFlow 训练；也为 1-NFE endpoint 提供监督。

**风险**：若 long interval 太少，1-NFE 学不好；若太多，JVP 爆炸。

### 2.3 技术顾问建议

1. **推荐 MVP 路径**：默认 LTG + 小网格 PAE 校准 + joint \((r,t)\) short-to-long curriculum。先不要上 learnable sampler。
2. **推荐终极路径**：loss-balanced sampler 与 PAE geometry-aware sampler 融合，但加 KL drift 与 local contrast 限制。
3. **实施顺序**：默认 LTG → PAE grid → r/t curriculum → QMC → online loss-balanced → geometry-aware。
4. **未来工作**：可学习 sampler、optimal-control 形式的 sampler、真正基于 Fisher information / mutual information 的理论最优采样。这些很有价值，但不应阻塞主线。

---

## 3. 探索区域 3：Per-patch Difficulty Head 与 PAE 语义先验的协同

### 3.1 发现的方案全景

- **方案 3.1：原版 Patch Forcing log-variance head。** 以 Gaussian NLL 学 per-patch uncertainty。
- **方案 3.2：PAE residual 作为 zero-cost difficulty prior。** 直接用 SSR/MCR/SCR 残差做调度。
- **方案 3.3：PAE prior bootstrap difficulty head（推荐）。** 把 PAE residual 注入模型，并作为 early auxiliary target。
- **方案 3.4：multi-signal / multi-head difficulty。** 区分语义难度、纹理难度、边界/流形难度、轨迹难度。
- **方案 3.5：UQ 方法：ensemble / MC dropout / evidential。** 用不确定性估计校准 adaptive NFE。
- **方案 3.6：cross-patch difficulty。** 难度不只来自 patch 自身，还来自邻域/attention context 的复杂度。

### 3.2 方案详情

#### 方案 3.1：原版 log-variance head

**一句话原理**：模型输出 `u_pred` 与 `logvar_i`，用 heteroscedastic Gaussian NLL 让高误差 patch 学到高 variance。

**具体实现步骤**：

```python
err = (u_pred.detach() - v).pow(2).mean(dim=-1)  # [B,N], if head target stopgrad pred
logvar = out.logvar.clamp(-8, 8)
loss_nll = 0.5 * (err * torch.exp(-logvar) + logvar).mean()
loss = loss_mf_or_fm + lambda_var * loss_nll
```

更稳定的写法是对 `u_pred` 的 stop-gradient 与否做 ablation：

- `err=(sg(u_pred)-v)^2`：variance head 不反向影响 velocity。
- `err=(u_pred-v)^2`：完整 NLL，可能影响主训练。

**难度评估**：低。

**风险**：logvar 可通过整体偏移降低 NLL；需要 clamp、calibration metric 与 λ sweep。

---

#### 方案 3.2：PAE residual zero-cost difficulty prior

**一句话原理**：PAE 训练时已经知道每个 patch 与 VFM/流形约束的对齐残差，这些 residual 可作为不额外训练的难度估计。

**具体实现步骤**：

1. 对训练集和验证集缓存：
   - `SSR_i`：空间结构对齐残差。
   - `MCR_i`：局部连续性/扰动一致性残差。
   - `SCR_i`：语义一致性残差。
2. 归一化：

\[
\hat d_i=\operatorname{ranknorm}(w_sSSR_i+w_mMCR_i+w_cSCR_i).
\]

3. 推理时用 `d_i` 分配 extra NFE：

```python
hard = d_prior > torch.quantile(d_prior, 0.8)
steps_i = torch.where(hard, 2, 1)
```

**难度评估**：低到中，取决于 PAE code 是否暴露 residual maps。

**预期效果**：提供 no-head baseline；可检验 PAE prior 与真实 denoising difficulty 的相关性。

**风险**：PAE residual 是 tokenizer 视角，不等于 diffusion trajectory error；可能对边界/纹理过敏。

---

#### 方案 3.3：PAE prior bootstrap difficulty head（推荐）

**一句话原理**：用 PAE residual 作为 difficulty head 的先验或 warm-start，而不是完全替代 learned uncertainty。

**具体实现步骤**：

1. 计算 `diff_prior_i`，注入 time conditioning：

```python
diff_emb = diff_mlp(diff_prior[..., None])  # [B,N,D]
time_emb = time_mlp(t_vec) + diff_emb
adaln_params = adaln_mlp(time_emb + class_emb[:, None, :])
```

2. early stage 加辅助校准 loss：

\[
L_{\text{prior}}=\operatorname{Huber}(\log\sigma_i^2-\operatorname{sg}(a\hat d_i+b)).
\]

3. `λ_prior` 逐步 decay，让 head 后期由真实 denoising error 主导：

```python
lambda_prior = lambda0 * max(0, 1 - step / prior_decay_steps)
loss += lambda_prior * huber(logvar, prior_target.detach())
```

4. 验证三个相关性：
   - `corr(PAE prior, actual denoising error)`
   - `corr(logvar, actual denoising error)`
   - `corr(logvar+prior, adaptive sampling gain)`

**涉及组件**：PAE preprocessing、模型 conditioning、difficulty head loss、推理 scheduler。

**难度评估**：中。

**预期效果**：早期更快学到“哪里难”，减少 difficulty head 冷启动；可能提升 adaptive few-NFE 的 Pareto。

**风险**：先验过强会锁死错误难度；必须 decay 或 gate。

---

#### 方案 3.4：multi-signal / multi-head difficulty

**一句话原理**：单个 \(\sigma_i^2\) 无法区分“语义不确定”“纹理高频”“边界流形”“轨迹曲率”四类难度；用多头输出更可解释。

**候选信号**：

| 信号 | 来源 | 可能代表 |
|---|---|---|
| SSR residual | PAE | spatial/semantic alignment 难 |
| MCR residual | PAE | manifold boundary / local curvature |
| SCR residual | PAE | semantic ambiguity |
| velocity norm / residual | diffusion training | trajectory 难 |
| attention entropy | transformer | context 不确定 |
| image/latent edge energy | tokenizer/input | 高频细节 |
| local t contrast | sampler | 异步邻域难 |

**具体实现步骤**：

```python
out = model(...)
logvar_sem, logvar_tex, logvar_geom, logvar_traj = out.logvar_heads.chunk(4, dim=-1)
difficulty = (
    w_sem * sigmoid(logvar_sem) +
    w_tex * sigmoid(logvar_tex) +
    w_geom * sigmoid(logvar_geom) +
    w_traj * sigmoid(logvar_traj)
)
```

训练时只有总 NLL 必需；各 head 可用 weak labels bootstrap。

**难度评估**：中高。更多 head 带来调参与解释成本。

**预期效果**：对论文 ablation 与可视化有价值；对 MVP 不必要。

---

#### 方案 3.5：UQ 方法用于校准 adaptive NFE

**一句话原理**：logvar head 是单模型 aleatoric proxy，可用 ensemble/MC dropout/evidential 方法估计 epistemic uncertainty，判断模型真的“不知道”的 patch。

**具体实现步骤**：

- **MC dropout**：只在推理 scheduler 的 uncertainty pass 打开 dropout，重复 K=2/4 次，计算预测方差。
- **light ensemble**：保留 EMA 与 non-EMA 两个模型，或训练两个小 head。
- **evidential head**：输出 Normal-Inverse-Gamma 参数，直接给不确定性分解。

**难度评估**：中。

**预期效果**：可以改善 adaptive scheduler 的可靠性，尤其避免 logvar 被训练 loss hack。

**风险**：增加推理 NFE；对 ImageNet FID 的收益不一定抵消成本。

---

#### 方案 3.6：cross-patch difficulty

**一句话原理**：一个 patch 的难度常由邻域关系决定，例如物体边界、遮挡、重复纹理；difficulty 应包含邻域 t contrast、attention entropy 与 context disagreement。

**具体实现步骤**：

1. 记录 attention entropy：

\[
H_i=-\sum_j A_{ij}\log A_{ij}.
\]

2. 记录邻域时间差：

\[
C_i=\operatorname{mean}_{j\in\mathcal N(i)} |t_i-t_j|.
\]

3. 组合：

\[
D_i=\alpha\sigma_i^2+\beta\hat d_i+\gamma H_i+\eta C_i.
\]

4. 用 `D_i` 而非单独 `σ_i²` 做推理预算分配。

**难度评估**：中。

**风险**：attention entropy 计算/存储成本大；不同层 attention 难解释。

### 3.3 技术顾问建议

1. **推荐 MVP 路径**：原版 logvar head + PAE residual zero-cost baseline + PAE prior bootstrap head。三者必须一起 ablation，否则无法证明 synergy。
2. **推荐终极路径**：multi-signal difficulty score，但模型 head 保持简单；复杂性放在 scheduler/analysis，而非主干。
3. **实施顺序**：logvar baseline → 离线 PAE residual map → prior injection/aux loss → cross-patch signals → UQ。
4. **未来工作**：evidential uncertainty、active learning coreset、难度类型可解释分解；这些适合扩展论文或后续工作。

---

## 4. 探索区域 4：1-NFE 推理与 Patch Forcing Adaptive Sampling 的整合

### 4.1 发现的方案全景

- **方案 4.1：纯 global 1-NFE。** 所有 patch 从 noise 一步到 data，最符合 MeanFlow 卖点。
- **方案 4.2：uniform few-NFE。** 所有 patch 用 2/4/8 NFE，作为质量-速度 Pareto baseline。
- **方案 4.3：difficulty-threshold adaptive NFE（推荐 MVP）。** 易 patch 1 步，难 patch 2–4 步；按 logvar/PAE prior/top-k 分配。
- **方案 4.4：Look-ahead with frozen confident context。** 先生成 confident patches，再作为上下文 refine hard patches。
- **方案 4.5：active-mask asynchronous solver。** 每个 patch 有自己的 remaining time 和 step count，forward pass 只更新 active/hard patches。
- **方案 4.6：learned NFE controller。** 用 RL/bandit/controller 学预算分配；高潜力但不做 MVP。
- **方案 4.7：FD-loss repurpose 到 1-NFE。** 若 FD-loss 足够强，multi-step PAE×HT 模型可直接后训练成 one-step generator。

### 4.2 方案详情

#### 方案 4.1：纯 global 1-NFE

**一句话原理**：在 \(t=1,r=0\) 上一次 forward，所有 patch 同步一步生成。

**具体实现步骤**：

```python
z = torch.randn(B, N, C, device=device)
t_vec = torch.ones(B, N, device=device)
r_vec = torch.zeros(B, N, device=device)
out = model(z, r_vec, t_vec, y)
z0_hat = z - (t_vec - r_vec)[..., None] * out.u
x = pae.decode(z0_hat)
```

**难度评估**：低。

**预期效果**：速度最快，是所有报告指标必须包含的 baseline。

**风险**：hard patch 没有上下文传播机会，可能出现局部伪影、边界不一致。

---

#### 方案 4.2：uniform few-NFE

**一句话原理**：MeanFlow 仍可多步使用，把 \([1,0]\) 拆成 K 个 interval，所有 patch 同步更新。

**具体实现步骤**：

```python
schedule = torch.linspace(1, 0, K+1, device=device)
for k in range(K):
    t = schedule[k].expand(B, N)
    r = schedule[k+1].expand(B, N)
    u = model(z, r, t, y).u
    z = z - (t-r)[..., None] * u
```

**难度评估**：低。

**预期效果**：提供 Pareto 曲线：1/2/4/8 NFE vs FID/FDr/sFID。即使主打 1-NFE，也需要证明 few-NFE 可稳步改善。

**风险**：不是 HT adaptive；若 few-NFE 才好，1-NFE narrative 会弱化。

---

#### 方案 4.3：difficulty-threshold adaptive NFE（推荐 MVP）

**一句话原理**：用 difficulty score 给每个 patch 分配额外 split interval；全局 forward pass 可按 mask 更新 hard patches。

**具体实现步骤**：

1. 第一次 1-NFE 得到 `z0_hat` 与 difficulty：

```python
out = model(z, zeros, ones, y)
z_easy = z - out.u
diff = out.logvar.squeeze(-1)  # or combined score
```

2. 选择 hard patches：

```python
q = torch.quantile(diff, 0.8, dim=1, keepdim=True)
hard = diff > q
```

3. hard patches 用两步或四步更新，easy patches 保持或作为 context：

```python
z_ctx = z_easy.detach()  # confident context
z_work = z.clone()
# first half for hard patches
u1 = model(z_work, mid_vec, one_vec, y, context_override=z_ctx, mask=hard).u
z_mid = torch.where(hard[...,None], z_work - 0.5*u1, z_ctx)
# second half
u2 = model(z_mid, zero_vec, mid_vec, y, context_override=z_ctx, mask=hard).u
z_final = torch.where(hard[...,None], z_mid - 0.5*u2, z_ctx)
```

4. 报告两种 NFE：
   - **global forward NFE**：模型调用次数。
   - **patch-normalized NFE**：\(\frac{1}{N}\sum_i K_i\)。

**涉及组件**：推理 sampler、difficulty head、mask update。

**难度评估**：中。

**预期效果**：在 patch-normalized NFE 1.2–1.8 下接近 uniform 2–4 NFE 质量；对难 patch 伪影有明显改善。

**风险**：如果 full transformer forward 必须处理所有 tokens，wall-clock 速度未必按 patch NFE 下降；但质量/计算预算叙事仍成立。

---

#### 方案 4.4：Look-ahead with frozen confident context

**一句话原理**：让 confident patches 先到 \(t=0\)，冻结为 context；hard patches 在更干净的邻域上下文中重新 denoise。

**具体实现步骤**：

1. 第一步全图 1-NFE，得到 `z0_first` 与 difficulty。
2. easy patch 设为 `z_context=z0_first`，hard patch 保留在 `t=1` 或 `t=0.5`。
3. 第二次 forward 时模型看到 mixed-time latent：easy at 0，hard at high/mid t。
4. hard patch 更新到 0；easy patch 不再改或只做小 residual refinement。

**涉及组件**：HT inference scheduler、attention context、masking。

**难度评估**：中高。关键是 mixed-time context 的训练分布是否覆盖。

**预期效果**：继承 Patch Forcing look-ahead 的核心优势；对物体边界/局部细节可能提升。

**风险**：confident patch 若第一步错了，会把错误传播给 hard patch；需要 confidence calibration。

---

#### 方案 4.5：active-mask asynchronous solver

**一句话原理**：维护每个 patch 的当前时间 \(t_i\)，每次只推进 active patches；这更接近 CFD local time stepping。

**具体实现步骤**：

```python
t_vec = torch.ones(B, N, device=device)
z = torch.randn(B, N, C, device=device)
for global_iter in range(max_iters):
    diff = estimate_difficulty(z, t_vec)
    dt = choose_local_dt(diff, budget)  # [B,N]
    r_vec = (t_vec - dt).clamp_min(0)
    active = dt > 0
    out = model(z, r_vec, t_vec, y)
    z_new = z - dt[..., None] * out.u
    z = torch.where(active[..., None], z_new, z)
    t_vec = torch.where(active, r_vec, t_vec)
    if t_vec.max() == 0: break
```

**难度评估**：高。训练分布必须覆盖这种异步 schedule，否则推理 mismatch 严重。

**预期效果**：理论上最贴近 HT；可实现细粒度 compute allocation。

**风险**：global forward 次数可能增加，batch 并行差；不建议 MVP。

---

#### 方案 4.6：learned NFE controller

**一句话原理**：把 NFE 分配看作 budgeted decision problem，controller 根据 difficulty map、class、当前 t 分配 patch steps。

**实现草案**：

- 状态：`difficulty map`, `t_vec`, `attention entropy`, `PAE prior`。
- 动作：每 patch `0/1/2 extra steps` 或 top-k hard set。
- reward：`-FID_proxy - λ*NFE`，用 small-sample feature FD/CLIP-FD 作 proxy。
- 训练：先 imitation top-k heuristic，再 REINFORCE/bandit fine-tune。

**难度评估**：高。

**建议**：仅 future work；当前项目用 heuristic 足够。

---

#### 方案 4.7：FD-loss repurpose 到 1-NFE

**一句话原理**：先训练 multi-step 或 few-step generator，再用 FD-loss 直接优化 1-NFE 输出分布，使模型适应一步推理。

**对接方式**：

- Base generator 可是 `PAE×HT×FM`，不一定是 MeanFlow。
- FD post-training 的 sampling args 固定为 1-NFE。
- 若 FD-loss 成功，MeanFlow 的“从头 one-step”必要性下降。

**难度评估**：中。

**风险**：指标可能好但视觉/holdout 差；详见第 6 节。

### 4.3 技术顾问建议

1. **推荐 MVP 路径**：global 1-NFE、uniform 2/4/8 NFE、threshold adaptive 1/2/4 NFE 三条曲线同时做。没有 Pareto 曲线，单个 FID 数字说服力不足。
2. **推荐终极路径**：difficulty-threshold adaptive + look-ahead frozen context；若训练分布覆盖 mixed-time states，可作为 PDM-3-HT 区别于普通 MeanFlow 的亮点。
3. **实施顺序**：纯 1-NFE → uniform few-NFE → top-k adaptive → look-ahead → active-mask asynchronous。
4. **未来工作**：RL controller、CFD-style local error estimator、Richardson extrapolation / Adams-Bashforth correction。这些会增加复杂度，不应进入主论文关键路径。

---

## 5. 探索区域 5：Proposition 1 的 Per-Patch 推广与 JVP 稳定性

### 5.1 发现的方案全景

- **方案 5.1：直接沿用 v1 Proposition。** 简单但不严谨；只能作为直觉背景。
- **方案 5.2：条件性 per-patch theorem（推荐）。** 明确添加 attention/time embedding/network norm/LTG contrast 等假设。
- **方案 5.3：反例与边界条件。** 证明 MCR alone 不足，避免论文过度承诺。
- **方案 5.4：数值稳定性诊断套件。** JVP norm、cross_ratio、attention spectral proxy、NaN、max LR 等。
- **方案 5.5：稳定化工程。** fp32 JVP island、spectral/QK norm control、local attention JVP、time contrast regularization。
- **方案 5.6：若失败的 fallback。** JVP-free SplitMeanFlow / Flow Matching + FD-loss。

### 5.2 方案详情

#### 方案 5.1：直接沿用 v1 Proposition

**一句话原理**：把 PAE-MCR ⇒ latent geometry smoother ⇒ score/velocity smoother ⇒ MeanFlow JVP stable 的链条直接搬到 per-patch。

**问题**：per-patch setting 中 score 是 multi-time marginal \(p_{\boldsymbol t}(z)\)，velocity \(v_i\) 依赖所有 patch 状态与时间；attention 的 Jacobian 不受 PAE-MCR 直接约束。因此这条证明在 reviewer 面前站不住。

**建议用途**：作为 motivation，不作为 theorem。

---

#### 方案 5.2：条件性 per-patch theorem（推荐）

**一句话原理**：把 Proposition 1 改写为“在若干可监控/可正则化的条件下，PAE-MCR 降低 JVP 上界中的 latent geometry 项”，而不是声称 MCR 单独保证稳定。

**可能的定理陈述**：

> **Proposition 1-HT（条件性版本）**：设 PAE decoder 在数据流形邻域内局部 bi-Lipschitz，PAE-MCR 使 latent density 的局部条件数 \(\kappa_z\) 有界；设 transformer velocity field \(u_\theta(z,\boldsymbol r,\boldsymbol t)\) 关于 latent tokens 与 time embeddings 是 Lipschitz 的，常数分别为 \(L_z,L_t\)，且 attention operator 的 Jacobian spectral norm 有界 \(\|J_{attn}\|_2\le A\)。若 LTG sampler 保证 \(\boldsymbol t\in[\epsilon,1-\epsilon]^N\) 且局部时间差 \(\max_{(i,j)\in E}|t_i-t_j|\le C_t\)，则 full-bundle MeanFlow JVP 满足
>
> \[
> \left\|\frac{dU_\theta}{dc}\right\|
> \le
> \Big(L_z\,\|\boldsymbol\Delta\odot v\| + L_t\,\|\boldsymbol\Delta\|\Big)\,\Phi(A,C_t,\kappa_z),
> \]
>
> 其中 \(\Phi\) 随 PAE latent geometry 条件数与 attention Lipschitz 常数单调增加。PAE-MCR 通过降低 \(\kappa_z\) 与局部 velocity roughness 降低该上界，但若 \(A\) 或 \(C_t\) 无界，则稳定性不保证。

**关键点**：

- theorem 中必须有 attention/network 项。
- MCR 是“降低上界的一个因子”，不是唯一条件。
- LTG local contrast 也进入 bound，连接第 2 节 sampler。

**具体实现 / 论文写法**：

1. 主文给简化 theorem。
2. Appendix 给 proof sketch：
   - PAE-MCR → local latent perturbation smoothness。
   - time-vector score/velocity Lipschitz 分解。
   - self-attention Lipschitz bound 作为假设或 lemma。
   - chain rule 得到 JVP bound。
3. 实验验证 theorem 的三个 measurable proxies：
   - PAE vs VA-VAE 的 JVP norm 分布。
   - attention spectral proxy 是否与 JVP 爆炸相关。
   - local t contrast 是否提高 cross_ratio。

**难度评估**：中高。理论证明需要谨慎限定。

**预期效果**：能保住理论 contribution，同时避免过度 claim。

---

#### 方案 5.3：反例与边界条件

**一句话原理**：构造满足 PAE latent 平滑但 attention 对 time perturbation 极敏感的模型，说明 MCR alone 不足。

**反例草案**：

- 两个 patch，latent 分布平滑，decoder Lipschitz。
- Attention logits：

\[
 A_{12}=\operatorname{softmax}(\alpha(t_2-t_1))
\]

当 \(\alpha\to\infty\) 时，\(t_2-t_1\) 的微小变化导致 attention 从 patch 1 切换到 patch 2。

- 即使 \(z\) manifold 很平滑，\(\partial A/\partial t\sim \alpha\) 可任意大，因此 \(\partial u_i/\partial t_j\) 爆炸。

**论文价值**：

- 主动承认 theorem 条件边界，反而增强可信度。
- 为 sampler 的 local contrast regularization 与 attention norm control 提供动机。

---

#### 方案 5.4：数值稳定性诊断套件

**一句话原理**：把“JVP 稳定”变成一组可量化、可对比、可放进论文表格的指标。

**必测指标**：

| 指标 | 定义 | 目的 |
|---|---|---|
| `jvp_norm_p50/p95/p99/p999` | \(\|du/dc\|\) 分位数 | 检查 tail explosion |
| `target_norm` | \(\|v-du/dc\|\) | 判断 target 方差 |
| `cross_ratio` | \(\|J_full-J_diag\|/\|J_full\|\) | 衡量 cross-patch 耦合 |
| FD finite-diff rel err | \(\|JVP-FD\|/\|FD\|\) | 查 AD/公式 bug |
| NaN/Inf frequency | 每 1K step 次数 | 训练稳定性 |
| max stable LR | 不发散最大 lr | 验证 Proposition 1 推论 |
| attention spectral proxy | power iteration 估计 | 验证 attention 风险 |
| local t contrast correlation | corr(contrast, jvp_norm) | 验证 LTG/sampler 条件 |

**实现建议**：

- 每 500–1000 step 只对小 batch 计算 full diagnostics，避免拖慢主训。
- 保存 heatmap：JVP norm over spatial map、difficulty over map、t over map。
- 分 tokenizer 对比：PAE、SD-VAE、VA-VAE。

---

#### 方案 5.5：稳定化工程

**一句话原理**：把 JVP path 当作数值敏感模块单独保护。

**具体措施**：

1. **fp32 JVP island**：JVP forward 禁用 bf16 autocast。
2. **gradient clipping**：global norm + per-parameter norm；记录 clipping rate。
3. **QK norm / attention temperature 控制**：限制 softmax 过尖，降低 time perturbation 引起的 attention switch。
4. **time embedding smoothing**：避免高频 sinusoidal/time MLP 导致 \(\partial u/\partial t\) 大。
5. **local time contrast regularization**：sampler 层面限制邻域 \(|t_i-t_j|\)。
6. **JVP path fallback kernels**：若 flash/fused attention 不支持 forward-mode，使用 PyTorch eager/SDPA math kernel。
7. **r/t curriculum**：从短 interval 开始，逐渐增加 long interval。
8. **loss clipping**：对 `u_tgt` 或 `du_dc` 做 percentile clipping 作为安全阀，需 ablation。

**难度评估**：中。

**预期效果**：显著降低 NaN 与 JVP tail；可能轻微牺牲速度。

---

#### 方案 5.6：fallback 路径

若 JVP 稳定性 gate 失败，不应让整个 PDM-3-HT 死亡。推荐 fallback 顺序：

1. **Full-bundle → local/stop-context JVP。** 保留部分 MeanFlow novelty。
2. **MeanFlow → SplitMeanFlow / interval consistency。** 去掉 JVP。
3. **MeanFlow → standard Flow Matching + FD-loss。** 工程最稳、仍可冲指标。
4. **HT adaptive few-NFE 而非 1-NFE。** 若 1-NFE 质量差，定位改为 few-NFE high-quality。

### 5.3 技术顾问建议

1. **推荐 MVP 路径**：条件性 theorem + 反例 + 诊断套件。不要试图证明 MCR alone。
2. **推荐终极路径**：把 theorem、sampler local contrast、attention spectral proxy、JVP telemetry 串成完整闭环。
3. **实施顺序**：toy counterexample → PAE/VA-VAE JVP norm 对比 → attention spectral proxy → LR/NaN experiments → theorem 写作。
4. **未来工作**：严格 self-attention Lipschitz theory、random matrix bound、PDE nonlocal stability 类比；本篇只需 proof sketch 与实证验证。

---

## 6. 探索区域 6：FD-loss / Representation Fréchet Loss 与 PDM-3-HT 的整合策略

### 6.1 发现的方案全景

- **方案 6.1：策略 A，简单叠加。** 先完成 PAE×HT×MeanFlow，再 FD-loss post-train。
- **方案 6.2：策略 B，绕过 MeanFlow。** 训练 PAE×HT×Flow Matching，再用 FD-loss repurpose 到 1-NFE。
- **方案 6.3：策略 C，双轨制。** MeanFlow 与 Flow Matching 两条都做，分别 FD-loss。
- **方案 6.4：策略 D，FD-first staged dual-track（推荐）。** Flow Matching 主线 + MeanFlow 风险分支 + FD-loss 前置闸门与决策 gate。
- **方案 6.5：多 representation、部分 hold-out 的 FD-loss。** 训练 representation 与评测 representation 故意错开，防止 PAE+DINOv2 双重锁定。
- **方案 6.6：queue vs EMA 选择。** queue 更接近 FID population，EMA 更稳；二者都做 ablation。
- **方案 6.7：crop/multiscale FD，而非 per-patch FD。** patch-level FD 不自然，建议用 image crops 或 decoded multiscale crops。
- **方案 6.8：FD-loss 与 difficulty head 融合。** 用 difficulty 选择 crop 或加权 FD，而不是直接对 latent patch 做 FD。
- **方案 6.9：FD-loss 的 metric-gaming 防御。** hold-out reps、precision/recall、sFID、nearest-neighbor、人评网格。

### 6.2 方案详情

#### 方案 6.1：策略 A，简单叠加

**一句话原理**：保持 v2 路线不变，把 FD-loss 作为 Phase 3.5 后训练提升指标。

**具体实现步骤**：

1. 完成 `PAE×HT×MeanFlow` base model。
2. 固定 1-NFE sampling args。
3. 解码 PAE latent 到 image space。
4. 用 frozen feature extractors 计算 generated features。
5. 与 reference statistics 计算 differentiable FD loss。
6. 小学习率 fine-tune 10K–50K iter。

**难度评估**：低到中。

**预期效果**：若 base 1-NFE 已足够好，FD-loss 可进一步压 FID/FDr。

**风险**：如果等到 Week 12 才接 FD-loss，任何 repo/feature/reference stats 问题都会太晚暴露；且若 MeanFlow 失败，策略 A 没有主线 fallback。

---

#### 方案 6.2：策略 B，绕过 MeanFlow

**一句话原理**：承认 FD-loss 可以把 multi-step generator repurpose 到 1-step，因此先做更稳的 Flow Matching，不碰 per-patch JVP。

**具体实现步骤**：

1. 训练 `PAE×HT×Flow Matching`，用 Euler/ODE sampler 得到高质量 multi-step。
2. 固定 post-training sampler 为 1-NFE 或 2-NFE。
3. 用 FD-loss 直接优化该 sampler 输出。
4. 与 MeanFlow branch 对比 base/post-train 指标。

**难度评估**：中。

**预期效果**：工程成功率最高；很可能以最少风险达到漂亮 FID。

**风险**：理论 contribution 削弱；reviewer 可能认为只是“把已有组件 + FD-loss 拼起来”。需要用 HT/PAE ablation 证明新意。

---

#### 方案 6.3：策略 C，双轨制

**一句话原理**：MeanFlow 与 Flow Matching 两条完整训练路线都做，分别 FD-loss post-train，最大化论文信息量。

**难度评估**：高。算力与时间压力大。

**适用条件**：只有当 Phase 0/1 复现非常顺利、B300 窗口稳定、代码栈成熟时才考虑。

**风险**：两条都做半成品是最坏情况。

---

#### 方案 6.4：策略 D，FD-first staged dual-track（推荐）

**一句话原理**：不是简单 A/B/C 三选一，而是把 FD-loss 提前变成项目 gate；Flow Matching 保底，MeanFlow 按稳定性逐级晋级。

**推荐流程**：

| 阶段 | 主线 | MeanFlow 分支 | FD-loss |
|---|---|---|---|
| Week 1–2 | 复现 PAE/HT/FM baseline | toy full-bundle JVP | 复现 FD-loss repo，小模型跑通 |
| Week 3–5 | PAE×HT×FM | DiT-Tiny/B JVP gate | 对 FM baseline 做短 post-train |
| Week 6–7 | FM 扩大规模 | 若 gate 通过，接 Per-Patch MeanFlow | 确定 reps/queue/ref stats |
| Week 8–11 | B300 主训：优先最稳路线 | MeanFlow 只在 gate 通过后用 burst | 保持 post-train pipeline ready |
| Week 12–13 | — | — | 对最佳 checkpoint FD post-train |

**决策 gate**：

- 若 MeanFlow 分支出现 `jvp_p99/target_norm > 10`、NaN 高、max LR 低于 FM 50% 以上，则不进入大规模。
- 若 FM+FD-loss 已达到目标并 hold-out 稳定，MeanFlow 降级为理论/ablation。
- 若 FD-loss 出现严重 hold-out overfit，则强调 base/few-NFE 质量，MeanFlow 分支继续。

**难度评估**：中。不是多做一整条路线，而是早期小成本并行 gate。

**预期效果**：最大化项目成功概率；既保留理论 upside，又避免被 JVP 风险拖死。

**这是本报告的明确推荐。**

---

#### 方案 6.5：多 representation、部分 hold-out 的 FD-loss

**一句话原理**：FD-loss 本质是 representation-space moment matching；如果训练和评测用同一 representation，容易被质疑 metric gaming。PDM-3-HT 尤其要避免 DINOv2 双重锁定，因为 PAE 已用 DINOv2/VFM 对齐。

**具体实现步骤**：

1. 将 representations 分为三组：
   - **train judges**：参与 FD-loss。
   - **validation judges**：调参可看但不反向。
   - **hold-out judges**：最终报告，训练期间不看或极少看。
2. 若 PAE 使用 DINOv2，对 DINOv2 FD-loss 降权或作为 hold-out。
3. 推荐初始组合：
   - train：Inception + CLIP/SigLIP + MAE/ConvNeXt。
   - hold-out：DINOv2、RADIO、EVA-CLIP、SigLIP2、不同 MAE/ViT 变体。
4. 报告 `FDr_train` 与 `FDr_holdout` 的 gap。

**重要核验点**：任务书称 FDr⁶ 使用 Inception、SigLIP2、MAE、DINOv2、CLIP、RADIO；公开 FD-loss GitHub 脚本/源码抽样显示支持 Inception、ConvNeXt、DINOv2、MAE、SigLIP、CLIP 等。**representation 名称与最终评测协议需在 Phase 0 以论文最终版 / 官方脚本确认。**

**难度评估**：中。主要成本是 feature extractor 显存与 reference stats 准备。

**风险**：多 representation loss 梯度冲突；可用权重归一化、GradNorm/PCGrad/CAGrad。

---

#### 方案 6.6：queue vs EMA estimator

**一句话原理**：FD-loss 通过 queue 或 EMA 解耦 population size 与 batch size；queue 更接近真实 FD，EMA 更平滑。

**源码核验要点**：公开 FD-loss 仓库中可见：

- `FeatureQueue` 支持 circular queue 与 online/EMA stats。
- `compute_frechet_distance_loss` 可从 raw features 或 `(mu, sigma)` 计算 differentiable FD。
- `diff_all_gather` 保留当前 rank 的梯度。
- 训练 step 中生成图像、提取 judge features、计算 normalized FD loss、反向，再把 `new_feats.detach()` 入队。
- repo 中 feature extractors 支持 Inception、ConvNeXt 以及 timm 模型（如 DINOv2、CLIP、MAE、SigLIP 等）。

**推荐实现**：

```python
sampled_z = model.sample_images_with_grad(z_noise, y, sampling_args)
sampled_x = pae.decode(sampled_z)
sampled_x = sampled_x.clamp(0, 1)

loss_fd = 0.0
for judge in judges:
    feats = judge.extract(sampled_x)          # gradients to image/model
    feats_all = diff_all_gather(feats)
    if judge.use_queue:
        all_feats = judge.queue.build_feats_snapshot(feats_all)
        fd = frechet_loss(judge.mu_ref, judge.sigma_ref, all_feats=all_feats)
    else:
        mu, sigma = judge.queue.build_feats_stats(feats_all)
        fd = frechet_loss(judge.mu_ref, judge.sigma_ref, mu=mu, sigma=sigma)
    loss_fd = loss_fd + judge.weight * fd / (fd.detach() + eps)

loss_total = loss_anchor + lambda_fd * loss_fd
loss_total.backward()
for judge in judges:
    judge.queue.enqueue(feats_all.detach())
```

**选择建议**：

- MVP：queue-based estimator，population 50K，与论文设置更一致。
- Ablation：EMA estimator，看是否更稳、显存更省。
- Pro 6000 96GB 下，50K×feature_dim queue 通常可接受，但多 judge 同时启用需检查 CPU/GPU buffer 放置。

**风险**：queue 早期样本质量差会污染 population；可 warmup 后启用 FD-loss，或 queue reset。

---

#### 方案 6.7：crop/multiscale FD，而非 per-patch FD

**一句话原理**：patch latent 不是自然图像，直接 per-patch FD 与 ImageNet representation 不匹配；更合理的是 decoded image crop/multiscale FD。

**具体实现步骤**：

1. 从 decoded images 中按 difficulty map 采 crops：
   - hard patch 周围 64×64/128×128 crop。
   - random crop 作为对照。
2. 用同一 representation 或轻量 crop encoder 计算 crop FD。
3. Loss：

\[
L=L_{FD}^{\text{global}}+\lambda_{crop}L_{FD}^{\text{crop}}.
\]

4. crop reference stats 从真实图像同样 crop protocol 离线计算。

**难度评估**：中高。

**预期效果**：可能改善局部纹理与 hard patch；比 per-patch FD 更符合视觉 representation。

**风险**：crop FD 可能牺牲全局布局；不做 MVP。

---

#### 方案 6.8：FD-loss × difficulty head 协同

**一句话原理**：difficulty head 给出局部不确定性，FD-loss 给分布级信号；可用 difficulty 引导哪些 crop/样本进入 FD-loss，而不是简单全图平均。

**实现草案**：

- sample weighting：高 difficulty 图像或 crop 在 FD-loss 中更高概率出现。
- representation weighting：语义难度高用 CLIP/SigLIP，纹理难度高用 Inception/ConvNeXt。
- schedule：先 global FD，后期加 difficulty-guided crops。

**难度评估**：中。

**建议**：作为 Phase 4 ablation，不进入首个 FD-loss MVP。

---

#### 方案 6.9：FD-loss 的 metric-gaming 防御

**一句话原理**：FD-loss 优化的是 representation moments，不必然等价于真实视觉质量；必须主动设计反过拟合评测。

**必做 guardrails**：

1. **hold-out representation FDr**：未参与训练的 backbones。
2. **sFID / precision / recall / density / coverage**：避免只优化全局 feature moments。
3. **class accuracy / classifier consistency**：class-cond ImageNet 必测。
4. **nearest-neighbor memorization**：对训练集最近邻可视化与距离统计。
5. **diversity metrics**：同 class 多样性、intra-class covariance。
6. **qualitative grids**：固定 seed、随机 seed、hard classes、failure cases。
7. **FD-optimized vs non-FD base 分开报告**：不要混淆 base model 与 post-training gain。
8. **FD-loss 权重与训练时长 sweep**：展示过训练会不会 hold-out 下降。

**论文 framing**：

- 诚实承认：FID 接近或低于 validation reference 时，单一 FID 已接近 noise floor。
- 把 FD-loss 作为“metric-aware distributional post-training”，不是宣称视觉质量已由 FID 单独证明。
- 强调 hold-out reps 与人类可视化支撑。

### 6.3 技术顾问建议

1. **明确推荐：策略 D，而非 A/B/C。** A 太晚暴露 FD 风险；B 太早放弃理论贡献；C 资源压力过大。D 用 Flow Matching 保底、MeanFlow gate、FD-loss 前置，风险收益最好。
2. **MVP 路径**：先复现 FD-loss queue + 单 representation；再做 2–3 个 train judges；最后加入 hold-out FDr。不要一开始做 per-patch/crop FD。
3. **终极路径**：多 representation FD-loss + hold-out reps + difficulty-guided crop FD ablation + base/post-train 双报告。
4. **组件 redundancy 判断**：FD-loss 可能让 MeanFlow 的“必须性”下降，但不会让 PAE 与 HT 自动冗余。PAE 仍提供 latent geometry 与 tokenizer 质量；HT 仍提供 spatial heterogeneity 与 adaptive inference。真正可能降级的是“复杂 per-patch JVP 作为主线”的必要性。

---

## 7. 意外发现：任务书未覆盖的新方向

### 7.1 发现的方向全景

- **方向 7.1：把 MeanFlow 从主目标降级为 control variate / consistency regularizer。** 不一定用 MeanFlow 直接生成，而是用它改善 FM 的 long-interval consistency。
- **方向 7.2：coarse-to-fine / multigrid PDM。** 用低分辨率全局语义 trajectory + 高分辨率 patch local trajectory，借鉴 multigrid。
- **方向 7.3：动态 patchification / adaptive token budget。** HT 只改变时间，不改变空间 token 粒度；真正的 spatial adaptivity 可能要动态 token merge/split。
- **方向 7.4：Consistency / Shortcut + FD-loss 可能是更强工程路线。** 若目标是 1-NFE 指标，而非 MeanFlow 理论，JVP-free one-step 目标更稳。
- **方向 7.5：ImageNet 256 指标已接近评测噪声，论文需要“指标防御层”。** FID <1 时，评测可信度可能比模型本身更受挑战。
- **方向 7.6：PAE 与 FD-loss 的 representation 过拟合闭环。** PAE 用 VFM 对齐，FD-loss 也用 VFM judge；这既是 synergy，也可能是 reviewer 攻击点。
- **方向 7.7：先训 strong few-NFE，不执念 1-NFE。** 若 1-NFE 牺牲视觉质量，few-NFE high-quality + FD-loss 可能更有实际价值。

### 7.2 方向详情

#### 方向 7.1：MeanFlow as control variate / consistency regularizer

**核心机制**：不要求模型主输出严格满足 MeanFlow 1-NFE，而是在标准 Flow Matching 中加入 long-interval consistency term，作为 regularizer 降低 few-step 与 one-step 差距。

**实现**：

\[
L=L_{FM}+\lambda_{long}\|z_r^{one}-\operatorname{sg}(z_r^{multi})\|^2.
\]

其中 `multi` 可用 teacher-free split 或 EMA teacher。这样保留 MeanFlow 的平均速度思想，但不需要 full JVP。

**价值**：若 JVP 失败，这个方向能保留部分理论叙事：PDM-3-HT 学的是 spatially heterogeneous long-interval transport。

---

#### 方向 7.2：coarse-to-fine / multigrid PDM

**核心机制**：多重网格思想认为低频全局误差需要 coarse grid，高频局部误差需要 fine grid。PDM-3-HT 当前所有 token 同一尺度，可能低效。

**实现草案**：

1. PAE latent 32×32。
2. 建立 16×16 或 8×8 pooled latent 作为 global grid。
3. coarse model 预测全局 semantic velocity；fine model 预测 residual velocity。
4. HT difficulty 只在 fine residual 上分配 extra NFE。

**风险**：结构侵入性较大，接近新模型；建议 future work。

---

#### 方向 7.3：动态 patchification / adaptive token budget

**核心机制**：HT 给 hard patch 更多时间步，但 transformer 每次仍处理同样 tokens。对 wall-clock 来说，真正省算力需要 token pruning/merging/splitting。

**实现草案**：

- easy background tokens merge 成 super-token。
- hard boundary tokens split 或保留高分辨率。
- 用 difficulty map 决定 token layout。

**价值**：对实际推理速度比 patch NFE 更关键。

**风险**：改变 positional encoding 与 PAE latent grid，非 MVP。

---

#### 方向 7.4：Consistency / Shortcut + FD-loss 作为替代主线

**核心机制**：如果 FD-loss 能把任意 base generator 推向 1-NFE，主线目标应是“给 FD-loss 一个好初始化”，而不是必须从头 one-step。Consistency/shortcut 训练可能比 MeanFlow JVP 更稳。

**实施建议**：在 Phase 1 加一个 tiny shortcut baseline：`PAE×HT×FM` + interval consistency。若表现接近 MeanFlow 且稳定性更好，可作为 Plan B 主线。

---

#### 方向 7.5：指标防御层

**核心机制**：当 FID 目标 ≤0.7，reviewer 的问题会从“质量够不够好”转向“指标是否可信”。因此评测设计是 contribution 的一部分。

**建议**：

- 报告 FID confidence interval / bootstrap。
- 报告 validation-vs-train reference gap。
- 使用多 seed 50K samples。
- 固定公开 evaluation script。
- 给出 FD-loss 过训练曲线：train judge 降、holdout judge 是否升。

---

#### 方向 7.6：PAE × FD-loss 的 representation closed loop

**核心机制**：PAE 用 DINOv2/VFM 语义 prior 改造 latent；FD-loss 又可能用 DINOv2/SigLIP/CLIP 等 representation 优化输出。优点是 representation alignment 一致；缺点是模型可能学习“让这些网络开心”的图像统计。

**建议**：将此作为论文讨论亮点：

- 正面：representation-aligned latent 与 representation-level distribution matching 协同。
- 反面：必须用 disjoint/hold-out reps 防止闭环过拟合。

---

#### 方向 7.7：不要执念 1-NFE

**核心机制**：1-NFE 是漂亮卖点，但如果 2–4 NFE 质量显著更好、速度仍快，实际价值可能更高。

**建议**：论文标题/叙事不要只押“one-step”；应写成“spatially adaptive few-step/one-step generation”。指标表同时突出：

- 1-NFE best。
- patch-normalized 1.5 NFE。
- uniform 4 NFE high-quality。

---

## 8. 最终整体建议

### 8.1 对 PDM-3-HT / Path C 的整体判断

**信心评级：中高。**

原因：

- 相比 v1 DDT，HT 与 LightningDiT 的工程改动小得多，避免了 encoder/decoder 与 per-patch timestep 的结构冲突。
- PAE 与 HT 有自然协同：PAE 提供 patch-level latent geometry 与 residual prior，HT 使用 patch-level time/difficulty。
- MeanFlow 若成功，能提供强理论与 1-NFE 卖点。
- FD-loss 给了项目强保底与 SOTA 指标上限。

**最大 dealbreaker**：

1. **Full-bundle JVP 不稳定或不可扩展。** attention cross terms 可能让 MeanFlow 分支无法主训。
2. **FD-loss 被认为是 representation gaming。** 即使 FID 0.7，若 hold-out FDr/视觉质量不支持，论文会被质疑。

**reviewer 最可能 challenge 的点**：

- Per-Patch MeanFlow identity 是否数学正确？是否漏掉 cross-patch terms？
- PAE-MCR 是否真的保证 JVP 稳定，还是只是 post-hoc 相关？
- FD-loss 后的 SOTA 是否只是优化评测 representation？
- 三组件叠加的贡献是否可分离？是否每个组件都有必要？
- 算力预算下 ablation 是否足够完整？

### 8.2 FD-loss 整合的明确建议

**推荐策略：全新策略 D：FD-first staged dual-track。**

- Flow Matching 是工程主线，保证项目能产出高质量 base。
- Per-Patch MeanFlow 是理论高风险分支，必须 gate。
- FD-loss 前置到 Phase 0/1，作为 pipeline 与指标闸门。

**为什么不是 A/B/C**：

- A：太保守且太晚接 FD-loss，无法应对 MeanFlow 失败。
- B：成功率高但过早放弃理论贡献，论文新意可能不足。
- C：信息量最大但资源消耗大，容易两线都做不深。
- D：保留 C 的 upside，同时用 gate 限制成本。

**FD-loss 是否让部分组件 redundant？**

- **MeanFlow：部分 redundant。** 如果 FD-loss 能把 FM multi-step repurpose 到 1-NFE，则 MeanFlow 的工程必要性下降；但理论贡献仍有价值。
- **PAE：不 redundant。** PAE 影响 base latent geometry、训练稳定性、tokenizer 上限；FD-loss 只能后训练输出分布，不能替代好 latent。
- **HT：不 redundant。** HT 提供 spatial heterogeneity 与 adaptive sampling，是与 FD-loss 不同层级的解耦。
- **MCR：可能需要重新定位。** 不应只说 MCR 是 JVP 稳定保证；也应说它是 latent geometry 与 FD/representation alignment 的先验。

### 8.3 v1 / v2 / v2.1 路线比较

| 版本 | 核心 | 优点 | 主要风险 | risk-adjusted return |
|---|---|---|---|---|
| v1 PDM-3 | PAE × DDT × MeanFlow | “三层解耦”故事完整；DDT 有结构 novelty | DDT 与 HT/MeanFlow 不兼容；encoder/decoder + JVP 复杂；工程重 | 中低 |
| v2 PDM-3-HT | PAE × HT × MeanFlow | HT 实证收益高；改动小；per-patch story 强 | Per-Patch MeanFlow 数学/JVP 风险高 | 中高 |
| v2.1 PDM-3-HT+FD | PAE × HT × MeanFlow/FM × FD-loss | 指标上限最高；有 FD 保底；可规避部分 JVP 风险 | representation gaming；组件贡献归因复杂 | 最高，但需严谨评测 |

**结论**：v2.1 的 risk-adjusted return 最高，但必须把路线从“MeanFlow 单主线”改为“FM 保底 + MeanFlow gate + FD-loss 前置”。

### 8.4 如果只能给三条建议

1. **先实现并验证 full-bundle JVP，不要把 diagonal per-patch 公式当严格理论。** 用 toy/finite-difference/cross_ratio 做 gate。
2. **把 FD-loss 提前到 Phase 0/1 跑通，并设计 hold-out representation 防御。** 不要等 base model 完成后才接入。
3. **保持 `PAE×HT×Flow Matching` 为主线保底，MeanFlow 只在稳定性通过后扩大。** 这能最大化项目成功概率。

### 8.5 如果 Proposition 1 三个推论全部失败：Plan B / Plan C

**失败定义**：PAE 没有比 SD-VAE/VA-VAE 更低 JVP norm、NaN 频率不低、max stable LR 不高。

#### Plan B：PAE × HT × Flow Matching + FD-loss

- 放弃“PAE-MCR 保证 MeanFlow JVP”的强 claim。
- 保留 PAE/HT 的 empirical contribution。
- 用 FD-loss 将 strong multi-step/few-step FM generator repurpose 到 1-NFE。
- 理论改为：PAE geometry improves latent diffusion and HT difficulty calibration，不再绑定 JVP theorem。

#### Plan C：HT adaptive few-NFE + JVP-free consistency

- 如果 FD-loss hold-out 也不稳，则主打 base model 质量与 adaptive few-NFE。
- 使用 SplitMeanFlow/consistency/shortcut 代替 JVP。
- 指标定位从“1-NFE FID 0.7”调整为“few-NFE high-quality + robust FDr + speed-quality Pareto”。

#### Plan D-negative：回退到 PAE×REPA-E/LightningDiT 强 baseline

- 如果 HT 与 PAE 也边际收益小，保留 PAE + LightningDiT + FD-loss 作为最稳可投稿 baseline。
- 论文主题转向 representation-aligned tokenizer 与 distribution-level post-training。

### 8.6 FD-loss 的元批判与论文 framing

**FD-loss 的 0.72 是否代表真实 visual quality 进步？**

答案应谨慎：**它代表在特定 representation Fréchet metrics 上的显著进步，但不自动等价于真实视觉质量全面进步。**FD-loss 直接优化 representation moments；当评测也使用相同或相近 representation 时，存在 reward hacking / representation overfitting 风险。尤其 PAE 已经使用 VFM 对齐，若 FD-loss 再使用相同 VFM，闭环风险更高。

**如何避免 reviewer 用 representation gaming reject？**

1. **base 与 FD-post 分开报告**：清楚说明哪些结果是普通生成训练得到，哪些是 FD-loss 后训练得到。
2. **hold-out reps**：最终表格必须有未参与 FD-loss 的 representation FDr。
3. **多指标**：FID、FDr⁶、sFID、precision/recall、density/coverage、class accuracy、nearest-neighbor、diversity。
4. **过训练曲线**：展示 train FD 降低时 hold-out 是否恶化。
5. **定性失败案例**：主动展示 FD-loss 后的 artifacts 与无 FD base 对比。
6. **诚实措辞**：把 FD-loss 称为 distribution-level post-training，不说“FID 0.7 因此视觉质量绝对超过所有模型”。
7. **representation disjoint design**：PAE 用 DINOv2 时，FD-loss train judges 不应完全依赖 DINOv2。

---

## 9. 更新版 16–18 周执行计划

### Phase 0：环境、复现与 gate 设计（Week 1–2）

**目标**：不要只复现 baseline，还要把高风险 pipeline 提前打通。

- 复现 LightningDiT/SiT/PAE/HT/MeanFlow 最小配置。
- 跑通 FD-loss repo：queue、reference stats、feature extractors、1-NFE sampling with grad。
- 写 toy full-bundle JVP harness。
- 确认 fused attention 与 `torch.func.jvp` 兼容性。
- 建立 telemetry logger：JVP norm、cross_ratio、t stats、difficulty calibration、FD holdout。

**Gate**：FD-loss 能在小 generator 上反向；JVP finite-difference rel error < 5–10%。

### Phase 1：三组件独立验证（Week 3–5）

- `PAE×Flow Matching` baseline。
- `PAE×HT×Flow Matching` baseline。
- `PAE×MeanFlow` scalar/global baseline。
- `PAE×HT` difficulty head + PAE prior ablation。
- FD-loss 对 FM baseline 的短 post-train。

**Gate**：HT 在 PAE latent 上有稳定收益；FD-loss 不出现明显 hold-out collapse。

### Phase 2：Per-Patch MeanFlow 推导与实现（Week 6–7）

- full-bundle JVP training step。
- diagonal/local/stop-context ablation。
- r/t curriculum。
- PAE vs VA-VAE/SD-VAE JVP stability 对比。

**Gate**：JVP p99 可控、NaN 低、max LR 不显著劣于 FM；否则切 Plan B。

### Phase 3：主训（Week 8–11，B300 burst）

- 若 MeanFlow gate 通过：训练 `PAE×HT×MeanFlow`。
- 若 gate 不通过：训练 `PAE×HT×Flow Matching` 强主线。
- 同步进行 sampler/difficulty 最小 ablation。

**Gate**：base no-FD 指标与 Pareto 曲线达到可投稿阈值。

### Phase 3.5：FD-loss post-training（Week 12–13）

- Queue-based FD-loss first。
- Multi-rep train judges + hold-out reps。
- λ_fd、LR、training length sweep。
- 1-NFE 与 adaptive few-NFE 两种 sampling args 都试。

**Gate**：FID/FDr 提升不以 hold-out 崩坏为代价。

### Phase 4：理论与 ablation（Week 14–15）

- Proposition 1-HT 三推论。
- attention spectral proxy / local contrast / cross_ratio。
- PAE residual vs difficulty head。
- FD-loss representation overfit ablation。
- v1/v2/v2.1 组件归因表。

### Phase 5：写作（Week 16–18）

- 主文：方法、theorem、实验主表、FD-loss 元讨论。
- Appendix：full-bundle derivation、toy proof harness、FD implementation、更多 grids。
- 准备 reviewer 质疑回答：identity、JVP、FD gaming、组件必要性。

---

## 10. 工程检查清单

### 10.1 Per-Patch MeanFlow checklist

- [ ] `model.forward` 支持 per-token `r_vec/t_vec`。
- [ ] full-bundle JVP tangent 为 `(Δ*v, 0, Δ)` 或 `(Δ*v, Δ, Δt)`，没有重复乘 \(\Delta\)。
- [ ] `r=t` 样本退化为 Flow Matching。
- [ ] JVP path fp32。
- [ ] finite-difference audit 定期运行。
- [ ] full/diagonal/local JVP 都可切换。
- [ ] JVP norm tail、NaN、cross_ratio 记录。

### 10.2 HT/LTG checklist

- [ ] 默认 LTG baseline。
- [ ] PAE latent 小网格 sweep。
- [ ] 记录 max/mean/quantile/local contrast。
- [ ] r/t curriculum。
- [ ] difficulty calibration metrics。

### 10.3 FD-loss checklist

- [ ] reference stats 路径、数据 split、preprocessing 完全固定。
- [ ] queue warmup/reset 策略。
- [ ] train/validation/hold-out representations 分离。
- [ ] FD loss normalization 防止尺度不均。
- [ ] 保留 anchor loss，防止 FD-only drift。
- [ ] 最近邻、precision/recall、sFID、class accuracy。
- [ ] FD-post 与 base 结果分开报告。

---

## 11. 核心参考资料与可核验链接

> 以下为本报告建议使用 / 已核验的核心资料入口。具体数值以论文最终版与官方评测脚本为准。

### 核心组件

- PAE / Prior-Aligned AutoEncoder：任务书给定 arXiv `2605.07915`，GitHub `ZhengrongYue/PAE`。建议 Phase 0 核验最终论文与 release tag。
- Patch Forcing / Heterogeneous Timesteps：arXiv `2604.19141`；GitHub：<https://github.com/CompVis/patch-forcing>；项目页：<https://compvis.github.io/patch-forcing>
- MeanFlow：arXiv `2505.13447`；建议核验官方代码与 `torch.func.jvp` 用法。
- FD-loss / Representation Fréchet Loss：arXiv `2604.28190`；GitHub：<https://github.com/Jiawei-Yang/FD-Loss>

### FD-loss 源码核验入口

- Differentiable FD loss：<https://raw.githubusercontent.com/Jiawei-Yang/FD-Loss/main/frechet_distance/losses.py>
- Feature queue：<https://raw.githubusercontent.com/Jiawei-Yang/FD-Loss/main/frechet_distance/queue.py>
- Feature extractors：<https://raw.githubusercontent.com/Jiawei-Yang/FD-Loss/main/frechet_distance/repr_models.py>
- FD post-training main script：<https://raw.githubusercontent.com/Jiawei-Yang/FD-Loss/main/main_fd.py>
- Experiment scripts README：<https://raw.githubusercontent.com/Jiawei-Yang/FD-Loss/main/scripts/README.md>

### 相关方法与工具

- PyTorch `torch.func.jvp` 官方文档：<https://pytorch.org/docs/stable/generated/torch.func.jvp.html>
- PyTorch `torch.func.vmap` 官方文档：<https://pytorch.org/docs/stable/generated/torch.vmap.html>
- Diffusion Forcing：arXiv `2407.01392`
- MuLAN：arXiv `2312.13236`
- REPA：arXiv `2410.06940`
- VA-VAE / LightningDiT：arXiv `2501.01423`
- REPA-E：arXiv `2504.10483`
- Min-SNR：arXiv `2303.09556`
- TPDM：arXiv `2412.01243`
- SplitMeanFlow：任务书给定 arXiv `2507.16884`，需以后续正式版本核验。
- Understanding MeanFlow Training：任务书给定 arXiv `2511.19065`，需以后续正式版本核验。

---

## 12. 最终可执行结论

如果现在就要开始执行，建议按下面 5 条落地：

1. **今天先写 full-bundle JVP toy harness。** 这是 Per-Patch MeanFlow 生死线。
2. **同时把 FD-loss repo 接到一个最小 PAE/LightningDiT sampling-with-grad pipeline。** 这是 v2.1 最大新增风险与最大新增机会。
3. **主线代码先实现 `PAE×HT×Flow Matching`。** 它是所有后续 MeanFlow/FD-loss/adaptive inference 的稳定地基。
4. **MeanFlow 分支只在 JVP gate 通过后扩大。** 不要用 B300 试错公式。
5. **从第一天起分离 train reps 与 hold-out reps。** 否则即使拿到 FID 0.7，也可能在审稿中被 representation gaming 一票否决。
