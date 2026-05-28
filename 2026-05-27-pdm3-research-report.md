# PDM-3 三层级解耦扩散模型研究报告

> 输入任务书：`/workspace/PDM/2026-05-27-pdm3-research-brief.md`  
> 输出日期：2026-05-27  
> 主题：PAE × DDT × MeanFlow 三层级解耦扩散模型的实现路径、隐藏风险与 fallback framing  
> 结论先行：**方向值得做，但必须把“MeanFlow JVP 稳定性 + PAE/RAE 语义 latent 下的训练爆炸风险 + 竞争路线已出现”作为 Phase 0–2 的硬闸门。**

---

## 0. 执行摘要

### 0.1 总体判断

PDM-3 的三组件组合有真实研究价值：

- **PAE** 在 latent 层显式塑形 diffusion-friendly manifold；PAE 论文声称其三类 latent 几何属性（空间结构一致、局部流形连续、全局语义组织）比单纯 rFID 更能预测 gFID，并在 ImageNet-256 达到 gFID 1.03、收敛加速至 RAE 的约 13×（任务书中给出的配置，已通过 arXiv/HuggingFace/GitHub 摘要核验）。
- **DDT** 在网络层把“语义抽取”和“速度/细节解码”拆开，DDT-XL/2 报告 ImageNet-256 FID 1.31，并声称训练收敛约 4× 加速；官方 GitHub README 还列出 22En/6De、FID 1.26 的 checkpoint 配置。
- **MeanFlow** 在目标层直接学习平均速度，原论文报告从零训练、无需蒸馏的 ImageNet-256 1-NFE FID 3.43。

但是，外部研究中出现了一个对 PDM-3 **非常关键的新信号**：**MeanFlow Transformers with Representation Autoencoders (MF-RAE, arXiv:2511.13019)** 已经把 MeanFlow 和 RAE 类语义 latent 结合，并明确观察到“naive RAE latent MeanFlow training suffers from severe gradient explosion”，随后用 consistency mid-training、teacher distillation、bootstrapping 稳定训练，并报告 ImageNet-256 1-step FID 2.03、训练成本显著降低。这个结果既是 PDM-3 的强支撑（语义 latent × MeanFlow 确实有潜力），也是强警报（**语义 latent 不会自动稳定 MeanFlow，甚至可能更容易爆炸**）。

因此，本报告建议：

1. **不要把 Proposition 1 作为先验真理推进**；应把它变成 Phase 2 的可证伪假设，并用一组指标而非单一 JVP norm 验证。
2. **工程 MVP 必须先实现“JVP 安全带”与“JVP-free fallback”**：fp32 JVP island + finite-difference audit + SplitMeanFlow-style interval consistency 双轨并行。
3. **DDT ratio 不要预设“PAE 语义更强所以 encoder 变浅”**；这只是一个合理假设。需要通过 semantic-gap 指标、proxy sweep、LayerDrop/supernet 搜索来确定。
4. **CFG 需要变成模型输入，而不是训练一个固定 omega 的模型**；否则 built-in CFG 会导致 inference-time sweep 受限，拖慢 Phase 3 决策。
5. **最终论文 framing 最稳妥的主线**应是：
   - 主贡献：平均速度目标 × 解耦语义/细节架构 × diffusion-friendly latent manifold 的系统 co-design；
   - 理论贡献：PAE/MCR 对 MeanFlow 稳定性的**有条件命题 + 多指标实证**；
   - fallback：若 Prop 1 失败，将其降级为 negative/inconclusive finding，转向 DDT × MeanFlow / SplitMeanFlow 的架构目标协同。

### 0.2 需要立刻加入实验计划的“硬闸门”

| 闸门 | 通过条件 | 若失败的直接动作 |
|---|---|---|
| JVP harness | PAE/DDT/MeanFlow 小模型 10K step 无 NaN，JVP p99 与 target variance 可记录 | 改用 SplitMeanFlow/finite-difference 或 detach encoder tangent |
| PAE vs SD-VAE vs VA-VAE vs RAE 稳定性 | 至少 2 个稳定性指标 PAE 优于基线 | Proposition 1 降级；不要宣称 MCR 保证 |
| DDT ratio proxy | 20/4、16/8、12/12、0/24 在 20–40 epoch proxy 上拉开趋势 | 主训前不要锁 DDT-L ratio |
| CFG 可调性 | 一个模型支持 omega/interval sweep，无需多训模型 | 若做不到，优先 no-CFG 与 few-step 质量，CFG 作为后置 |
| MF-RAE 竞争对照 | 能复现/引用其关键稳定技巧或解释差异 | 论文 novelty 需要重写，避免被 reviewer 质疑未比较 |

---

## 1. 探索区域 1：DDT 因子分解下 MeanFlow JVP 的数值稳定性

### 1.1 发现的方案全景

- **方案 1.1：fp32 JVP island + 全量稳定性 telemetry（MVP，已知但需系统化）**  
  把 `torch.func.jvp` 相关前向、time embedding、target 构造强制放在 fp32，并记录 JVP/target/grad 的长尾统计。

- **方案 1.2：时间区间 curriculum 与 endpoint clipping（已知扩展）**  
  先从 `r=t` 或小间隔 `t-r` 学起，再逐步放大区间；避免 `t≈0/1` 的数值病态。

- **方案 1.3：DDT-specific stop-gradient / stop-tangent 放置（新发现，强推荐 ablation）**  
  对 `s_t = f_e(z_t,t,c)` 在 MeanFlow target 的 JVP 路径上 detach，切断 `∂f_d/∂s · ∂f_e/∂t`，保留 encoder 在 prediction loss 上的梯度。

- **方案 1.4：finite-difference JVP audit / fallback（新发现，来自数值分析）**  
  用中心差分近似 `du/dt`，不依赖 forward-mode AD；主要作为 audit，也可在 PyTorch forward-mode 不支持某些算子时做 fallback。

- **方案 1.5：SplitMeanFlow / Interval Splitting Consistency（新发现，最高价值 fallback）**  
  用积分可加性得到纯代数一致性目标，完全绕开 JVP。SplitMeanFlow 论文明确声称这样更高效、更稳定、更硬件友好。

- **方案 1.6：JVP-aware attention kernel 与算子白名单（新发现，工程关键）**  
  在 JVP 路径禁用不支持 forward-mode 的 FlashAttention/Triton/Mamba 类 fused op；普通路径仍可用高性能 kernel。

- **方案 1.7：Lipschitz/Jacobian regularization（新发现，理论-工程桥）**  
  显式惩罚 `||J_u · direction||`、time embedding 曲率、encoder/decoder 局部谱范数，减少 JVP 长尾。

- **方案 1.8：Neural ODE adjoint/checkpointing 经验迁移（跨领域）**  
  不建议直接用 adjoint 替代 JVP，但借鉴“离散化后优化 vs 连续伴随”误差审计、adaptive checkpoint、有限差分一致性检查。

- **方案 1.9：Blackwell/bf16 chaos harness（新发现，必须做）**  
  把 NaN、Inf、overflow、dtype cast、grad scaler、非确定性 kernel 作为独立测试基建，而不是等主训时撞墙。

### 1.2 方案详情

#### 方案 1.1：fp32 JVP island + 全量稳定性 telemetry

**一句话原理**：MeanFlow 的 target 由 prediction 与 JVP 共同构成；JVP 的尾部误差会直接污染监督信号，因此把 JVP 子图提升到 fp32，并把所有可能爆炸的中间量可视化。

**灵感来源**：
- MeanFlow 使用平均速度 identity 训练并依赖 JVP，报告 1-NFE ImageNet-256 FID 3.43：<https://arxiv.org/abs/2505.13447>
- PyTorch `torch.func.jvp` 是 forward-mode AD，文档提示某些算子可能未实现 forward-mode：<https://docs.pytorch.org/docs/stable/generated/torch.func.jvp.html>
- PyTorch AMP 文档说明 autocast 可按区域控制混合精度：<https://docs.pytorch.org/docs/stable/accelerator/amp.html>
- `clip_grad_norm_` 可用 `error_if_nonfinite=True` 直接暴露非有限梯度：<https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html>

**具体实现步骤**：

1. 把 DDT 包装成纯函数，显式传入 `(z, r, t, class)`：

   ```python
   from torch.func import jvp

   def ddm_forward(z, r, t, y, *, force_fp32=False):
       if force_fp32:
           with torch.autocast('cuda', enabled=False):
               z = z.float(); r = r.float(); t = t.float()
               # class embedding 可保持 fp32 或至少 time embedding fp32
               return model(z, r, t, y, force_fp32_time=True)
       else:
           return model(z, r, t, y)
   ```

2. JVP path 只在 target 构造时启用 fp32：

   ```python
   def net_for_jvp(z_, r_, t_):
       return ddm_forward(z_, r_, t_, y, force_fp32=True)

   z = z.float() if warmup_fp32 else z
   v = (eps - z0).float()        # linear FM velocity
   zero_r = torch.zeros_like(r)
   one_t = torch.ones_like(t)

   u_pred_jvp, du_dt = jvp(
       net_for_jvp,
       (z_t.float(), r.float(), t.float()),
       (v.float(), zero_r.float(), one_t.float()),
       strict=False,
   )

   target = v - (t - r).view(-1, *([1] * (v.ndim - 1))).float() * du_dt
   target = target.detach()

   # prediction path 可以走 AMP；但 MVP 阶段建议先 fp32/bf16 分开对比
   u_pred = ddm_forward(z_t, r, t, y, force_fp32=False)
   loss = F.mse_loss(u_pred.float(), target.float())
   ```

3. 每 N step 记录以下统计，按 tokenizer、ratio、precision 分组：

   ```python
   def norm_per_sample(x):
       return x.flatten(1).norm(dim=1)

   metrics = {
       'jvp_norm_p50': norm_per_sample(du_dt).quantile(0.50),
       'jvp_norm_p95': norm_per_sample(du_dt).quantile(0.95),
       'jvp_norm_p99': norm_per_sample(du_dt).quantile(0.99),
       'target_norm_p99': norm_per_sample(target).quantile(0.99),
       'u_norm_p99': norm_per_sample(u_pred).quantile(0.99),
       'target_to_v_ratio_p99': (norm_per_sample(target) / (norm_per_sample(v) + 1e-6)).quantile(0.99),
       'delta_mean': (t-r).mean(),
       'delta_p99': (t-r).quantile(0.99),
   }
   ```

4. 梯度安全带：

   ```python
   loss.backward()
   total_norm = torch.nn.utils.clip_grad_norm_(
       model.parameters(), max_norm=0.5, error_if_nonfinite=True
   )
   optimizer.step()
   ```

5. 增加“bad batch dump”：一旦发现 `target` 或 `du_dt` 非有限，保存 `z0/eps/t/r/class/tokenizer_id/model_state_hash`，便于重放。

**涉及组件**：MeanFlow 训练循环、DDT forward、time embedding、AMP、日志系统。

**与现有 PDM-3 系统的对接方式**：
- 不改变 PAE latent 格式；
- DDT forward 增加 `force_fp32_time` 或 `force_fp32`；
- 训练配置增加 `jvp_precision=fp32`、`jvp_telemetry_interval`、`bad_batch_dump_dir`。

**难度评估**：中。核心代码不难，难点在 forward-mode AD 与 fused attention/operator 兼容性。

**预期效果**：
- 不能从根本上消除 JVP 长尾，但能显著降低早期 NaN；
- 让 Proposition 1 的实验数据可用；
- 预计训练开销增加 10–30%，取决于是否禁用高性能 attention kernel。

**先例/参考实现**：
- MeanFlow 原论文：<https://arxiv.org/abs/2505.13447>
- PyTorch `jvp` 文档：<https://docs.pytorch.org/docs/stable/generated/torch.func.jvp.html>
- 非官方 PyTorch MeanFlow repo 提到 JVP 与 FlashAttention/Triton 的兼容问题及显存增长：<https://github.com/haidog-yaqub/MeanFlow>

**风险与不确定性**：
- fp32 island 可能仍然无法处理某些 fused op 的 forward-mode；
- 如果 DDT-L + PAE-f16d32 显存过高，fp32 JVP 可能无法在单张 Pro 6000 96GB 上跑全 batch；
- 需要先在 DDT-B/PAE cache latent 上做 10K step harness。

---

#### 方案 1.2：时间区间 curriculum 与 endpoint clipping

**一句话原理**：MeanFlow 的难度随区间长度 `Δ=t-r` 与 endpoint 区域数值病态上升；先学局部速度，再学长跳平均速度。

**灵感来源**：
- MeanFlow 默认 75% `r=t`、25% `r≠t`，本身已包含局部 FM anchor。
- 任务书中列出的 Understanding MeanFlow Training（arXiv:2511.19065，需进一步核验）指出小间隔到大间隔 curriculum 可能有意义。
- Neural ODE/数值积分中的 adaptive step-size 与 stiffness 处理思想：先保证局部稳定，再扩展步长。

**具体实现步骤**：

1. 设定 `delta_max(step)`：

   ```python
   def delta_max(step):
       # 0-20k: 只学 r=t / 极小 interval
       if step < 20_000:
           return 0.02
       # 20k-100k: cosine 增长到 1.0
       p = min(1.0, (step - 20_000) / 80_000)
       return 0.02 + 0.98 * 0.5 * (1 - math.cos(math.pi * p))
   ```

2. 采样时限制：

   ```python
   t = sample_lognorm_time(batch).clamp(eps_t, 1 - eps_t)
   if torch.rand(()) < p_nonzero_interval(step):
       d = torch.rand_like(t) * delta_max(step)
       r = (t - d).clamp(eps_t, t)
   else:
       r = t
   ```

3. `eps_t` 从 0.05 起步，后期可降到 0.01；不要一开始覆盖极端 endpoint。

4. 对每个 bucket 记录 JVP 分布：`t∈[0,0.05]`、`[0.05,0.2]`、`[0.2,0.8]`、`[0.8,0.95]`、`[0.95,1]`。

**涉及组件**：MeanFlow time sampler、训练配置、日志。

**与现有 PDM-3 系统的对接方式**：新增 sampler 配置，不改模型。

**难度评估**：低。

**预期效果**：早期训练稳定性显著提高；但如果过度 curriculum，最终 1-NFE 长跳能力可能不足，需要最后 20–30% steps 充分覆盖 `r=0,t=1`。

**先例/参考实现**：
- MeanFlow 原始采样比例：<https://arxiv.org/abs/2505.13447>
- Neural ODE adaptive step 与稳定性背景：<https://arxiv.org/abs/1806.07366>

**风险与不确定性**：长区间能力收敛慢；需要 mini-experiment 比较“无 curriculum / 50K curriculum / 100K curriculum”的 1-NFE FID proxy。

---

#### 方案 1.3：DDT-specific stop-gradient / stop-tangent 放置

**一句话原理**：DDT 中 JVP 的高风险项是 `∂f_d/∂s · ∂f_e/∂t`；可以只在 target 的 JVP 路径切断 encoder 的 time tangent，保留 encoder 在 prediction loss 上学习。

**灵感来源**：
- DDT 将语义 encoder 与 velocity decoder 解耦，官方动机就是降低语义/细节目标冲突：<https://arxiv.org/abs/2504.05741>
- MeanFlow target 只对 target stop-gradient，并不要求所有内部链式项必须无偏；工程上可以引入 bias-stability tradeoff。

**具体实现步骤**：

1. baseline：完整 JVP。

   ```python
   def full_net(z, r, t):
       s = encoder(z, t, y)
       return decoder(z, s, r, t, y)
   u, du = jvp(full_net, (z_t, r, t), (v, 0*r, 1+0*t))
   ```

2. stop-tangent target：prediction 用完整图，target 的 JVP 用 detached `s`。

   ```python
   # prediction path: encoder receives gradients through u_pred
   s_live = encoder(z_t, t, y)
   u_pred = decoder(z_t, s_live, r, t, y)

   # target path: block encoder tangent and gradient
   with torch.no_grad():
       s_const = encoder(z_t.float(), t.float(), y).detach()

   def decoder_only_for_jvp(z_, r_, t_):
       # s_const 不随 t_ 变化，只保留 decoder direct time derivative 与 z direction
       return decoder(z_.float(), s_const, r_.float(), t_.float(), y, force_fp32_time=True)

   _, du_dt_approx = jvp(
       decoder_only_for_jvp,
       (z_t.float(), r.float(), t.float()),
       (v.float(), torch.zeros_like(r).float(), torch.ones_like(t).float())
   )
   target = (v.float() - (t-r).view(-1,1,1,1).float() * du_dt_approx).detach()
   loss = F.mse_loss(u_pred.float(), target)
   ```

3. 三个 ablation：
   - **full-chain**：完整 JVP；
   - **stop-s-tangent**：target 中 `s` detached；
   - **ema-s-target**：target 中 `s` 来自 EMA encoder，减少 target 抖动。

4. 记录 bias proxy：比较 `full_du_dt` 与 `detached_du_dt` 在小 batch 上的 cosine / relative error。

**涉及组件**：DDT encoder/decoder、MeanFlow target。

**与现有 PDM-3 系统的对接方式**：增加 `jvp_mode = full | stop_s_tangent | ema_s`。

**难度评估**：中。实现容易，但要解释 bias 与稳定性权衡。

**预期效果**：
- 如果 NaN 主要来自 encoder time tangent，可大幅稳定；
- 可能牺牲理论纯度与最终 FID；
- 可作为 paper 中的“stability ablation”，即使不用于最终模型也有价值。

**先例/参考实现**：没有直接先例；属于将 stop-gradient 技巧应用到 DDT × MeanFlow 的新工程设计。

**风险与不确定性**：
- target 有偏，可能导致 1-NFE 学不到真实平均速度；
- reviewer 可能质疑理论 identity 被破坏；
- mini-experiment：在 DDT-B 上对比 20K step 的 loss、JVP p99、1K sample FID proxy。

---

#### 方案 1.4：finite-difference JVP audit / fallback

**一句话原理**：用中心差分沿 `(z,t)` 联合方向近似 `du/dt`，绕开 forward-mode AD，作为数值正确性审计或 fallback。

**灵感来源**：数值分析中用 finite difference 检查 AD/JVP 正确性；PyTorch `jvp` 文档也提醒 forward-mode 可能因未实现算子失败。

**具体实现步骤**：

```python
@torch.no_grad()
def choose_eps(z, t):
    # bf16 下 eps 不能太小；fp32 audit 可用 1e-3~1e-2 sweep
    return 1e-3

def fd_du_dt(net, z, r, t, v, y, eps=1e-3):
    # derivative of u(z_t, r, t) along dz/dt=v, dr/dt=0, dt/dt=1
    shape = (-1,) + (1,) * (z.ndim - 1)
    z_plus  = z + eps * v
    z_minus = z - eps * v
    t_plus  = (t + eps).clamp(0, 1)
    t_minus = (t - eps).clamp(0, 1)
    u_plus  = net(z_plus,  r, t_plus,  y)
    u_minus = net(z_minus, r, t_minus, y)
    return (u_plus - u_minus) / (2 * eps)
```

使用方式：
- 每 500–1000 step 抽小 batch，比较 `du_dt_jvp` 与 `du_dt_fd`；
- 如果某些 kernel 不支持 JVP，则短期用 FD target 训练小模型；
- 对 FD 误差做 `eps ∈ {1e-2, 3e-3, 1e-3}` sweep，避免截断误差/舍入误差。

**涉及组件**：训练循环、debug harness。

**对接方式**：新增 `jvp_audit_interval`，可选 `target_mode=fd`。

**难度评估**：低到中；额外 2 次 forward，慢但简单。

**预期效果**：
- 最适合作为 correctness audit；
- 作为主训练 target 会显著增加 compute，且 FD 噪声可能偏大。

**先例/参考实现**：AD/数值分析通用做法；PyTorch JVP 文档：<https://docs.pytorch.org/docs/stable/generated/torch.func.jvp.html>

**风险与不确定性**：高维网络差分误差长尾明显；必须只作为 fallback，不建议作为最终主线。

---

#### 方案 1.5：SplitMeanFlow / Interval Splitting Consistency（JVP-free）

**一句话原理**：平均速度满足区间可加性：先从 `t→s` 再从 `s→r` 的总位移等于 `t→r` 的总位移；用这个代数一致性训练平均速度，不求导。

**灵感来源**：SplitMeanFlow（arXiv:2507.16884）提出 Interval Splitting Consistency，声称 MeanFlow differential identity 是其无穷小极限，并且其训练无需 JVP、更稳定、更硬件友好：<https://arxiv.org/abs/2507.16884>

**具体实现步骤**：

1. 采样 `r < s < t`，网络预测三段 average velocity：

   ```python
   u_rt = net(z_t, r, t, y)       # long interval
   u_st = net(z_t, s, t, y)       # upper sub-interval
   z_s_hat = (z_t - (t - s).view(-1,1,1,1) * u_st).detach()
   u_rs = net(z_s_hat, r, s, y)   # lower sub-interval
   ```

2. 位移可加性目标：

   ```python
   lhs = (t - r).view(-1,1,1,1) * u_rt
   rhs = (t - s).view(-1,1,1,1) * u_st.detach() \
       + (s - r).view(-1,1,1,1) * u_rs.detach()
   loss_split = F.mse_loss(lhs, rhs)
   ```

3. 加 FM anchor，防止自一致性塌缩：

   ```python
   # r=t anchor: u(z_t,t,t) should equal instantaneous v = eps - z0
   u_fm = net(z_t, t, t, y)
   loss_fm = F.mse_loss(u_fm, v)
   loss = loss_fm + lambda_split(step) * loss_split
   ```

4. 用 EMA teacher 稳定 RHS：

   ```python
   with torch.no_grad():
       u_st_T = ema_net(z_t, s, t, y)
       z_s_T = z_t - (t-s)*u_st_T
       u_rs_T = ema_net(z_s_T, r, s, y)
   target_disp = (t-s)*u_st_T + (s-r)*u_rs_T
   loss_split = mse((t-r)*u_rt, target_disp)
   ```

5. 对 DDT 特别处理：encoder 仍只看 `(z_time, time, c)`，decoder 看 `(r,t)`。对 `u_rs`，encoder 输入应是 `z_s_hat, s`，而不是复用 `z_t,t`。

**涉及组件**：MeanFlow objective、DDT forward、多步/任意步推理。

**对接方式**：新增 `objective=meanflow_jvp | split_meanflow | hybrid`。hybrid 推荐：前期 SplitMeanFlow 稳定预训，后期少量 JVP fine-tune 对齐原 MeanFlow identity。

**难度评估**：中到高。公式实现不复杂，但需要防止自蒸馏塌缩、teacher lag、区间采样偏差。

**预期效果**：
- 极大降低 PyTorch/Blackwell/bf16 JVP 风险；
- 可能与 DDT 更兼容，因为不再要求 `∂f_e/∂t` 稳定；
- 如果最终 FID 接近 JVP MeanFlow，可作为 PDM-3 的最稳工程路线。

**先例/参考实现**：
- SplitMeanFlow arXiv：<https://arxiv.org/abs/2507.16884>
- 非官方实现可参考但需审计：<https://github.com/primepake/splitmeanflow>

**风险与不确定性**：
- SplitMeanFlow 主要在 speech/TTS 产品中报告部署，不一定直接迁移 ImageNet latent；
- 自一致性目标容易学到错误但自洽的 flow map；
- mini-experiment：CIFAR/Imagenette/latent ImageNet subset 上对比 JVP MeanFlow 与 SplitMeanFlow 的 loss stability、1-step FID proxy。

---

#### 方案 1.6：JVP-aware attention kernel 与算子白名单

**一句话原理**：forward-mode AD 对 fused kernels 支持不完整；JVP path 用保守 PyTorch op，普通 path 用高性能 kernel。

**灵感来源**：
- PyTorch `torch.func.jvp` 文档提示某些 operator 可能不支持 forward-mode AD。
- 非官方 MeanFlow PyTorch repo README 明确提到 JVP 与 Flash Attention/Triton 类库兼容问题、显存增长问题：<https://github.com/haidog-yaqub/MeanFlow>

**具体实现步骤**：

1. 在 attention 模块增加 backend 参数：

   ```python
   class Attention(nn.Module):
       def forward(self, x, *, jvp_safe=False):
           if jvp_safe:
               return torch.nn.functional.scaled_dot_product_attention(
                   q, k, v, dropout_p=0.0, is_causal=False
               )  # 必要时强制 math/eager backend
           else:
               return flash_attn_func(q, k, v)
   ```

2. 在 `ddm_forward(force_fp32=True)` 里设置 `jvp_safe=True`。

3. 用单元测试覆盖：

   ```python
   def test_jvp_all_blocks():
       x = torch.randn(2, 1024, d, device='cuda')
       t = torch.rand(2, device='cuda')
       fn = lambda x_, t_: block(x_, t_, jvp_safe=True)
       y, jy = torch.func.jvp(fn, (x, t), (torch.randn_like(x), torch.ones_like(t)))
       assert torch.isfinite(y).all() and torch.isfinite(jy).all()
   ```

**涉及组件**：DDT attention block、MLP block、time embedding、normalization。

**难度评估**：中。需要梳理代码中所有 fused/custom op。

**预期效果**：减少 “forward-mode AD not implemented” 与 silent numerical bug；代价是 JVP path 变慢。

**风险与不确定性**：某些高性能 kernel 是训练吞吐关键；JVP-safe path 可能导致实际训练吞吐低于预算。

---

#### 方案 1.7：Lipschitz/Jacobian regularization

**一句话原理**：直接惩罚网络输出对 `(z,t,s)` 的局部敏感度，减少 JVP 长尾，给 Proposition 1 增加可操作控制项。

**灵感来源**：PAE 的 MCR 关注 local manifold continuity；数值稳定本质上需要控制局部 Jacobian。CFG++ 也把 CFG 病理解释为 off-manifold 现象，说明 manifold 约束与 guidance/flow 稳定相关：<https://arxiv.org/abs/2406.08070>

**具体实现步骤**：

1. JVP norm penalty（低频采样，不每 step 做）：

   ```python
   if step % reg_interval == 0:
       rand_dir = torch.randn_like(z_t)
       rand_dir = rand_dir / (rand_dir.flatten(1).norm(dim=1).view(-1,1,1,1) + 1e-6)
       _, j_rand = jvp(lambda zz: net(zz, r, t, y), (z_t.float(),), (rand_dir.float(),))
       loss_reg = lambda_jac * (norm_per_sample(j_rand).clamp(max=jac_clip) ** 2).mean()
   ```

2. Time curvature penalty：

   ```python
   eps = 1e-2
   u0 = net(z_t, r, t, y)
   up = net(z_t, r, (t+eps).clamp(0,1), y)
   um = net(z_t, r, (t-eps).clamp(0,1), y)
   loss_time_curv = ((up - 2*u0 + um) / eps**2).square().mean()
   ```

3. Encoder/decoder 分别监控：`||∂s/∂t||`、`||∂u/∂s||`，定位爆炸来自哪里。

**涉及组件**：DDT、MeanFlow objective、日志。

**难度评估**：中高。额外 JVP/forward 增加 compute；正则过强会损害表达能力。

**预期效果**：降低 JVP p99/p999 与 target variance；可能略牺牲最终 FID但提升稳定学习率。

**风险与不确定性**：正则可能把模型变“钝”，1-NFE 细节不足。建议只作为 ablation 或在前期启用后期衰减。

---

#### 方案 1.8：Neural ODE adjoint/checkpointing 经验迁移

**一句话原理**：不用 adjoint 替代 MeanFlow JVP，但借鉴 Neural ODE 社区对连续系统梯度误差、checkpointing、stiffness 的诊断方法。

**灵感来源**：
- Neural ODE 通过黑盒 ODE solver 与 adjoint sensitivity 训练连续深度模型：<https://arxiv.org/abs/1806.07366>
- torchdiffeq 支持 adjoint method 的 O(1) memory backprop：<https://github.com/rtqichen/torchdiffeq>
- Diffrax 文档区分 recursive checkpoint adjoint 与 backsolve adjoint，强调不同梯度策略的误差/内存权衡：<https://docs.kidger.site/diffrax/api/adjoints/>

**具体实现步骤**：

1. 建立 JVP-vs-FD 一致性测试（方案 1.4）。
2. 对 DDT block 做 checkpoint，而不是试图对 MeanFlow identity 做 adjoint：

   ```python
   x = torch.utils.checkpoint.checkpoint(block, x, t, use_reentrant=False)
   ```

3. 做 stiffness proxy：统计同一 batch 在 `t` 小扰动下输出变化率：

   ```python
   stiffness_proxy = (net(z, r, t+eps, y) - net(z, r, t, y)).flatten(1).norm(dim=1) / eps
   ```

4. 如果 stiffness proxy 的 p99 与 NaN 高度相关，纳入 Proposition 1 的替代指标。

**涉及组件**：训练内存、诊断系统。

**难度评估**：中。

**预期效果**：提升可诊断性；对主模型性能间接有帮助。

**风险与不确定性**：adjoint 思想可能被误用。MeanFlow 是显式学习平均速度，不是数值解 ODE；不要引入复杂 ODE solver 作为主线。

---

#### 方案 1.9：Blackwell/bf16 chaos harness

**一句话原理**：把混合精度、硬件 kernel、NaN 复现做成独立测试，而不是在主训中边跑边猜。

**具体实现步骤**：

1. 固定 16 个 batch，保存 PAE latent cache。
2. 对以下配置跑 1000 step micro-train：
   - fp32 all；
   - AMP bf16 all；
   - bf16 + fp32 JVP；
   - bf16 + stop-s-tangent；
   - SplitMeanFlow no JVP。
3. 每步检测：

   ```python
   for name, tensor in named_intermediates:
       if not torch.isfinite(tensor).all():
           raise FloatingPointError(f'{name} non-finite at step {step}')
   ```

4. 保存 `torch.__config__.show()`、CUDA/cuDNN/NCCL、GPU driver、PyTorch commit、attention backend。

**难度评估**：低。

**预期效果**：减少 Phase 3 B300 burst 时的不可控风险。

**风险与不确定性**：测试集小，不能完全代表长训；但仍是必要保险。

### 1.3 技术顾问建议

1. **推荐 MVP 路径**：方案 1.1 + 1.2 + 1.3 + 1.9。先让 JVP MeanFlow 在 DDT-B/PAE cache latent 上稳定 10K–20K step，并产出 JVP 分布数据。
2. **推荐终极路径**：JVP MeanFlow 与 SplitMeanFlow 双实现。主线若稳定用 JVP；不稳定则用 SplitMeanFlow/hybrid 作为最终 objective。
3. **实施顺序**：
   1. JVP-safe DDT forward 单元测试；
   2. fp32 island + telemetry；
   3. full-chain vs stop-s-tangent ablation；
   4. SplitMeanFlow prototype；
   5. 再进入 PAE/DDT/MeanFlow pairwise。
4. **值得做但不该在本论文主线做**：完整 Neural ODE adjoint/implicit differentiation 改造。这会把论文变成数值方法论文，偏离 ImageNet SOTA 主目标。

---

## 2. 探索区域 2：PAE SSR/MCR/SCR 在 MeanFlow 下的调度

### 2.1 发现的方案全景

- **方案 2.1：冻结默认 PAE tokenizer（MVP，已知）**  
  先不要重训 PAE，避免把 tokenizer 与 generator 两个不稳定源混在一起。

- **方案 2.2：MeanFlow-aware tokenizer 选择指标（新发现，强推荐）**  
  不用 full FID grid search，而用 JVP norm、target variance、gradient SNR、LR stability boundary 作为 PAE weight/schedule proxy。

- **方案 2.3：MCR 强早期、弱后期的 curriculum（已知扩展）**  
  早期优先流形连续性稳定 JVP，后期释放重建/细节能力。

- **方案 2.4：GradNorm/Uncertainty/PCGrad 动态损失权重（跨领域）**  
  把 SSR/MCR/SCR/reconstruction/MeanFlow-aware proxy 看成多任务损失，自动调权或做梯度冲突处理。

- **方案 2.5：Trajectory-smoothing PAE loss（新发现）**  
  不是只约束 latent 点邻域，而是约束 linear bridge 或同类 latent interpolation 上 decoder/VFM feature 的平滑性。

- **方案 2.6：PAE × MeanFlow end-to-end fine-tuning（REPA-E 类路线）**  
  只解冻 DAM/decoder 后层或 latent affine，和 MeanFlow 小 LR 联训。

- **方案 2.7：OT/Wasserstein latent conditioning（跨领域）**  
  对 latent covariance、class-conditional distribution、whitening、局部 density 条件数做约束，补足 MCR 不直接约束 `p_z` 的漏洞。

- **方案 2.8：SSR/SCR 的数学角色与潜在冲突（理论补强）**  
  SSR 影响空间局部平滑，SCR 影响全局语义聚类；二者可能与平均速度的全局轨迹平滑冲突。

- **方案 2.9：MeanFlow-aware PAE v2 / tangent tokenizer（Plan B）**  
  若默认 PAE 不改善稳定性，加入显式 tangent/Jacobian regularization 或用 MF-RAE 的稳定训练经验。

### 2.2 方案详情

#### 方案 2.1：冻结默认 PAE tokenizer

**一句话原理**：先把 tokenizer 当成固定数据变换，隔离 MeanFlow/DDT 不稳定性。

**灵感来源**：PAE 官方提出 tokenizer 框架，并开源代码与模型；其 README 声称显式塑形 latent manifold、ImageNet-256 gFID 1.03、13× faster convergence：<https://github.com/ZhengrongYue/PAE>；arXiv 页面给出三类 diffusion-friendly latent manifold 属性：<https://arxiv.org/abs/2605.07915>

**具体实现步骤**：

1. 使用 PAE-f16d32 预训练 tokenizer encode ImageNet train split，保存 latent cache：

   ```bash
   python tools/cache_pae_latents.py \
     --pae_ckpt PAE-f16d32.pt \
     --data /path/imagenet/train \
     --out /path/cache/pae_f16d32
   ```

2. 对 latent 做全局统计：mean/std、per-channel variance、class-conditional covariance、latent norm p99。
3. 训练 MeanFlow/DDT 时只读 latent，不回传 PAE。
4. tokenizer ablation 同步 cache：SD-VAE、VA-VAE、RAE、PAE。

**涉及组件**：PAE encoder、数据 pipeline、MeanFlow 输入归一化。

**对接方式**：把 `latent_shape` 从 SD-VAE 的 `4×32×32` 改为 PAE 的 `32×32×32`；模型 patch embedding 必须适配 channel 数。

**难度评估**：低到中。主要工作在 cache 与 I/O。

**预期效果**：最快验证 PAE latent 是否对 MeanFlow 友好；避免 end-to-end 混乱。

**风险与不确定性**：PAE latent channel 数大，DDT memory/throughput 与 SD-VAE 不可直接比较；需要报告 tokenizer decoder cost。

---

#### 方案 2.2：MeanFlow-aware tokenizer 选择指标

**一句话原理**：用 MeanFlow 训练稳定性指标筛 PAE loss/schedule，而不是一上来训完整生成模型看 FID。

**灵感来源**：PAE 论文指出 rFID 与 gFID 弱相关；MeanFlow/RAE follow-up 显示语义 latent 下梯度爆炸是核心问题。MF-RAE 报告 naive RAE latent MeanFlow 会 severe gradient explosion，并用两阶段稳定方案解决：<https://arxiv.org/abs/2511.13019>

**具体实现步骤**：

对每个 tokenizer 或 PAE checkpoint，跑固定 5K–20K step DDT-B/MeanFlow harness，记录：

```python
metrics = {
  'jvp_norm_p95_p99': ..., 
  'target_var': target.flatten(1).var(dim=0).mean(),
  'grad_norm_p99': ..., 
  'nan_count': ..., 
  'max_stable_lr': ..., 
  'fd_jvp_rel_error': ..., 
  'latent_cov_cond': eigmax(cov) / eigmin(cov),
  'class_separability': linear_probe_acc,
}
```

建议输出一个 tokenizer score：

```text
score = rank(JVP_p99) + rank(target_var) + rank(NaN_rate) + rank(max_stable_lr^-1) + 0.5*rank(FD_error)
```

**涉及组件**：PAE/SD-VAE/VA-VAE/RAE tokenizer、MeanFlow harness。

**对接方式**：新增 `tokenizer_eval.py`，在 Phase 1/2 独立运行。

**难度评估**：中。需要多 tokenizer cache 和统一归一化。

**预期效果**：显著减少 PAE loss grid search 成本，并为 Proposition 1 提供直接证据。

**风险与不确定性**：proxy 与最终 FID 可能不完全相关；需要少量 FID proxy 校准。

---

#### 方案 2.3：MCR curriculum

**一句话原理**：MCR 对局部连续性最相关；早期加大 MCR 稳定 latent geometry，后期减小避免过度平滑损害细节。

**灵感来源**：PAE 的 MCR 针对 local manifold continuity；MeanFlow target 对局部导数/轨迹平滑更敏感。

**具体实现步骤**：

如果需要重训或微调 PAE：

```python
lambda_mcr = lambda_mcr_final + (lambda_mcr_init - lambda_mcr_final) * cosine_decay(step, T=0.4*total_steps)
lambda_ssr = lambda_ssr_default
lambda_scr = lambda_scr_default * warmup(step, 0.1*total_steps)
loss_pae = loss_rec + lambda_ssr*L_ssr + lambda_mcr*L_mcr + lambda_scr*L_scr
```

推荐配置：
- `lambda_mcr_init = 2×default`；
- `lambda_mcr_final = 0.5–1×default`；
- 前 10% steps SCR warmup，避免过早 class clustering 导致局部几何扭曲。

**涉及组件**：PAE training/fine-tuning。

**对接方式**：仅影响 tokenizer 训练；生成模型不变。

**难度评估**：中。重训 PAE 成本不低。

**预期效果**：降低 latent JVP 长尾；可能提升最大稳定 LR。

**风险与不确定性**：过强 MCR 可能降低重建细节，最终 gFID 反而变差。先用方案 2.2 proxy 检验。

---

#### 方案 2.4：GradNorm/Uncertainty/PCGrad 动态损失权重

**一句话原理**：把 SSR、MCR、SCR、reconstruction、perceptual loss 视为多任务优化，自动平衡尺度或处理梯度冲突。

**灵感来源**：
- GradNorm 动态调整多任务梯度范数：<https://arxiv.org/abs/1711.02257>
- Kendall 等用 homoscedastic uncertainty 自动加权多任务损失：<https://arxiv.org/abs/1705.07115>
- PCGrad 将冲突任务梯度投影到非冲突方向：<https://arxiv.org/abs/2001.06782>
- CAGrad 进一步处理 multi-task conflict：<https://arxiv.org/abs/2110.14048>

**具体实现步骤**：

1. 低侵入版本：uncertainty weighting。

   ```python
   log_vars = nn.Parameter(torch.zeros(4))
   losses = torch.stack([L_rec, L_ssr, L_mcr, L_scr])
   weighted = torch.exp(-log_vars) * losses + log_vars
   loss = weighted.sum()
   ```

2. 进阶版本：每 100 step 计算各损失对共享 encoder/DAM 参数的 gradient cosine：

   ```python
   cos_ij = cosine(grad(L_i, shared_params), grad(L_j, shared_params))
   if cos_ij < -0.2: flag_conflict(i,j)
   ```

3. 若 MCR 与 reconstruction/SSR 长期冲突，使用 PCGrad 仅处理 PAE 训练，不进入 generator 主训练。

**涉及组件**：PAE training。

**对接方式**：新增 PAE loss balancer；保留默认权重作为 baseline。

**难度评估**：中高。PCGrad 对大模型开销较大；uncertainty weighting 较简单。

**预期效果**：减少手工 grid search；发现 SSR/MCR/SCR 冲突关系。

**风险与不确定性**：自动权重可能优化到“好重建但差生成”的局部最优；必须用 MeanFlow-aware proxy 校验。

---

#### 方案 2.5：Trajectory-smoothing PAE loss

**一句话原理**：MeanFlow 关心整段轨迹平均速度，因此 tokenizer 不只要点邻域连续，还要在 latent interpolation / noise bridge 上平滑。

**灵感来源**：MeanFlow 学习 interval average velocity；Optimal Transport / Wasserstein gradient flow 关注路径与分布演化。

**具体实现步骤**：

1. 同类 latent interpolation 平滑：

   ```python
   z1, z2 = sample_same_class_latents()
   a = torch.rand(B,1,1,1,device=z1.device)
   z_mix = (1-a)*z1 + a*z2
   x_mix = D(z_mix)
   f_mix = VFM(x_mix).detach()
   f_lin = (1-a_flat)*VFM(D(z1)).detach() + a_flat*VFM(D(z2)).detach()
   L_traj_sem = 1 - cosine(pool(f_mix), pool(f_lin))
   ```

2. Local tangent smoothing：

   ```python
   delta = torch.randn_like(z) * sigma
   L_tangent = lpips(D(z + delta), D(z)) / (delta.flatten(1).norm(dim=1)**2 + eps)
   ```

3. 与 MCR 结合：MCR 控制随机小扰动，trajectory smoothing 控制结构化方向（同类、同语义、noise bridge）。

**涉及组件**：PAE decoder、VFM、DAM。

**对接方式**：新增 tokenizer loss，不改 MeanFlow。

**难度评估**：高。需要额外 VFM forward，训练成本增加。

**预期效果**：若 Proposition 1 中 “MCR 不足以约束轨迹方向” 成立，该方案可能补足。

**风险与不确定性**：同类 interpolation 可能生成非自然图像，VFM feature 监督误导 decoder；建议作为 PAE-v2 future，不作为主线。

---

#### 方案 2.6：PAE × MeanFlow end-to-end fine-tuning

**一句话原理**：像 REPA-E 一样让 tokenizer 与 generator 协同适配，但只微调 PAE 的低风险部分。

**灵感来源**：REPA-E 将 autoencoder 与 DiT 端到端联合训练（任务书已列 arXiv:2504.10483）；MF-RAE 表明 MeanFlow 与语义 autoencoder 的结合需要专门稳定训练。

**具体实现步骤**：

1. 从 frozen PAE + trained PDM-3 checkpoint 开始，不从零端到端。
2. 只解冻：
   - latent affine norm；
   - DAM；
   - PAE decoder 最后若干 block；
   - 不解冻 VFM。
3. stop-gradient 放置：

   ```python
   z = E_pae(x)              # E 可冻结或只训练 affine
   loss_mf = meanflow_loss(z.detach() if freeze_E else z)
   x_rec = D_pae(z)
   loss_rec = rec_loss(x_rec, x)
   loss_total = loss_mf + alpha_rec*loss_rec + alpha_align*loss_align
   ```

4. PAE LR 是 DDT LR 的 0.01–0.1 倍；EMA 只对 generator 或分别维护。

**涉及组件**：PAE、DDT、MeanFlow 训练循环。

**对接方式**：从 Phase 4 ablation 开始，不进入 Phase 3 主训。

**难度评估**：高。debug 难，显存大，理论解释复杂。

**预期效果**：可能进一步提升 gFID，但也可能破坏已学 tokenizer manifold。

**风险与不确定性**：端到端训练一旦不稳定，难以定位是 PAE 还是 MeanFlow；不建议作为主线。

---

#### 方案 2.7：OT/Wasserstein latent conditioning

**一句话原理**：MCR 约束 decoder 局部连续性，但 MeanFlow 稳定还依赖 latent distribution 的条件数；显式控制 latent covariance/density。

**灵感来源**：Optimal transport 与 Wasserstein gradient flow 对分布路径平滑的关注；2026 年 W-Flow（需进一步核验）把 Wasserstein gradient flow 用于 one-step generative modeling，并报告强 one-step ImageNet 结果。

**具体实现步骤**：

1. Latent whitening：

   ```python
   z_norm = (z - mean[None,:,None,None]) / (std[None,:,None,None] + 1e-6)
   ```

2. Covariance condition penalty（PAE 训练时）：

   ```python
   Z = z.flatten(2).permute(0,2,1).reshape(-1, C)
   cov = Z.T @ Z / Z.shape[0]
   loss_cov = ((cov - torch.eye(C,device=cov.device))**2).mean()
   ```

3. Class-conditional covariance audit：如果某些 class covariance 极端 anisotropic，MeanFlow target 可能高方差。

**涉及组件**：PAE latent normalization、训练数据统计。

**对接方式**：先作为 audit 与 normalization，不急着加入 PAE loss。

**难度评估**：中。

**预期效果**：减少 latent anisotropy 导致的高 JVP；增强 Prop 1 理论条件。

**风险与不确定性**：过强 whitening 会破坏 PAE 的语义组织；建议先离线统计和 normalization。

---

#### 方案 2.8：SSR/SCR 的数学角色与潜在冲突

**一句话原理**：MCR 不是全部；SSR 可能控制空间局部 Jacobian，SCR 可能控制 class-conditional flow 的全局边界，但二者也可能制造高曲率。

**理论建议**：

- **MCR**：可对应 decoder 局部 Lipschitz 或感知 Lipschitz，但只在采样扰动分布覆盖的方向有效。
- **SSR**：patch-level spatial coherence 可能降低 latent 中邻近 patch 的局部高频噪声，间接降低 DiT attention 对局部扰动的放大。
- **SCR**：增强 class cluster 与 global semantics，可能降低 conditional generation 难度；但若 class cluster 过窄、类间边界过陡，无条件/CFG 混合时可能产生高曲率。

**具体实现步骤**：做 PAE loss ablation：

| tokenizer | 目的 |
|---|---|
| PAE full | 主线 |
| PAE -MCR | 直接验证 Prop 1 |
| PAE -SSR | 看空间结构对 JVP 的影响 |
| PAE -SCR | 看语义聚类对 CFG/conditional flow 的影响 |
| PAE high-MCR | 看过强连续性是否损害 FID |

每个只跑 MeanFlow stability harness，不一定跑完整 FID。

**难度评估**：中高，取决于是否能拿到 PAE ablation checkpoints。

**风险与不确定性**：重训 tokenizer 成本较大；若无 checkpoint，可用 small subset 训练 proxy PAE。

---

#### 方案 2.9：MeanFlow-aware PAE v2 / tangent tokenizer

**一句话原理**：如果默认 MCR 不能稳定 JVP，直接在 tokenizer 训练时加入 generator tangent proxy。

**具体实现步骤**：

1. 训练一个小 MeanFlow probe `u_probe` 或用 EMA generator。
2. 对 PAE latent 加扰动，惩罚 probe target 变化：

   ```python
   z = E(x).detach()
   delta = sigma * torch.randn_like(z)
   metric = (u_probe(z + delta, r, t) - u_probe(z, r, t)).square().mean()
   L_mf_tangent = metric / (delta.square().mean() + 1e-6)
   ```

3. 只用于 tokenizer fine-tune，不让 probe 反向污染主 generator。

**难度评估**：高。

**预期效果**：理论上最贴合 MeanFlow；工程上风险最大。

**风险与不确定性**：会把 tokenizer 过拟合到一个不成熟 generator；建议只作为 Plan C/Future work。

### 2.3 技术顾问建议

1. **MVP 路径**：冻结默认 PAE + tokenizer stability harness（方案 2.1/2.2）。
2. **终极路径**：PAE full + targeted MCR/SCR ablation + 若必要做轻量 fine-tune。
3. **实施顺序**：
   1. cache PAE/SD-VAE/VA-VAE/RAE；
   2. 跑统一 MeanFlow/DDT-B stability harness；
   3. 只在 PAE 明显有问题时考虑 MCR schedule 或 PAE-v2；
   4. end-to-end PAE × MeanFlow 放 Phase 4 后。
4. **不该在本论文做的方向**：完整 OT/Wasserstein tokenizer 训练与 MeanFlow-aware PAE v2 可作为后续论文，不要拖垮 PDM-3 主线。

---

## 3. 探索区域 3：DDT encoder/decoder ratio 在 PAE latent 下的重平衡

### 3.1 发现的方案全景

- **方案 3.1：静态 ratio proxy sweep（MVP，已知但缩短）**  
  不做 80 epoch 大扫，先做 20–40 epoch 小模型/中模型 proxy。

- **方案 3.2：semantic gap 指标预测 encoder 需求（新发现）**  
  用 linear probe、CKA、mutual information proxy、encoder feature drift 估计 PAE latent 已含多少语义。

- **方案 3.3：LayerDrop / once-for-all DDT supernet（跨领域）**  
  训练一个可裁剪 encoder-depth 的 DDT，通过 shared weights 选择 ratio。

- **方案 3.4：time-dependent dynamic depth（新发现）**  
  不同 `(r,t)` / noise level 对 semantic encoder 的需求不同，按 time 动态选择 encoder depth。

- **方案 3.5：decoder-only / ordinary DiT fallback（必须 baseline）**  
  0En/全部 decoder 是判断 DDT 是否仍有正贡献的关键负控。

- **方案 3.6：encoder sharing schedule 在 PAE latent 下加速 few-step（已知扩展）**  
  1-NFE 不受益，但 2/4/8 NFE 可用更激进 sharing。

- **方案 3.7：非对称宽度而非只调深度（新发现）**  
  PAE latent channel 大，可能需要 wide input/head 而非更深 encoder。

- **方案 3.8：NAS/Once-for-All 完整搜索（未来工作）**  
  系统搜索 ratio、width、patch size、attention backend，但不建议主线使用。

### 3.2 方案详情

#### 方案 3.1：静态 ratio proxy sweep

**一句话原理**：用短训 proxy 找趋势，避免在 B300 主训前盲押 20En/4De 或 16En/8De。

**灵感来源**：DDT 原论文发现模型越大越 encoder-heavy，DDT-XL/2 报告 22En/6De：<https://arxiv.org/abs/2504.05741>；官方 README 列出 22en6de checkpoint：<https://github.com/MCG-NJU/DDT>

**具体实现步骤**：

推荐 sweep：

| 配置 | 目的 |
|---|---|
| 20En/4De | DDT-L 原始倾向 |
| 16En/8De | PAE 假设下的主候选 |
| 12En/12De | 平衡候选 |
| 8En/16De | decoder-heavy |
| 0En/24De | DiT fallback/负控 |
| 22En/6De | DDT-XL 风格，若参数预算允许 |

训练设置：
- 同一 PAE latent cache；
- DDT-B 或 L/2 reduced width；
- 20–40 epoch 或固定 token budget；
- 先 no-CFG，记录 loss、FID-10K、class acc of generated samples、JVP metrics。

**涉及组件**：DDT architecture config、MeanFlow objective。

**对接方式**：config 中 `encoder_depth`、`decoder_depth`、`share_embed`、`patch_size` 参数化。

**难度评估**：中。需要可靠小规模 FID proxy。

**预期效果**：明确 PAE latent 下 ratio 偏移方向；减少主训风险。

**风险与不确定性**：短训最优不等于长训最优；要结合 semantic-gap 指标。

---

#### 方案 3.2：semantic gap 指标预测 encoder 需求

**一句话原理**：如果 PAE latent 已能线性预测 ImageNet class 且与 DINO feature 高 CKA，DDT encoder 可变浅；否则仍需重 encoder。

**具体实现步骤**：

1. Linear probe：

   ```python
   z = pae_encode(x)                      # B,C,H,W
   feat = z.mean(dim=(2,3))               # B,C
   train_logistic_regression(feat, label)
   ```

2. DINO/PAE CKA：比较 PAE latent patch features 与 DINOv2 patch features。
3. Encoder contribution：训练中记录不同 encoder depth 的 `s_t` 对 class label 的 linear probe acc，以及 `s_t` 与 input latent 的 CKA。
4. Semantic gap score：

   ```text
   gap = 1 - normalized(linear_probe_acc) + 1 - CKA(PAE,DINO) + encoder_feature_drift
   ```

**涉及组件**：PAE latent、DDT encoder。

**难度评估**：低到中。

**预期效果**：给 ratio 选择提供解释，增强论文说服力。

**风险与不确定性**：linear probe 高不代表生成需要的所有语义都已编码；只能辅助。

---

#### 方案 3.3：LayerDrop / once-for-all DDT supernet

**一句话原理**：训练时随机丢 encoder block，让一个模型覆盖多个 encoder depth，训练后选择最优子网。

**灵感来源**：
- LayerDrop 使 Transformer 可按需裁剪深度：<https://arxiv.org/abs/1909.11556>
- Once-for-All 训练一个 supernet 再 specialize 子网络：<https://arxiv.org/abs/1908.09791>

**具体实现步骤**：

1. 定义最大 encoder depth，例如 20；训练时随机选择 prefix depth：

   ```python
   depth_choices = [8, 12, 16, 20]
   d = random.choice(depth_choices)
   s = encoder(z, t, y, active_depth=d)
   u = decoder(z, s, r, t, y)
   ```

2. 用 sandwich rule：每个 batch 同时训练最小、最大、随机 depth，累积 loss。
3. 每 10K step 用 EMA 权重评估各 depth 的 FID-10K/JVP。
4. 主训前固定选一个 depth，避免最终模型动态复杂。

**涉及组件**：DDT encoder。

**难度评估**：中高。训练复杂且吞吐下降。

**预期效果**：以 1.5–2× sweep 成本覆盖多个 ratio。

**风险与不确定性**：不同 depth 共享权重导致 proxy 与独立训练不一致；可作为 ratio search，不作为最终训练。

---

#### 方案 3.4：time-dependent dynamic depth

**一句话原理**：不同 time/noise level 需要不同语义抽取深度；让 encoder depth 成为 `t` 或 `Δ=t-r` 的函数。

**灵感来源**：SkipNet/BlockDrop 学习动态跳层，按输入选择计算路径：<https://arxiv.org/abs/1711.09485>、<https://arxiv.org/abs/1711.08393>

**具体实现步骤**：

低风险版本：手工 schedule。

```python
def depth_by_time(t):
    # 示例，需实验验证
    # 高噪声阶段 class condition 更重要，可能需要较深 encoder；低噪声阶段 latent 本身语义清晰，可浅
    return torch.where(t > 0.7, 20, torch.where(t > 0.3, 16, 12))
```

进阶版本：learned gate。

```python
gate_logits = gate_mlp(time_embed(t))          # depth choices
weights = gumbel_softmax(gate_logits)
s_all = [encoder_prefix(z,t,y,d) for d in depth_choices]
s = sum(w[:,None,None] * s_d for w, s_d in zip(weights.T, s_all))
loss_budget = lambda_budget * expected_depth(weights)
```

**涉及组件**：DDT encoder、training/inference。

**难度评估**：高。动态图影响吞吐与可复现性。

**预期效果**：few-step 推理可节省计算；1-NFE 下质量提升不确定。

**风险与不确定性**：复杂度高、论文叙事分散；建议未来工作。

---

#### 方案 3.5：decoder-only / ordinary DiT fallback

**一句话原理**：如果 PAE latent 已足够语义化，DDT encoder 可能冗余；必须用 0En baseline 证明 DDT 仍有价值。

**具体实现步骤**：

- `encoder_depth=0` 时：

  ```python
  s = None 或 s = shallow_embed(z,t,y)
  u = decoder(z, s, r, t, y)
  ```

- 保持参数量近似：把 encoder 参数转移到 decoder depth/width，避免不公平。

**涉及组件**：DDT architecture。

**难度评估**：低。

**预期效果**：强负控；若 0En 接近或超过 DDT，论文 architecture contribution 需改写。

**风险与不确定性**：如果参数重分配不公平，结论不稳。

---

#### 方案 3.6：encoder sharing schedule 在 PAE latent 下加速 few-step

**一句话原理**：PAE latent 更平滑时，相邻 denoising step 的 encoder output 可能更相似，可更激进复用 `s_t`。

**灵感来源**：DDT 原论文提出相邻 denoising step 共享 self-condition，并用动态规划找 sharing strategy：<https://arxiv.org/abs/2504.05741>

**具体实现步骤**：

1. 对 2/4/8 NFE 采样，记录：

   ```python
   sim = cosine(s_t.flatten(1), s_t_next.flatten(1))
   ```

2. 用阈值共享：若 `sim > 0.98`，复用 encoder；否则重新算。
3. 对 PAE vs SD-VAE 比较可共享比例。

**涉及组件**：推理流程。

**难度评估**：中。

**预期效果**：对 1-NFE 无帮助；对 few-step fallback 可加速 1.5–2×。

**风险与不确定性**：PDM-3 主目标强调 1-NFE，此 trick 只能作为 few-step 附加贡献。

---

#### 方案 3.7：非对称宽度而非只调深度

**一句话原理**：PAE-f16d32 的 channel 远高于 SD-VAE；瓶颈可能在输入/head 宽度而不是 encoder depth。

**灵感来源**：RAE 论文指出高维 representation latent 对 DiT 架构提出挑战，并使用 lightweight, wide DDT head 获得强 ImageNet 结果：<https://arxiv.org/abs/2510.11690>

**具体实现步骤**：

1. patch embedding 从 `C=32` 到 model dim 的投影用 wide stem：

   ```python
   stem = nn.Sequential(
       nn.Conv2d(32, 4*hidden, kernel_size=1),
       nn.SiLU(),
       nn.Conv2d(4*hidden, hidden, kernel_size=patch, stride=patch),
   )
   ```

2. encoder/decoder depth sweep 同时比较：
   - narrow stem + deep encoder；
   - wide stem + shallow encoder；
   - wide stem + decoder-heavy。

**涉及组件**：DDT input embedding、latent normalization。

**难度评估**：中。

**预期效果**：可能比单纯加深 encoder 更适合 PAE/RAE 高维 latent。

**风险与不确定性**：增加参数与显存；需要控制 <1B。

### 3.3 技术顾问建议

1. **MVP 路径**：方案 3.1 + 3.2 + 3.5。不要先做动态 depth。
2. **终极路径**：静态最优 ratio + wide stem + few-step sharing 附加实验。
3. **实施顺序**：
   1. semantic-gap 统计；
   2. 0En/12En/16En/20En proxy；
   3. 选 1–2 个 ratio 做中训；
   4. 只有在 ratio 不稳定时才做 LayerDrop supernet。
4. **未来工作**：time-dependent dynamic depth 与 NAS，很有趣但不应进入主论文关键路径。

---

## 4. 探索区域 4：1-NFE CFG 集成与质量提升

### 4.1 发现的方案全景

- **方案 4.1：omega-conditioned built-in CFG（新发现，强推荐）**  
  不训练固定 CFG scale 的模型，而把 `omega` 作为输入，使一个模型支持 inference sweep。

- **方案 4.2：DDT 架构感知 CFG：encoder-only / decoder-only / both / 双 omega（新发现）**  
  利用 DDT 语义/细节分工，分别控制 class guidance 注入位置。

- **方案 4.3：interval-conditioned guidance schedule（新发现）**  
  把 guidance interval 从 sampling trick 变成 `(r,t)` 条件下的平均 guidance 强度。

- **方案 4.4：latent CFG rescale / dynamic thresholding（已知扩展）**  
  用 Imagen dynamic thresholding、guidance rescale 思想抑制小模型高 omega 过饱和。

- **方案 4.5：PAE manifold projection / CFG++ 风格修正（新发现）**  
  用 PAE encode-decode 或 latent projector 把 guided one-step 输出拉回 PAE manifold。

- **方案 4.6：CFG-free / training-free guidance：SAG/PAG（可选）**  
  对 1-NFE 主线价值有限，因为通常需要额外 forward，但可作为 few-step 质量补丁。

- **方案 4.7：1-NFE + few-NFE mixed / any-step model（强推荐 fallback）**  
  一个模型支持 1/2/4/8 NFE，用 few-step 弥补 1-step 质量上限。

- **方案 4.8：小模型 CFG 病理监控（MVP）**  
  高 guidance 下不仅看 FID，还看 recall、class accuracy、RGB/latent norm、saturation、mode collapse。

### 4.2 方案详情

#### 方案 4.1：omega-conditioned built-in CFG

**一句话原理**：MeanFlow built-in CFG 若固定 omega，会失去 inference-time sweep；把 omega 连续嵌入网络，让训练覆盖多个 guidance strength。

**灵感来源**：
- CFG 原论文通过混合 conditional/unconditional score 在质量与多样性之间权衡：<https://arxiv.org/abs/2207.12598>
- MeanFlow 将 CFG 内置到 velocity field，以避免 sampling-time 双 forward（任务书与 MeanFlow 解读均指出）。
- 非官方 PyTorch MeanFlow repo 指出其隐式 CFG 固定 scale，inference 不可调，是实际限制：<https://github.com/haidog-yaqub/MeanFlow>

**具体实现步骤**：

1. 训练时采样 `omega`：

   ```python
   omega = sample_omega(batch)  # e.g. mixture: 50% 0/1, 50% Uniform[1,3.5]
   omega_emb = fourier_embed(omega)
   ```

2. DDT encoder/decoder 都接收 `omega_emb`，但可通过 mask 做 ablation。

3. target 构造：如果训练框架能得到 conditional/unconditional velocity target，构造：

   ```python
   v_cfg = omega * v_cond + (1 - omega) * v_uncond
   ```

   若当前代码只有 data-pair velocity `v=eps-z`，则需要用 condition dropout 训练同一 vector field，并在 MeanFlow identity 中对 condition/uncondition 两个 forward 构造 guided target；这部分必须严格对照 MeanFlow 原论文实现，避免“伪 CFG”。

4. 推理时 sweep：`omega ∈ {1.0,1.5,2.0,2.5,3.0,3.5}`。

**涉及组件**：MeanFlow objective、DDT condition embedding、training sampler。

**对接方式**：config 增加 `omega_conditioned=True`、`omega_sampler`。

**难度评估**：中高。难点在正确复现 MeanFlow built-in CFG target。

**预期效果**：避免为每个 omega 训练一个模型；大幅提高 Phase 3 CFG 搜索效率。

**风险与不确定性**：omega-conditioned 模型可能在所有 omega 上都不如固定 omega 专家；可最后对最优 omega fine-tune 专家模型。

---

#### 方案 4.2：DDT 架构感知 CFG

**一句话原理**：DDT encoder 学语义、decoder 学细节；class guidance 不一定应同等注入两者。

**具体实现步骤**：

实验矩阵：

| 模式 | encoder class cond | decoder class cond | omega |
|---|---|---|---|
| both | yes | yes | shared |
| encoder-only | yes | no/weak | `ω_e` |
| decoder-only | no/weak | yes | `ω_d` |
| dual | yes | yes | `ω_e,ω_d` separate |

实现：

```python
s = encoder(z_t, t, class_emb * cond_mask_e, omega_e)
u = decoder(z_t, s, r, t, class_emb * cond_mask_d, omega_d)
```

建议训练时采样：

```python
omega_e ~ Uniform[0.5, 3.0]
omega_d ~ Uniform[0.5, 2.0]
```

**涉及组件**：DDT encoder/decoder condition injection。

**难度评估**：中。

**预期效果**：encoder-only guidance 可能提升 class fidelity 且减少 decoder 过饱和；dual guidance 可能最佳但复杂。

**风险与不确定性**：搜索空间变大；先在 short-run 下看 class accuracy vs FID/reconstruction artifacts。

---

#### 方案 4.3：interval-conditioned guidance schedule

**一句话原理**：guidance interval 在多步采样中是随 time 变化的 guidance；MeanFlow 的 `(r,t)` 表示一整段区间，因此可把区间内 guidance 的平均强度作为条件。

**灵感来源**：REPA/现代 DiT 常用 guidance interval；MeanFlow 直接建模 interval average velocity。

**具体实现步骤**：

1. 定义 continuous guidance schedule `g(τ)`，如只在 `[τ_min, τ_max]` 启用。
2. 对 MeanFlow interval 计算平均 guidance：

   ```python
   def avg_guidance(r, t, tau_min=0.0, tau_max=0.7, omega=2.0):
       overlap = (torch.minimum(t, tau_max) - torch.maximum(r, tau_min)).clamp_min(0)
       return omega * overlap / (t - r + 1e-6)
   omega_eff = avg_guidance(r, t)
   ```

3. 把 `omega_eff` 输入模型，target 用对应 guided field。
4. 对 1-NFE，`r=0,t=1` 时 `omega_eff` 变成 interval 平均值；对 few-step 则自然分段。

**涉及组件**：CFG sampler、MeanFlow condition。

**难度评估**：中。

**预期效果**：比固定 omega 更接近 guidance interval 的成功经验；提升 high omega 稳定性。

**风险与不确定性**：如果 MeanFlow 原生 built-in CFG 不支持连续 schedule，需重新推导 target；先作为 empirical trick。

---

#### 方案 4.4：latent CFG rescale / dynamic thresholding

**一句话原理**：高 CFG 会把样本推离数据范围/流形；对 guided delta 或 x0/latent 做 rescale/clipping，抑制过饱和。

**灵感来源**：
- Imagen 提出 dynamic thresholding 以支持大 guidance weight 并减少 saturation：<https://arxiv.org/abs/2205.11487>
- “Common Diffusion Noise Schedules and Sample Steps are Flawed” 提出 rescale classifier-free guidance 防止 over-exposure：<https://arxiv.org/abs/2305.08891>

**具体实现步骤**：

1. latent dynamic thresholding：

   ```python
   def dynamic_threshold_latent(z, p=0.995, max_val=3.0):
       s = torch.quantile(z.abs().flatten(1), p, dim=1).view(-1,1,1,1)
       s = torch.maximum(s, torch.ones_like(s))
       return torch.clamp(z, -s, s) / s * max_val
   ```

2. CFG delta rescale：

   ```python
   delta = u_cond - u_uncond
   delta_rescaled = delta * (u_uncond.flatten(1).std(dim=1) / (delta.flatten(1).std(dim=1)+1e-6)).view(-1,1,1,1)
   u_cfg = u_uncond + omega * ((1-phi)*delta + phi*delta_rescaled)
   ```

3. 1-NFE 输出后先 latent clamp/project，再 PAE decode。

**涉及组件**：推理流程；若 built-in CFG 无 cond/uncond 双输出，则用于 omega-conditioned 模型的 output correction。

**难度评估**：低到中。

**预期效果**：降低过饱和、提升 precision/recall balance；对 FID 改善需实测。

**风险与不确定性**：latent clipping 可能破坏 PAE decoder 的自然性；需要同时看 rFID/visual artifacts。

---

#### 方案 4.5：PAE manifold projection / CFG++ 风格修正

**一句话原理**：CFG++ 将高 guidance 问题解释为 off-manifold；PDM-3 可用 PAE encode-decode 作为近似 manifold projector。

**灵感来源**：CFG++ 提出 manifold-constrained CFG，认为传统 CFG 的模式坍塌/不可逆等问题源于 off-manifold：<https://arxiv.org/abs/2406.08070>

**具体实现步骤**：

1. 1-NFE 得到 latent `z0_hat`。
2. PAE projector：

   ```python
   with torch.no_grad():
       x_hat = pae_decoder(z0_hat)
       z_proj = pae_encoder(x_hat)
   z_final = (1 - alpha) * z0_hat + alpha * z_proj
   x_final = pae_decoder(z_final)
   ```

3. `alpha ∈ {0,0.25,0.5,0.75,1}` sweep。
4. 如果 PAE encoder costly，可训练一个小 latent projector `P(z)` 蒸馏 `E(D(z))`。

**涉及组件**：推理流程、PAE encoder/decoder。

**难度评估**：中。

**预期效果**：改善高 omega 下的 off-manifold artifacts；可能牺牲多样性。

**风险与不确定性**：额外 PAE encode/decode 成本可能使 1-NFE latency 优势下降；必须报告 end-to-end latency。

---

#### 方案 4.6：CFG-free / training-free guidance：SAG/PAG

**一句话原理**：通过自注意力扰动构造弱模型或结构退化样本，guidance away from degraded prediction，无需 classifier/condition。

**灵感来源**：
- Self-Attention Guidance (SAG) 在 ADM/Stable Diffusion/DiT 等模型上提升样本质量：<https://arxiv.org/abs/2210.00939>
- Perturbed-Attention Guidance (PAG) 通过替换 self-attention map 为 identity 构造退化预测：<https://arxiv.org/abs/2403.17377>

**具体实现步骤**：

在 DDT decoder 的某些 self-attention 层构造 perturbed forward：

```python
u_normal = net(z, r, t, y, perturb_attn=False)
u_weak   = net(z, r, t, y, perturb_attn=True, layers=['mid'])
u_guided = u_normal + scale_pag * (u_normal - u_weak)
```

**涉及组件**：DDT attention、推理。

**难度评估**：中。

**预期效果**：few-step 时可能改善结构；1-NFE 下需要额外 forward，违背单步成本优势。

**风险与不确定性**：对 class-conditional ImageNet 不一定优于 CFG；建议只作为 optional appendix。

---

#### 方案 4.7：1-NFE + few-NFE mixed / any-step model

**一句话原理**：不要把全部赌注押在 1-NFE；同一模型支持 1/2/4/8 steps，用户可按质量/速度取舍。

**灵感来源**：MeanFlow 本身支持 arbitrary `(r,t)` 更新；SplitMeanFlow 与 flow-map/shortcut/consistency 类方法都天然支持 few-step。2026 AnyFlow 等 flow-map distillation 工作也强调 any-step 生成（需进一步核验）。

**具体实现步骤**：

1. 训练覆盖多种 interval：`(0,1)`、`(0,0.5),(0.5,1)`、四段、八段。
2. 推理 API：

   ```python
   def sample(nfe):
       z = torch.randn(...)
       times = torch.linspace(1, 0, nfe+1)
       for t, r in zip(times[:-1], times[1:]):
           z = z - (t-r) * net(z, r, t, y)
       return decode(z)
   ```

3. FID report 同时列 1/2/4/8 NFE。

**涉及组件**：MeanFlow objective、sampler、DDT encoder sharing。

**难度评估**：低到中。

**预期效果**：即使 1-NFE FID 不达标，2/4-NFE 可能支撑主质量目标；增强产品价值。

**风险与不确定性**：论文目标若强调 1-NFE，few-step fallback 不能完全替代。

---

#### 方案 4.8：小模型 CFG 病理监控

**一句话原理**：高 omega 常提升 precision 但损害 recall/多样性并造成过饱和；只看 FID 会误判。

**具体实现步骤**：

每次 omega sweep 同时记录：
- FID/sFID/IS/Precision/Recall；
- generated sample class accuracy；
- RGB histogram saturation ratio；
- latent norm p99；
- duplicate/near-duplicate rate；
- per-class FID 或 class coverage。

**难度评估**：低。

**预期效果**：避免为了 FID 牺牲 recall/coverage；提前发现 reviewer 会问的问题。

### 4.3 技术顾问建议

1. **MVP 路径**：omega-conditioned CFG + both/encoder-only/decoder-only 三模式短训 + pathology metrics。
2. **终极路径**：interval-conditioned dual-omega CFG + latent rescale + few-step fallback。
3. **实施顺序**：
   1. 复现 MeanFlow built-in CFG；
   2. 加 omega embedding；
   3. 做 DDT placement ablation；
   4. 加 dynamic threshold/rescale；
   5. 最后考虑 CFG++ projector。
4. **未来工作**：SAG/PAG/CFG-free guidance 可作为 appendix 或后续，不应拖累主线。

---

## 5. 探索区域 5：Proposition 1 的可证伪性、反例与 fallback framing

### 5.1 发现的方案全景

- **方案 5.1：指标组替代单一 JVP norm（强推荐）**  
  JVP norm 只是 proxy；加入 target variance、gradient SNR、finite-difference agreement、Jacobian spectral norm、Lyapunov proxy。

- **方案 5.2：加固 Proposition 1 的数学假设（必须）**  
  MCR 只约束 decoder 感知连续性，不足以推出 score/velocity Lipschitz；需要加上 bi-Lipschitz、latent density bounded、time interval away from endpoints 等假设。

- **方案 5.3：主动构造反例（必须）**  
  构造“满足 MCR 但 JVP 不稳定”的情况，迫使 proposition 更严谨。

- **方案 5.4：多 tokenizer、多 seed、paired statistical test（MVP）**  
  不做 PAE vs SD-VAE 两点比较；做 SD-VAE/VA-VAE/RAE/PAE/PAE-MCR-ablated。

- **方案 5.5：若 MCR fails，改成 architectural co-design framing（Plan B）**  
  不要硬讲理论；把 PAE 作为 empirical tokenizer，主贡献转向 DDT × MeanFlow。

- **方案 5.6：robust control / Lyapunov 视角（跨领域）**  
  直接分析 one-step map `Φ(z)=z-(t-r)uθ` 的局部扩张/收缩，而不是只看 `u` 的 JVP。

- **方案 5.7：benchmark stress test（新发现）**  
  ImageNet-256 可能太容易或太 noisy；用高 LR/bf16/small batch/ImageNet-512 subset/stress split 放大稳定性差异。

### 5.2 方案详情

#### 方案 5.1：指标组替代单一 JVP norm

**一句话原理**：训练稳定性是 target 方差、梯度噪声、局部 Jacobian、optimizer dynamics 的共同结果，不能只看 JVP norm。

**具体实现指标**：

1. **JVP norm distribution**：`du_dt` p50/p95/p99/p999。
2. **Target variance**：

   ```python
   target_var = target.flatten(1).var(dim=0).mean()
   ```

3. **Gradient SNR**：同一 batch split 成 microbatches，估计梯度均值/方差。

   ```python
   snr = norm(mean_grad) / (sqrt(sum(var_grad)) + eps)
   ```

4. **Finite-difference agreement**：`||JVP-FD||/||FD||`。
5. **Local Jacobian spectral norm**：power iteration 估计 `J_u` 最大奇异值。
6. **One-step map Lipschitz**：

   ```python
   Phi = lambda z: z - (t-r)*net(z,r,t,y)
   lip_phi = ||Phi(z+δ)-Phi(z)|| / ||δ||
   ```

7. **Loss Hessian trace / sharpness proxy**：用 Hutchinson 估计或 SAM-style perturbation。
8. **NaN frequency / max stable LR**：工程稳定边界。

**涉及组件**：实验评估脚本。

**难度评估**：中。

**预期效果**：即使 JVP norm 不显著，也可能从 target variance/gradient SNR 找到 PAE 优势；或反过来及时否定理论。

**风险与不确定性**：指标过多导致 p-hacking。需要预注册主指标：JVP p99、target variance、NaN rate、max stable LR。

---

#### 方案 5.2：加固 Proposition 1 的数学假设

**一句话原理**：从 MCR 到 MeanFlow JVP bounded 中间跨了多个不自动成立的环节，必须列出附加假设。

**建议改写为有条件命题**：

> 若 PAE decoder `D` 在 latent data support 的邻域上是 bi-Lipschitz，latent density `p_z` 在紧支撑上上下有界且 score Lipschitz，flow path 避开 `t=0/1` endpoint，且 DDT/MeanFlow 网络类满足局部 Lipschitz regularity，则 MCR 降低 `D` 的局部 Lipschitz 上界与 latent manifold condition number，从而降低 MeanFlow target 的方差上界。

必须声明：
- MCR 是经验 regularizer，不等于严格全局 Lipschitz bound；
- LPIPS/VFM perceptual metric 与 L2 只在局部、数据 manifold 附近近似等价；
- `C(T,L_D,p_z)` 在 endpoint 可能发散，所以实验中需单独分析 time buckets。

**涉及组件**：理论章节。

**难度评估**：中高。

**预期效果**：降低 reviewer 对“跳步证明”的攻击。

**风险与不确定性**：命题变弱；但弱而正确胜过强而易反例。

---

#### 方案 5.3：主动构造反例

**一句话原理**：如果能构造反例，就能知道 proposition 需要哪些假设；这比被 reviewer 构造更好。

**反例类型**：

1. **感知不敏感折叠**：decoder 对某些 latent 高频方向变化不敏感，MCR loss 小，但 latent distribution 在该方向高度折叠，score/JVP 可很大。
2. **类簇过度分离**：SCR 让 class clusters 很紧、类间边界陡峭，conditional/unconditional CFG 混合时 velocity field 高曲率。
3. **decoder 平滑但 network time embedding 不平滑**：MCR 约束 tokenizer，不约束 DDT 的 `∂u/∂t`。
4. **anisotropic latent covariance**：局部连续但 covariance condition number 极大，小方差方向 score 爆炸。

**具体 mini-experiment**：

在 2D synthetic mixture 上训练 toy autoencoder：
- A：普通 AE；
- B：MCR-like smooth decoder；
- C：smooth decoder + anisotropic latent scaling。  
然后训练 tiny MeanFlow，比较 JVP/NaN/target variance。若 C 爆炸，说明 MCR 需加 covariance/density 假设。

**难度评估**：低到中。

**预期效果**：理论章节更可信。

---

#### 方案 5.4：多 tokenizer、多 seed、paired statistical test

**一句话原理**：Prop 1 的证据必须是分布级统计，不是单次训练曲线。

**具体实验设计**：

Tokenizers：
- SD-VAE；
- VA-VAE/LightningDiT；
- RAE；
- PAE full；
- PAE -MCR 或 high-MCR（若可得）。

Seeds：至少 3 个 seed，短训 20K–50K step。

统计：
- 对同一 batch/time samples 计算 paired JVP metrics；
- bootstrap confidence interval；
- Mann-Whitney 或 paired t-test 仅作为辅助，不要过度统计显著性叙事。

**难度评估**：中。

**预期效果**：避免“差异在噪声内”的尴尬。

**风险与不确定性**：多 tokenizer 归一化不公平；必须统一 latent scaling 与 model capacity。

---

#### 方案 5.5：MCR fails 后的 fallback framing

**一句话原理**：如果三个推论都失败，不要硬保 Proposition 1；把理论降级，论文改成 empirical/architectural co-design。

**Plan B framing**：

> PDM-3 studies a three-level factorization for efficient image generation: representation-level manifold shaping, architecture-level semantic/detail decoupling, and objective-level average-velocity learning. While PAE’s MCR does not independently explain MeanFlow stability, the combined system reveals which factorization choices matter empirically.

主贡献改为：
- DDT × MeanFlow 的首次系统实现；
- DDT encoder tangent/stop-gradient/SplitMeanFlow 的稳定训练配方；
- PAE/RAE/SD-VAE tokenizer 对 MeanFlow 的系统对比；
- 1-NFE/few-NFE quality-speed tradeoff。

**Plan C**：
- 若 DDT × MeanFlow 也不稳定：转向 SplitMeanFlow objective；
- 若 PAE 无收益：使用 RAE/VA-VAE，论文聚焦 DDT × MeanFlow；
- 若 1-NFE FID 不达标：转向 any-step/few-step 模型，主打 2/4-NFE。

**难度评估**：低，属于叙事策略。

**风险与不确定性**：投稿档次可能下降；但比理论站不住更安全。

---

#### 方案 5.6：robust control / Lyapunov 视角

**一句话原理**：稳定采样关心的是 flow map 是否放大扰动，而非单独的 velocity JVP。

**具体实现步骤**：

定义 one-step/few-step map：

```python
def Phi(z, r, t, y):
    return z - (t-r).view(-1,1,1,1) * net(z,r,t,y)
```

估计局部扩张率：

```python
delta = normalize(torch.randn_like(z)) * sigma
lip = (Phi(z+delta,r,t,y) - Phi(z,r,t,y)).flatten(1).norm(dim=1) / sigma
```

若 `lip > 1` 的比例高，说明采样 map 局部扩张，可能导致 1-NFE artifacts。

**涉及组件**：评估脚本。

**难度评估**：低。

**预期效果**：比 JVP norm 更贴近生成稳定性。

**风险与不确定性**：局部 Lipschitz 与 FID 的关系仍需实证。

---

#### 方案 5.7：benchmark stress test

**一句话原理**：ImageNet-256 主指标可能看不出稳定性差异；用 stress 条件放大差异。

**具体 stress**：
- bf16 all vs fp32 JVP；
- LR sweep：`5e-5,1e-4,2e-4,4e-4`；
- batch size 128 vs 256；
- endpoint-heavy time sampler；
- ImageNet-512 subset；
- high omega CFG。

**预期效果**：即使主 FID 差异小，也可验证 PAE 是否提升稳定边界。

**风险与不确定性**：stress test 可能被 reviewer 认为人为；需定位为 supplementary stability analysis。

### 5.3 技术顾问建议

1. **MVP 路径**：方案 5.1 + 5.4，预注册主指标：JVP p99、target variance、NaN rate、max stable LR。
2. **终极路径**：有条件 Proposition + counterexample + metric suite + PAE ablation。
3. **实施顺序**：
   1. 先在 Phase 1 建 telemetry；
   2. Phase 2 做 tokenizer paired comparison；
   3. 若至少 2/3 推论成立，再写理论主线；
   4. 若失败，立即切换 Plan B。
4. **未来工作**：完整 robust-control 证明与 Wasserstein gradient-flow 理论可作为后续，不必强塞主论文。

---

## 6. 意外发现与路线级问题

### 6.1 意外发现 A：MF-RAE 已经直接验证“语义 latent × MeanFlow”，且发现 naive training 爆炸

**一句话原理**：用 RAE 语义 latent 替代 SD-VAE latent 训练 MeanFlow，但需要专门稳定训练策略。

**来源**：MeanFlow Transformers with Representation Autoencoders（arXiv:2511.13019）报告：RAE latent 下 naive MeanFlow 会严重梯度爆炸；采用 Consistency Mid-Training、flow matching teacher distillation、bootstrapping 后，ImageNet-256 1-step FID 2.03，并降低训练/采样成本：<https://arxiv.org/abs/2511.13019>

**对 PDM-3 的影响**：
- 正面：说明 PAE/RAE 类语义 latent 与 MeanFlow 是值得做的，不是空想。
- 负面：它直接挑战 Proposition 1 的“语义/连续 latent 自动稳定 JVP”直觉。RAE 语义 latent 反而可能导致 gradient explosion。
- 竞争：PDM-3 必须与 MF-RAE 比较，至少解释 PAE 相比 RAE 的差别（MCR/SSR/SCR 显式流形塑形）以及 DDT 的新增价值。

**建议动作**：
1. 把 MF-RAE 列为必读和核心 baseline；
2. 复现其稳定技巧中的至少一个：consistency mid-training 或 teacher distillation；
3. 在论文中明确：PDM-3 与 MF-RAE 的差别是 PAE 的显式 manifold regularization + DDT 解耦架构，而不仅是“换 tokenizer”。

---

### 6.2 意外发现 B：RAE/RAEv2/PAE 竞争会快速压缩 novelty

**来源**：RAE 论文提出 frozen DINO/SigLIP/MAE encoder + trained decoder，并报告 ImageNet-256 no guidance FID 1.51、with guidance 1.13：<https://arxiv.org/abs/2510.11690>；其 GitHub 已开源：<https://github.com/bytetriper/RAE>。搜索还发现 Improved Baselines with Representation Autoencoders（arXiv:2605.18324，需进一步核验）声称 RAEv2 与 REPA 机制互补，80 epoch 达 gFID 1.06。

**对 PDM-3 的影响**：
- 单纯“PAE tokenizer 更好”不是足够 novelty；
- 必须强调 MeanFlow/1-NFE、DDT factorization、JVP 稳定性这三个 PAE/RAE 论文没有系统解决的点。

**建议动作**：把 RAE/RAEv2 作为 tokenizer baseline，避免 reviewer 说你只挑弱 VAE 对比。

---

### 6.3 意外发现 C：JVP-free / flow-map 系路线正在变成主流 fallback

**来源**：
- SplitMeanFlow 用区间分裂一致性绕开 JVP：<https://arxiv.org/abs/2507.16884>
- 2026 年 AnyFlow/flow-map distillation 类工作（需进一步核验）强调 arbitrary time-pair transition 与 any-step generation。
- Riemannian MeanFlow 推出 manifold flow map/semigroup identities 与稳定技巧：<https://arxiv.org/abs/2602.07744>

**对 PDM-3 的影响**：如果 PDM-3 仍坚持 PyTorch JVP 路线，工程风险和 reviewer 质疑都会高。需要至少实现 SplitMeanFlow-style fallback。

---

### 6.4 意外发现 D：纯 1-NFE 可能不是最终最优叙事

**来源**：RMFlow（arXiv:2602.00849）明确指出 MeanFlow 的纯 1-NFE generation often cannot yield compelling results，提出 coarse 1-NFE + noise-injection refinement：<https://arxiv.org/abs/2602.00849>。W-Flow（arXiv:2605.11755，需进一步核验）据称通过 Wasserstein gradient flows 达到 one-step ImageNet-256 FID 1.29，说明 one-step 竞争正在变激烈。

**对 PDM-3 的影响**：
- 1-NFE FID ≤ 3.0 目标可行，但未必足够新；
- with-CFG FID ≤ 1.3 的主目标可能仍依赖 few-step/guidance interval；
- 建议报告 1/2/4/8 NFE Pareto curve，而不是只押 1-NFE。

---

### 6.5 意外发现 E：PAE-f16d32 的系统成本可能被低估

PAE-f16d32 latent 是 `32×32×32`，相比常见 SD-VAE `4×32×32` channel 增加 8×。这带来：

- patch embedding 与 first/last projection 参数/激活增大；
- DDT encoder/decoder self-attention token 数是否不变取决于 patch size，但 channel projection 成本明显上升；
- PAE decoder/encoder 的 end-to-end 1-NFE latency 可能吞掉 MeanFlow 单步收益；
- 若加入 PAE manifold projection，则更需报告端到端 latency。

**建议动作**：每个实验报告三种成本：
1. generator NFE/GFLOPs；
2. tokenizer decode GFLOPs；
3. total wall-clock latency。

---

### 6.6 意外发现 F：VAR/pixel-space 路线仍是路线级威胁，但不建议当前切换

VAR（arXiv:2404.02905）在 ImageNet-256 报告强 AR 结果；JiT/PixelDiT/DiP 等 pixel-space 路线也快速发展。它们说明 latent DiT 不是唯一答案。但 PDM-3 的时间/算力/生态已经围绕 PAE/DDT/MeanFlow，当前切换会失去可控性。

**建议动作**：在 related work 中正面讨论，不在本项目主线切换。

---

## 7. 最终整体建议

### 7.1 对 PDM-3 整体方向的判断

**信心评级：中高潜力 / 中高风险。**

**为什么有潜力**：
- 三个组件作用层级确实不同：latent geometry、network factorization、training objective；
- PAE/RAE 类工作证明 representation-aware tokenizer 对 ImageNet latent diffusion 极其关键；
- MeanFlow 与 flow-map 类方法是 one/few-step 生成的重要方向；
- DDT 的 semantic/detail 解耦与 MeanFlow 的 content/displacement decomposition 有自然契合点。

**为什么风险高**：
- MeanFlow JVP 在 PyTorch + bf16 + DDT 分叉路径下可能成为最大工程瓶颈；
- MF-RAE 已经显示语义 latent 下 naive MeanFlow 会梯度爆炸，直接挑战“PAE 自动稳定”直觉；
- PAE、DDT、MeanFlow 都是强组件，三者叠加后如果提升不显著，reviewer 会说是 stacking tricks；
- <90 H100-day 预算下，要同时复现多个 SOTA、做主训、做理论 ablation，非常紧。

### 7.2 最大的 1–2 个 dealbreaker

1. **JVP/target instability dealbreaker**：DDT × MeanFlow full-chain JVP 在 PAE latent 下不稳定或吞吐不可接受。如果没有 SplitMeanFlow/stop-tangent fallback，Phase 3 可能失败。
2. **理论证据 dealbreaker**：PAE 的 JVP norm/NaN/max LR 相比 RAE/VA-VAE 没有显著优势，Proposition 1 失去支撑。此时必须快速切换 framing。

### 7.3 Reviewer 最可能 challenge 的点

- “这是不是只是把 PAE、DDT、MeanFlow 三个 SOTA 组件堆起来？”
- “PAE 的 MCR 到 MeanFlow JVP 稳定性的证明假设是否过强/不可验证？”
- “为什么不比较 RAE/MF-RAE/RAEv2？”
- “DDT ratio 是否公平调参？参数量、训练 token budget、latent channel 数是否公平？”
- “1-NFE 的端到端 latency 是否真的低，还是 PAE decoder/projection 抵消收益？”
- “CFG built-in 是否牺牲 inference-time 可调性？omega sweep 是否公平？”
- “FID 改善是否来自更多 compute 或更好的 tokenizer，而非 DDT × MeanFlow 协同？”

### 7.4 如果只能给三条建议

1. **先建稳定性闸门，再谈 SOTA 主训。**  
   Phase 0–2 必须产出 JVP p99、target variance、NaN rate、max stable LR、FD agreement 的 tokenizer × architecture 对比矩阵。没有这张表，不要用 B300 跑全模型。

2. **把 SplitMeanFlow/JVP-free objective 当成必备 fallback，而不是 optional。**  
   PyTorch/Blackwell/bf16 + DDT full-chain JVP 的失败概率不低；JVP-free 路线也是近期文献趋势。

3. **把 MF-RAE/RAE 放进核心对照，主动重写 novelty。**  
   PDM-3 的创新不能是“语义 autoencoder + MeanFlow”，因为外部已有近似路线；必须是 PAE manifold regularization × DDT factorization × MeanFlow/flow-map stability 的系统 co-design。

### 7.5 如果 Proposition 1 三个推论全部失败：Plan B / Plan C

#### Plan B：砍掉强理论，保留 empirical + architectural co-design

标题/主线改为：

> Decoupled Average-Velocity Transformers for Fast Latent Image Generation

核心贡献：
- DDT × MeanFlow 的实现与稳定训练；
- stop-s-tangent / fp32 JVP island / SplitMeanFlow hybrid；
- PAE/RAE/VA-VAE tokenizer 对 MeanFlow 的系统比较；
- ImageNet-256 1/2/4 NFE Pareto curve。

PAE 角色：best tokenizer 或 empirical accelerator，不再声称 MCR 保证 JVP。

#### Plan C：若 JVP MeanFlow 本身失败，转向 SplitMeanFlow / flow-map model

主线改为：

> JVP-Free Decoupled Flow Maps in Prior-Aligned Latent Spaces

核心贡献：
- 用 Interval Splitting Consistency 训练 DDT；
- DDT encoder/decoder 与 flow-map interval 的结构对齐；
- PAE latent 下的 few-step/one-step 生成。

理论点：从 MCR ⇒ JVP 稳定，改为 PAE latent manifold ⇒ interval consistency target variance / flow-map contraction 更低。

#### Plan D：若 PAE 不优于 RAE/VA-VAE

- 使用 RAE 或 VA-VAE 做主 tokenizer；
- 把 PAE 放入 negative result；
- 论文贡献聚焦 DDT × MeanFlow 或 DDT × SplitMeanFlow；
- 若结果不够 SOTA，转短论文/工作坊，强调系统性 ablation 与工程稳定配方。

---

## 8. 建议的 15 周更新版执行计划

| 阶段 | 周次 | 必做事项 | Go/No-Go |
|---|---:|---|---|
| Phase 0 | 1 | 环境、PAE/RAE/SD-VAE latent cache、DDT/MeanFlow 单元测试 | JVP-safe forward 能跑 |
| Phase 1 | 2–3 | fp32 JVP island、telemetry、DDT-B tiny training | 10K step 无 NaN |
| Phase 2A | 4–5 | tokenizer stability matrix：SD-VAE/VA-VAE/RAE/PAE | PAE 至少 2 指标有优势，否则 Prop 1 降级 |
| Phase 2B | 6 | DDT ratio proxy sweep + semantic-gap | 锁 1–2 个 ratio |
| Phase 2C | 7 | CFG omega-conditioned prototype + SplitMeanFlow prototype | 至少一个 objective 稳定 |
| Phase 3 | 8–10 | B300 主训：首选稳定 objective + 最优 ratio | 中途 FID proxy 不达标则切 few-step |
| Phase 4 | 11–12 | Prop 1 ablation / stop-tangent / tokenizer comparison | 决定理论 framing |
| Phase 5 | 13–15 | 写作，补 reviewer 预期对照 | 提交 |

---

## 9. 核心参考链接

### PDM-3 核心组件

- PAE: What Matters for Diffusion-Friendly Latent Manifold? Prior-Aligned Autoencoders for Latent Diffusion, arXiv:2605.07915 — <https://arxiv.org/abs/2605.07915>  
  GitHub — <https://github.com/ZhengrongYue/PAE>
- DDT: Decoupled Diffusion Transformer, arXiv:2504.05741 — <https://arxiv.org/abs/2504.05741>  
  GitHub — <https://github.com/MCG-NJU/DDT>
- MeanFlow: Mean Flows for One-step Generative Modeling, arXiv:2505.13447 — <https://arxiv.org/abs/2505.13447>
- SplitMeanFlow: Interval Splitting Consistency in Few-Step Generative Modeling, arXiv:2507.16884 — <https://arxiv.org/abs/2507.16884>

### 直接竞争/强相关

- RAE: Diffusion Transformers with Representation Autoencoders, arXiv:2510.11690 — <https://arxiv.org/abs/2510.11690>  
  GitHub — <https://github.com/bytetriper/RAE>
- MF-RAE: MeanFlow Transformers with Representation Autoencoders, arXiv:2511.13019 — <https://arxiv.org/abs/2511.13019>
- RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation, arXiv:2602.00849 — <https://arxiv.org/abs/2602.00849>
- Riemannian MeanFlow, arXiv:2602.07744 — <https://arxiv.org/abs/2602.07744>
- Discrete MeanFlow, arXiv:2605.12805 — <https://arxiv.org/abs/2605.12805>

### PyTorch / 数值稳定

- `torch.func.jvp` 文档 — <https://docs.pytorch.org/docs/stable/generated/torch.func.jvp.html>
- PyTorch AMP 文档 — <https://docs.pytorch.org/docs/stable/accelerator/amp.html>
- PyTorch `clip_grad_norm_` 文档 — <https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html>
- Neural ODE, arXiv:1806.07366 — <https://arxiv.org/abs/1806.07366>
- torchdiffeq — <https://github.com/rtqichen/torchdiffeq>
- Diffrax adjoints — <https://docs.kidger.site/diffrax/api/adjoints/>

### 多任务损失/梯度冲突

- GradNorm, arXiv:1711.02257 — <https://arxiv.org/abs/1711.02257>
- Multi-Task Learning Using Uncertainty to Weigh Losses, arXiv:1705.07115 — <https://arxiv.org/abs/1705.07115>
- PCGrad / Gradient Surgery, arXiv:2001.06782 — <https://arxiv.org/abs/2001.06782>
- CAGrad, arXiv:2110.14048 — <https://arxiv.org/abs/2110.14048>

### CFG / guidance

- Classifier-Free Diffusion Guidance, arXiv:2207.12598 — <https://arxiv.org/abs/2207.12598>
- Imagen / dynamic thresholding, arXiv:2205.11487 — <https://arxiv.org/abs/2205.11487>
- Common Diffusion Noise Schedules and Sample Steps are Flawed / guidance rescale, arXiv:2305.08891 — <https://arxiv.org/abs/2305.08891>
- CFG++, arXiv:2406.08070 — <https://arxiv.org/abs/2406.08070>
- Self-Attention Guidance, arXiv:2210.00939 — <https://arxiv.org/abs/2210.00939>
- Perturbed-Attention Guidance, arXiv:2403.17377 — <https://arxiv.org/abs/2403.17377>

### 动态深度/NAS

- SkipNet, arXiv:1711.09485 — <https://arxiv.org/abs/1711.09485>
- BlockDrop, arXiv:1711.08393 — <https://arxiv.org/abs/1711.08393>
- LayerDrop, arXiv:1909.11556 — <https://arxiv.org/abs/1909.11556>
- Once-for-All, arXiv:1908.09791 — <https://arxiv.org/abs/1908.09791>

