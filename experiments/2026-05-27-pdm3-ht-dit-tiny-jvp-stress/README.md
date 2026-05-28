# PDM-3-HT Experiment 3 — DiT-Tiny / PAE-shape JVP Stress

本目录用于压测 PDM-3-HT 的 full-bundle Per-Patch MeanFlow JVP，在比前两步 toy 更接近真实训练的 proxy 上验证：

- PAE-shape latent tokens：`B x N x C`，默认 `C=32`，`N=256/1024`。
- LightningDiT-Tiny-like block：attention + MLP + per-token AdaLN conditioning。
- full-bundle JVP：一次 `torch.func.jvp` 同时覆盖所有 token 的 `(z_t, r_i, t_i)` tangent。
- central finite difference audit。
- MeanFlow-style stop-gradient target 的 backward/train-step stress。
- wall time 与 peak memory 记录。

注意：这不是官方 LightningDiT 代码，而是用于方法风险验证的 LightningDiT-like tiny proxy。它保留本实验关心的关键结构：全局 attention coupling、per-token time conditioning、PAE-like token shape。
