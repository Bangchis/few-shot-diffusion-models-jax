# Few-Shot DiT (FSDM) Architecture - JAX/Flax Port

## Pipeline Overview
1) **Encoder few-shot**: biến tập ảnh (set) `x_set ∈ ℝ^{B×ns×C×H×W}` thành embedding ngữ cảnh `hc`.
2) **Leave-One-Out (LOO)**: với mỗi phần tử trong set, encode phần còn lại → `c_i` dùng để điều kiện hóa chính phần tử đó.
3) **Backbone denoiser**: DiT (Diffusion Transformer) nhận ảnh nhiễu `x_t`, timestep `t`, conditioning `c`, trả về dự đoán (thường là `ε_θ`).
4) **Diffusion**: GaussianDiffusion JAX tính loss huấn luyện (MSE hoặc kèm VB/var learned) và sampling (DDPM/DDIM).
5) **Optional variational context**: posterior Gaussian trên `c`, KL(q||p) cộng vào loss.

## Encoder Few-Shot
- Files: `model/vit_jax.py`, `model/vit_set_jax.py`
- Input: `x_set ∈ ℝ^{B×ns×C×H×W}`
- Output: `hc ∈ ℝ^{B×hdim}` (hoặc token map nếu pool='none')
- sViT (set ViT): gộp ns ảnh theo kênh (Shifted Patch Tokenization), tạo patch tokens, qua Transformer, pool (cls/mean/agg) → `hc`.
- Tham số: `C,H,W` (kích thước ảnh), `ns` (set size), `hdim` (dim context), `patch_size`, `pool`.

## Leave-One-Out Conditioning
- Hàm `leave_one_out_c` (trong `model/vfsddpm_jax.py`):
  - Với phần tử i: encode set \ {i} → `hc_i`.
  - Variational (nếu dùng): `c_i = μ + σ ⊙ ε`, KL với p = N(0,I).
- Kết quả:
  - `c_set ∈ ℝ^{B×ns×hdim}`
  - film: `c = c_set.reshape(B·ns, hdim)`
  - lag: `c = c_set.reshape(B·ns, 1, hdim)` (1 token/element; có thể mở rộng)

**KL Gaussian**
```
KL = 0.5 * [ (σ_q^2 / σ_p^2) + (μ_q - μ_p)^2 / σ_p^2 - 1 + log σ_p^2 - log σ_q^2 ]
klc = mean_flat(KL) / log(2)
```

## DiT Denoiser (Few-Shot Conditioning)
- File: `model/set_diffusion/dit_jax.py`
- Input: `x ∈ ℝ^{B×C×H×W}`, `t ∈ ℤ^{B}`, conditioning `c`
  - film: `c ∈ ℝ^{B×hdim}` → Dense(hdim→hidden_size) + cộng vào time embedding
  - lag: `c ∈ ℝ^{B×N_c×hdim}` làm key/value cross-attention
- Quy trình:
  1. Pad 28→32 nếu cần; patchify size `p`, số patch `N = (H/p)^2`.
  2. Positional embedding sin-cos 2D: `pos_embed ∈ ℝ^{1×N×hidden_size}`.
  3. Timestep embedding: `t_emb = TimestepEmbedder(t)`.
  4. (Optional) Class embedding nếu class_cond.
  5. Conditioning:
     - film: `conditioning = t_emb + Dense(hdim→hidden_size)(c)`
     - lag: `conditioning = t_emb` (c dùng cho cross-attn)
  6. DiT blocks (depth L), AdaLN-Zero:
     - Tính `c_mod = Dense(6*hidden_size)(silu(conditioning))`
     - `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = split(c_mod)`
     - Self-attn: `x = x + gate_msa * MHA(modulate(LN(x), shift_msa, scale_msa))`
     - Nếu lag: cross-attn với `context_proj(c)`
     - MLP: `x = x + gate_mlp * MLP(modulate(LN(x), shift_mlp, scale_mlp))`
  7. Final layer: AdaLN + Dense về patch pixels, reshape lại (B,C,H,W), bỏ padding nếu có.
- Output: cùng shape input; là `ε` hoặc `x0` tùy `model_mean_type`.

## Diffusion (GaussianDiffusion JAX)
- File: `model/set_diffusion/gaussian_diffusion_jax.py`
- Tham số:
  - `betas` từ schedule (linear/cosine)
  - `model_mean_type`: EPSILON | START_X | PREVIOUS_X
  - `model_var_type`: FIXED_SMALL | FIXED_LARGE | LEARNED_RANGE | LEARNED
  - `loss_type`: MSE | RESCALED_MSE | KL | RESCALED_KL
- Training (MSE phổ biến):
```
x_t = sqrt(ᾱ_t) x0 + sqrt(1-ᾱ_t) noise
target = noise      (nếu model_mean_type=EPSILON)
loss = mean_flat((target - model_output)^2)
```
Nếu var learned (LEARNED_RANGE), thêm VB term cho phương sai.
- Sampling DDPM:
```
x_{t-1} = μ_θ(x_t,t,c) + σ_t * noise,  t = T-1..0
```
- DDIM: eta điều khiển mức ngẫu nhiên (eta=0 → deterministic).

## Conditioning gắn vào Diffusion
- `vfsddpm_loss`: tạo `c` bằng LOO, bọc `model_fn(x,t,_)` gọi DiT với `c` đã dựng; diffusion gọi `model_fn`.
- Loss cuối: `loss_diffusion` (+ `klc` nếu variational context).

## Tham số chính
- `image_size`, `in_channels`: kích thước/kênh ảnh.
- `sample_size (ns)`: số ảnh trong set.
- Encoder: `patch_size`, `hdim`, `pool`.
- DiT: `hidden_size`, `depth`, `num_heads`, `mlp_ratio`, `patch_size`, `context_channels`, `mode_conditioning` (film/lag), `learn_sigma`.
- Diffusion: `diffusion_steps`, `noise_schedule`, `timestep_respacing`, `use_kl`, `predict_xstart`, `rescale_timesteps`, `rescale_learned_sigmas`.
- Variational context: posterior MLP (μ, logvar), KL với N(0,I), sample `c = μ + exp(0.5 logvar)*ε`.

## Khác biệt chính so với bản PyTorch VFSDDPM
- Backbone DiT (Transformer) thay UNet.
- JAX/Flax/Optax, hỗ trợ pmap đa thiết bị.
- Lag hiện 1 token/element (có thể mở rộng nhiều token).
- Chưa port variational discrete/codebook; chỉ Gaussian.
- Logging/ckpt/sampling cần gắn thêm (main_jax cung cấp train loop cơ bản).

