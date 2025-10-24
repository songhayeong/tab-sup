### Tab-SEDD (Sketch)

- Goal: mix SEDD’s discrete score-entropy scheme with Tab-DDPM’s tabular diffusion.  
- Key idea: build a block-uniform transition matrix for one-hot features, train an MLP/ResNet diffusion model on top.

### Quickstart

1. **Install deps**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Download raw datasets**  
   ```bash
   python start/dataset/download_dataset.py
   ```
   This pulls selected UCI tables (e.g., Online Shoppers) into `data/<name>`.

3. **Prepare tabular tensors**  
   ```bash
   python start/prepare_shoppers_dataset.py
   ```
   - Outputs live under `start/dataset/shoppers/` (`X_num_*.npy`, `X_cat_*.npy`, `y_*.npy`, `info.json`).
   - Numeric columns get mean-imputed, quantile-normalized (Gaussian output).  
   - Categorical columns map missing values to a dedicated `__nan__` token.

   Repeat with `start/prepare_aimers_dataset.py` for the AIMERS benchmark.

4. **Train diffusion model with tqdm progress**  
   ```bash
   python -m start.train --config start/configs/shoppers.toml
   ```
   - Per-epoch tqdm bar tracks mini-batch progress; logs still emit periodic loss/val metrics.
   - Config defaults: mean imputation, `new_category` categorical policy, `quantile` normalization, EMA, checkpoints.

5. **Generate synthetic samples from a checkpoint**  
   The config ships a `[generation]` block with default values, so the minimal command is:
   ```bash
   python -m start.generate_samples --config start/configs/shoppers.toml
   ```
   Override at the CLI when needed (`--checkpoint`, `--num-samples`, `--output`).  
   Generated CSVs reuse the preprocessing metadata to decode categorical and numeric columns properly.

Method (toy demo)
1. Generate discrete + numeric toy data (`make_toy_dataset.py`).
2. Train BlockUniform diffusion on the hybrid features (`python -m start.train --config start/configs/toy.toml`).
3. Run the sampler, then decode tokens back to original categories / numeric context (script below). The sampler returns `(categorical_tokens, numeric_values)` so both parts can be saved together.
  ```bash


### Future Implementation Ideas

- Unified SEDD pipeline that consolidates discrete score-entropy variants.
- Gaussian + SEDD hybrid using a DiT backbone.
- TabDiff deep dive to surface overlooked implementation details.
- Critical review of “A Comprehensive Survey of Synthetic Tabular Data Generation.”
- Design and benchmark imputation experiment settings.
- Explore block-absorbing mechanisms within the diffusion graph.
