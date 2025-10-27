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
   The config ships a `[generation]` block with default values; just point it at the checkpoint you want:
   ```bash
   python -m start.generate_samples \
     --config start/configs/shoppers.toml \
     --checkpoint <path/to/epoch_xxxx.pt> \
     --output samples/shoppers_samples.csv
   ```
   Override `--num-samples` when you need a specific row count. The script decodes categorical tokens and inverts the numeric transformer so the CSV matches the original schema.

6. **Compare real vs synthetic tables (shape/trend)**  
   ```bash
   python -m start.evaluate_quality \
     --real data/shoppers/online_shoppers_intention.csv \
     --synthetic samples/shoppers_samples.csv \
     --info start/dataset/shoppers/info.json
   ```
   Reports the Tab-Diff style “shape” (distributional distance) and “trend” (mean shift) error rates for categorical, numeric, and target columns.

### Noise Schedules

- **Categorical** features follow a log-linear schedule.  
- **Numeric** features follow a power-mean (EDM) schedule and are trained with denoising score matching.  
  Training and sampling both use the same hybrid schedule so σ(t) stays consistent across pipelines.

### AIMERS Convenience CSV

`start/dataset/aimers/data.csv` bundles the imputed categorical tensors plus `임신 성공 확률` target and a `split` column. Use it for quick inspection without re-running the tensor preprocessing.

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
