### Tab-SEDD (Sketch)

- Goal: mix SEDD’s discrete score-entropy scheme with Tab-DDPM’s tabular diffusion.  
- Key idea: build a block-uniform transition matrix for one-hot features, train an MLP/ResNet diffusion model on top.

### Usage Snapshot

```bash
python -m start.train --config configs/example.toml
```

Config needs dataset paths, graph type (`block_uniform`), noise schedule, and model/optim params.  
Everything else (preprocessing → graph → training loop with EMA) wires itself.

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
