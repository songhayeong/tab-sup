### Tab-SEDD (Sketch)

- Goal: mix SEDD’s discrete score-entropy scheme with Tab-DDPM’s tabular diffusion.  
- Key idea: build a block-uniform transition matrix for one-hot features, train an MLP/ResNet diffusion model on top.

### Usage Snapshot

```bash
python -m start.train --config configs/example.toml
```

Config needs dataset paths, graph type (`block_uniform`), noise schedule, and model/optim params.  
Everything else (preprocessing → graph → training loop with EMA) wires itself.
