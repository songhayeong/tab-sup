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
  python - <<'PY'
  import torch, start
  from pathlib import Path
  from start import utils, sampling, noise_lib, TrainConfig

  cfg = utils.load_config('start/configs/toy.toml')
  cfg = utils.from_dict(TrainConfig, cfg)
  dataset, graph = start.prepare_dataset_and_graph(cfg, cfg.dataset.path, cfg.transformations, torch.device('cpu'), cache=False)
  noise = noise_lib.get_noise(cfg)

  model = start.model.modules.MLPDiffusion(
      d_in=graph.dim + dataset.n_num_features,
      num_classes=0,
      is_y_cond=False,
      rtdl_params=cfg.model.rtdl_params,
      dim_t=cfg.model.dim_t,
  )
  model.load_state_dict(torch.load('start/checkpoints/toy/epoch_0004.pt', map_location='cpu')['model'])
  model.eval()

  numeric_ctx = torch.from_numpy(dataset.X_num['train'].mean(axis=0, keepdims=True)).float()
  tokens, numeric = sampling.sample_block_uniform(
      model,
      graph,
      noise,
      num_samples=256,
      steps=cfg.sampling.steps,
      predictor=cfg.sampling.predictor,
      denoise=cfg.sampling.noise_removal,
      device=torch.device('cpu'),
      numeric_context=numeric_ctx,
  )
  decoded = sampling.decode_block_uniform_tokens(graph, tokens, getattr(dataset, 'cat_transform', None))
  df = sampling.decoded_to_dataframe(decoded, dataset.cat_columns, numeric=numeric, numeric_columns=[f"num_{i}" for i in range(numeric.shape[1])])
  df.to_csv(Path('start/dataset/toy/toy_samples.csv'), index=False)
  PY
  ```
