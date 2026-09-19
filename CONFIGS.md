# Comparison-method configuration reference

The three configurations below are the compact search spaces reported in Supplementary Table 4. The selected entries match the boldface configurations in that table.

## Short- and medium-term methods

### Wind

#### FFNN

- Configuration 1: `hidden_dim=512`, `num_layers=12`, `dropout=0.3`.
- Configuration 2: `hidden_dim=256`, `num_layers=8`, `dropout=0.3`.
- Configuration 3 — selected: `hidden_dim=128`, `num_layers=4`, `dropout=0.3`.

#### GEFCom12

- Configuration 1: `n_estimators=5`, `num_leaves=16`.
- Configuration 2: `n_estimators=10`, `num_leaves=32`.
- Configuration 3 — selected: `n_estimators=20`, `num_leaves=64`.

#### HEFTCom24

- Configuration 1: `n_estimators=100`, `num_leaves=200`.
- Configuration 2 — selected: `n_estimators=200`, `num_leaves=400`.
- Configuration 3: `n_estimators=400`, `num_leaves=700`.

#### CNN-RBFNN

- Configuration 1: `hidden_dim=64`, `num_layers=1`.
- Configuration 2: `hidden_dim=128`, `num_layers=2`.
- Configuration 3 — selected: `hidden_dim=256`, `num_layers=4`.

#### HPE

- Configuration 1: `n_estimators=50`, `max_features=32`.
- Configuration 2: `n_estimators=100`, `max_features=64`.
- Configuration 3 — selected: `n_estimators=150`, `max_features=128`.

### Solar

#### ANN

- Configuration 1: `hidden_dim=512`, `num_layers=3`, `dropout=0.3`.
- Configuration 2 — selected: `hidden_dim=384`, `num_layers=2`, `dropout=0.2`.
- Configuration 3: `hidden_dim=256`, `num_layers=2`, `dropout=0.2`.

#### CrossViViT

- Configuration 1: `d_model=384`, `depth=16`, `cross_depth=4`, `decoder_depth=4`, `heads=12`.
- Configuration 2: `d_model=256`, `depth=12`, `cross_depth=3`, `decoder_depth=3`, `heads=8`.
- Configuration 3 — selected: `d_model=128`, `depth=8`, `cross_depth=2`, `decoder_depth=2`, `heads=6`.

#### PVTransNet

- Configuration 1: `d_model=128`, `depth=1`, `heads=2`, `head_dim=64`.
- Configuration 2: `d_model=256`, `depth=4`, `heads=4`, `head_dim=128`.
- Configuration 3 — selected: `d_model=512`, `depth=12`, `heads=8`, `head_dim=128`.

#### CNN-LSTM

- Configuration 1: `d_model=128`, `depth=2`, `dropout=0.1`.
- Configuration 2 — selected: `d_model=256`, `depth=3`, `dropout=0.1`.
- Configuration 3: `d_model=128`, `depth=4`, `dropout=0.2`.

#### FusionSF

- Configuration 1: `d_model=128`, `heads=2`, `depth=3`, `look_back_steps=48`, `dropout=0.1`.
- Configuration 2 — selected: `d_model=256`, `heads=4`, `depth=6`, `look_back_steps=96`, `dropout=0.1`.
- Configuration 3: `d_model=512`, `heads=8`, `depth=12`, `look_back_steps=96`, `dropout=0.1`.

## Ultra-short-term methods

### Wind

#### TFT

- Configuration 1 — selected: `d_model=112`, `num_layers=9`, `heads=7`, `dropout=0.1`.
- Configuration 2: `d_model=56`, `num_layers=7`, `heads=7`, `dropout=0.1`.
- Configuration 3: `d_model=49`, `num_layers=5`, `heads=7`, `dropout=0.1`.

#### HBOLA

- Configuration 1: `hidden_size=64`, `num_layers=5`, `dropout=0.1`.
- Configuration 2 — selected: `hidden_size=96`, `num_layers=10`, `dropout=0.1`.
- Configuration 3: `hidden_size=128`, `num_layers=20`, `dropout=0.15`.

#### LSSVM-RBFNN

- Configuration 1: `embed_dim=64`, `hidden_dim=128`, `num_layers=2`, `num_prototypes=64`.
- Configuration 2: `embed_dim=128`, `hidden_dim=256`, `num_layers=2`, `num_prototypes=128`.
- Configuration 3 — selected: `embed_dim=128`, `hidden_dim=384`, `num_layers=3`, `num_prototypes=256`.

#### DMOM

- Configuration 1: `hidden_dim=32`, `alpha=0.3`, `dropout=0.05`.
- Configuration 2: `hidden_dim=48`, `alpha=0.5`, `dropout=0.1`.
- Configuration 3 — selected: `hidden_dim=64`, `alpha=0.7`, `dropout=0.15`.

#### SC-VAR

- Configuration 1: `hidden_dim=192`, `num_layers=2`, `dropout=0.15`.
- Configuration 2: `hidden_dim=320`, `num_layers=3`, `dropout=0.2`.
- Configuration 3 — selected: `hidden_dim=512`, `num_layers=4`, `dropout=0.25`.

### Solar

#### WPD-LSTM

- Configuration 1: `hidden_dim=128`, `dropout=0.1`.
- Configuration 2 — selected: `hidden_dim=256`, `dropout=0.1`.
- Configuration 3: `hidden_dim=256`, `dropout=0.2`.

#### ATCN

- Configuration 1 — selected: `hidden_dim=48`, `kernel_size=5`, `dropout=0.1`.
- Configuration 2: `hidden_dim=64`, `kernel_size=5`, `dropout=0.1`.
- Configuration 3: `hidden_dim=80`, `kernel_size=7`, `dropout=0.2`.

#### LSTM-GCN-MLP

- Configuration 1: `hidden_size=64`, `num_layers=2`, `dropout=0.1`.
- Configuration 2: `hidden_size=96`, `num_layers=3`, `dropout=0.1`.
- Configuration 3 — selected: `hidden_size=128`, `num_layers=3`, `dropout=0.15`.

#### NARX-GA

- Configuration 1: `hidden_dim=32`, `fc_hidden_dim=128`.
- Configuration 2: `hidden_dim=64`, `fc_hidden_dim=256`.
- Configuration 3 — selected: `hidden_dim=96`, `fc_hidden_dim=256`.

#### RFs-ALO

- Configuration 1: `n_estimators=500`, `num_leaves=10`.
- Configuration 2 — selected: `n_estimators=1000`, `num_leaves=5`.
- Configuration 3: `n_estimators=300`, `num_leaves=20`.
