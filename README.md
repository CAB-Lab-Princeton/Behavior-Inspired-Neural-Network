# Behavior-Inspired Neural Networks for Relational Inference (BINN)

Reference implementation for the AISTATS 2025 paper
**"Behavior-Inspired Neural Networks for Relational Inference"**
Yulong Yang\*, Bowen Feng\*, Keqin Wang, Naomi Ehrich Leonard, Adji Bousso Dieng, Christine Allen-Blanchette
(Princeton University). [arXiv:2406.14746](https://arxiv.org/abs/2406.14746).

---

## About the paper

Multi-agent systems — pedestrians, oscillators, sports players, celestial
bodies — evolve under interactions that are rarely directly observable.
Classical **relational inference** methods (NRI, dNRI, EqMotion, …) treat each
inter-agent relationship as the outcome of a *categorical* variable: agent *i*
and agent *j* are in exactly one of *K* relationship classes. Real-world
relationships are usually not so clean — they intermingle and coexist.

BINN introduces an intermediate level of abstraction between *observed
physical behavior* and *latent relationship categories*: each agent holds
real-valued **preferences** over a set of latent categories, and those
preferences evolve in time according to a **nonlinear opinion dynamics (NOD)**
model. In one sentence: an agent's preference *z<sub>ij</sub>* for category
*j* obeys

  ż<sub>ij</sub> = −d<sub>ij</sub> z<sub>ij</sub> + S ( u<sub>i</sub> · (α<sub>ij</sub> z<sub>ij</sub> + Σ<sub>k</sub> a<sup>a</sup><sub>ik</sub> z<sub>kj</sub> + Σ<sub>l</sub> a<sup>o</sup><sub>jl</sub> z<sub>il</sub> + …) ) + b<sub>ij</sub>

where *d*, *α*, *u* are intrinsic parameters, *A*<sup>a</sup> is an
inter-agent communication graph, *A*<sup>o</sup> is an inter-category belief
graph, *b* is an environmental input, and *S* is a saturating nonlinearity
(tanh). The nonlinearity admits a **pitchfork bifurcation**, which lets
preferences flip quickly and flexibly in response to weak inputs — a property
NRI-style categorical models cannot reproduce.

### Architecture (at a glance)

- **Encoder** *E<sub>z</sub>* (MPNN): physical state → agent preferences *z*.
- **Environmental-input encoder** *E<sub>b</sub>* (MPNN): physical state →
  *b*.
- **NOD block** *f<sub>NOD</sub>*: propagates *z* forward one timestep using
  learned *d*, *α*, *u*, belief matrix *A*<sup>o</sup>, and a communication
  matrix *A*<sup>a</sup> computed from inter-agent distances.
- **Decoder** *D<sub>x</sub>* (MPNN): *z* → predicted next physical state.
- **BINN+** variant: when the communication prior is meaningful (e.g.
  pedestrians influence each other more when closer), *A*<sup>a</sup> is
  built as inverse squared distance *times* a learnable prefactor.

### What BINN can do

1. **Long-horizon trajectory prediction** — outperforms NRI and dNRI on
   Mass-Spring, Kuramoto, TrajNet++, and NBA Player datasets (Table 1 in the
   paper).
2. **Identify mutually exclusive categories** — if the learned belief matrix
   and preference pattern satisfy the mutual-exclusivity condition (see §3.1
   of the paper), the dynamics on those two categories decouple and the
   latent space can be reduced (demonstrated on the pendulum).
3. **Control** — the environmental input *b* can be steered to drive an agent
   toward a desired behavior via the NOD bifurcation (Figure 5 of the paper).
4. **Robustness to latent dimension** — training quality is roughly invariant
   to the number of latent categories *N*<sub>o</sub> (Table 2 of the paper),
   because mutually exclusive categories collapse naturally.

---

## Repository layout

- [run.py](run.py) — entry point (CLI, train / test).
- [model_utils.py](model_utils.py) — training loop, NOD math, encoder/decoder
  orchestration.
- [modules.py](modules.py) — `MPNN`, `NonlinearOpinionDynamics`, `DLCMLP`,
  `DataGenerator`.
- [data_utils.py](data_utils.py) — dataset loaders and simulators (double
  pendulum, mass-spring, Kuramoto, charged particles, TrajNet++, NBA).
- [plot_utils.py](plot_utils.py) — trajectory plots, loss curves, NOD
  parameter dumps.

---

## Requirements

- Python 3.10+
- PyTorch 2.0+ (2.6 recommended; `torch.compile` requires 2.0+)
- NumPy, SciPy, Matplotlib, tqdm
- CUDA-capable GPU recommended. `--compile_mode=reduce-overhead` uses CUDA
  graphs and runs best on Ampere-class (or newer) GPUs.

---

## How to launch

All runs go through `run.py`. The `--dataset` flag selects the dataset and
implicitly sets `nAgent`, `nDim`, `dt`, and the RNN unroll length
(`data_offset`) — see `load_dataset` in [data_utils.py](data_utils.py).

Supported datasets: `DP2` (double pendulum), `MS5` (mass-spring, 5 agents),
`KM5` (Kuramoto, 5 oscillators), `CH5` (charged particles, 5 agents),
`TR5` (TrajNet++ synthetic pedestrian, 5 agents), `NBA` (SportsVU player
trajectories).

### Flags

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--train` | int | 0 | 1 = train |
| `--test` | int | 0 | 1 = test |
| `--dataset` | str | `IP` | `DP2` `MS5` `KM5` `CH5` `TR5` `NBA` |
| `--arch` | str | `DLCMLP` | Use `MPNN` for all experimental settings |
| `--nOpinion` | int | 2 | Number of latent categories *N*<sub>o</sub> |
| `--dOrder` | int | 2 | 1 for 1st-order systems (Kuramoto), 2 for pos/vel systems |
| `--inverse` | int | 0 | 1 = BINN+ (inverse-distance communication prior) |
| `--hid_dim` | int | 256 | Hidden dimension of MLPs |
| `--epoch` | int | 1000 | Training epochs |
| `--batch_size` | int | 256 | Mini-batch size |
| `--learning_rate` | float | 1e-3 | AdamW learning rate |
| `--weight_decay` | float | 0.0 | AdamW decoupled weight decay |
| `--scheduler_step` | int | 200 | StepLR step size |
| `--scheduler_gamma` | float | 0.1 | StepLR decay factor |
| `--activation` | str | `tanh` | `tanh` / `relu` / `elu` / `gelu` / `leaky_relu` |
| `--train_set` | int | 50000 | Number of training trajectories to simulate |
| `--test_folder` | str | — | Checkpoint folder under `Model/` for `--test=1` |
| `--seed` | int | 72 | Random seed |
| `--compile` | int | 0 | 1 = enable `torch.compile` on the RNN unroll |
| `--compile_mode` | str | `default` | `default` / `reduce-overhead` / `max-autotune` |
| `--tf32` | int | 0 | 1 = enable TF32 matmul (Ampere+) |

### Usage examples

```bash
# NBA trajectory prediction (BINN+ variant, standard launch)
python3 run.py --train=1 --test=1 --arch=MPNN --dataset=NBA \
               --nOpinion=4 --dOrder=2 --inverse=1 --hid_dim=256 \
               --epoch=1000 --batch_size=256 --learning_rate=1e-4 \
               --scheduler_step=200 --scheduler_gamma=0.25 \
               --activation=elu --train_set=50000 --seed=72 \
               --num_worker=24

# Same launch, with torch.compile enabled for faster training
python3 run.py --train=1 --test=1 --arch=MPNN --dataset=NBA \
               --nOpinion=4 --dOrder=2 --inverse=1 --hid_dim=256 \
               --epoch=1000 --batch_size=256 --learning_rate=1e-4 \
               --scheduler_step=200 --scheduler_gamma=0.25 \
               --activation=elu --train_set=50000 --seed=72 \
               --num_worker=24 \
               --compile=1 --tf32=1

# Maximum-speed variant (CUDA graphs via compile_mode=reduce-overhead)
python3 run.py --train=1 --test=1 --arch=MPNN --dataset=NBA \
               --nOpinion=4 --dOrder=2 --inverse=1 --hid_dim=256 \
               --epoch=1000 --batch_size=256 --learning_rate=1e-4 \
               --scheduler_step=200 --scheduler_gamma=0.25 \
               --activation=elu --train_set=50000 --seed=72 \
               --num_worker=24 \
               --compile=1 --compile_mode=reduce-overhead --tf32=1

# Mass-spring (MS5), BINN (non-inverse), tanh activation
python3 run.py --train=1 --test=1 --arch=MPNN --dataset=MS5 \
               --nOpinion=4 --dOrder=2 --inverse=0 --hid_dim=128 \
               --epoch=1000 --batch_size=256 --learning_rate=1e-3 \
               --scheduler_step=200 --scheduler_gamma=0.25 \
               --activation=tanh --train_set=50000 --seed=72

# Test only (reload a previously saved model)
python3 run.py --train=0 --test=1 --test_folder="20250107-120330" \
               --arch=MPNN --dataset=NBA --nOpinion=4 --dOrder=2 \
               --inverse=1 --hid_dim=256 --activation=elu \
               --num_worker=24
```

---

## Citation

```bibtex
@inproceedings{yang2025binn,
  title     = {Behavior-Inspired Neural Networks for Relational Inference},
  author    = {Yang, Yulong and Feng, Bowen and Wang, Keqin and
               Leonard, Naomi Ehrich and Dieng, Adji Bousso and
               Allen-Blanchette, Christine},
  booktitle = {Proceedings of the 28th International Conference on Artificial
               Intelligence and Statistics (AISTATS)},
  year      = {2025}
}
```

---

## Acknowledgements

This work was partially supported by NSF grant MRSEC DMR-2011750 and the
Princeton Catalysis Initiative.
