# PINN_Laplace_Multiscale_Electrostatics


A physics-informed neural network (PINN) that predicts electrostatic potential fields at **arbitrary zoom levels** over a fixed scene represented as a triangle mesh. Given any spatial location and zoom scale, the model returns the correct field value in a single forward pass — no re-solving required.

**Global L2 error: 1.94%** across a 100× zoom range, including scales never seen during training.

---

## What it does

The scene is a grounded conducting enclosure containing a segmented rectangular obstacle with a spatially varying cos² surface potential (corners at φ = 1, face midpoints at φ = 0), plus an asymmetric dipole charge distribution in the surrounding domain. The electrostatic potential satisfies the 2D Poisson equation:

```
∇²φ = −ρ(x, y)
```

The trained network predicts φ at any point and zoom level without re-solving the PDE.

---

## Key results

| Scale s | L2 error | Status |
|---------|----------|--------|
| 1.00 | 1.74% | Training scale |
| 0.50 | 2.03% | Training scale |
| 0.25 | 1.16% | Training scale |
| 0.10 | 0.70% | Training scale |
| 0.05 | 1.01% | **Unseen** |
| 0.02 | 2.24% | **Unseen** |
| 0.01 | 3.81% | **Unseen** |

The model generalises cleanly to zoom scales it never saw during training.

---

## Architecture

**Input (6D):** `[x, y, log(s), cx, cy, SDF]`
- `(x, y)` — world coordinates
- `log(s)` — log zoom scale; encodes the multiplicative structure of scale
- `(cx, cy)` — zoom window centre
- `SDF` — signed distance to nearest boundary (geometry-aware feature)

**Encoding:** dual-band random Fourier features on (x, y)
- Coarse band: σ = 1.0, 32 frequency pairs
- Fine band: σ = 3.0, 32 frequency pairs
- Total: 128 Fourier features + 4 context = 132-dimensional encoded input

**Network:** 6 hidden layers × 256 neurons, tanh activation, Xavier initialisation (~530K parameters)

**Output:** scalar φ̂(x, y, s, cx, cy)

---

## Training

Three-phase curriculum addressing the gradient imbalance between PDE and boundary losses:

1. **BC/data warmup** (steps 1–1500, lr = 1e-4): boundary conditions and supervised data only. BC loss reaches ~7e-5 before PDE is introduced.
2. **PDE ramp** (steps 1501–2000, ramp 0 → 0.083): PDE loss introduced gradually over 6000 steps to prevent BC collapse.
3. **L-BFGS** (8700+ evaluations): full-dataset second-order optimisation. BC reaches ~8e-5, data loss reaches ~2.7e-5.

Loss weights: `W_PDE = 1.0`, `W_BC = 600.0`, `W_DATA = 50.0`, `W_SCALE = 0.0`

The scale consistency loss (`W_SCALE`) was disabled — it conflicts with the spatially varying cos² BC, producing a gradient conflict that manifests as a swirling error artifact.

---

## Dataset

Three training pools:
- **Collocation** (~15,000 points): uniform interior + near obstacle boundary + near charge centres — no labels, enforces PDE residual
- **Boundary** (~4,000 points): sampled on outer wall (φ = 0) and obstacle surface (cos² profile) — enforces Dirichlet BCs
- **Supervised data** (~8,000 points): interior points with FD reference labels — enforces field accuracy

Training scales: `s ∈ {1.0, 0.5, 0.25, 0.1}`. Test scale `s = 0.05` held out entirely.

---

## Reference solution

Ground truth computed by sparse direct solve (`scipy.sparse.linalg.spsolve`) of the 5-point Laplacian on a 1024 × 1024 grid. The same matrix is reused with zero source term to compute the pure Laplace background, enabling a perturbation decomposition:

```
δφ = φ_Poisson − φ_Laplace
```

which isolates the charge-driven field from the boundary-driven background.

---

## Corner singularity

Near each re-entrant corner (interior angle 270°), the potential has the form:

```
φ̃(r, θ) ≈ Aₖ · r^(2/3) · sin(2θ/3)    as r → 0
```

The electric field diverges as |∇φ| ~ r^(−1/3). The cos² BC enhances this beyond the pure geometric singularity, because corners are simultaneously the geometric singularity locations and the local maxima of the boundary condition. The four corners carry distinct amplitudes Aₖ due to the asymmetric dipole charge distribution — visible in the corner singularity evaluation plots.

---

## Scene

Defined as a PSLG (Planar Straight Line Graph) and triangulated using the `triangle` package with flags `pq30a0.0005D`:
- 1,542 vertices, 2,900 triangles
- Minimum interior angle: 30°
- Maximum triangle area: 5 × 10⁻⁴

```
Domain: [0,1]² \ [0.35, 0.65]²
Charge centres: (0.20, 0.60) positive, (0.80, 0.40) negative (half amplitude)
```

---

## Dependencies

```
numpy
scipy
torch
shapely
triangle
matplotlib
sklearn
```

---

## Files

```
VINCI_LAPLACE.ipynb   — complete notebook: scene → reference → training → evaluation
README.md
```

---

## Physical context

This setup models a segmented conducting obstacle in a grounded enclosure — representative of patterned electrode geometries in MEMS devices, electrostatic actuators, and microelectronic components where surface potential varies spatially. The scale-conditioned PINN enables interactive field exploration at arbitrary resolution without re-solving.
