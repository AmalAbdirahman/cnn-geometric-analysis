# Adversarial Perturbations in CNNs: A Geometric Analysis

A from-scratch PyTorch investigation into *why* CNNs are vulnerable to adversarial examples — testing whether the vulnerability is a structural consequence of piecewise-linear ReLU decision boundaries, rather than a quirk or bug, using FGSM and PGD attacks on a CIFAR-10 classifier.

📄 [Full write-up (PDF)](./report.pdf) · 📓 [Notebook](./CNN_2.ipynb)

## Summary

I trained a VGG-style CNN on CIFAR-10 to a **78.5% clean test accuracy**, then attacked it with FGSM (single-step) and PGD (iterative) to see how quickly that accuracy collapses under small pixel perturbations — and, more importantly, to measure the *geometry* of why it collapses:

- **Accuracy under attack:** FGSM accuracy falls to 0.6% and PGD to 0.0% at ε = 0.03 — a 10,000-image full test-set sweep.
- **Directional alignment:** measured cosine similarity between the attack perturbation and the true loss gradient (0.75 for FGSM, 0.57 for PGD), testing this against the ~0.797 value predicted under an i.i.d. Gaussian gradient assumption.
- **Linearity breakdown:** quantified how far the loss surface departs from the first-order Taylor approximation FGSM relies on (Taylor error 0.96 at ε = 0.05), showing why iterative attacks (PGD) are needed once ε grows.
- **High-dimensional anisotropy:** compared loss growth along the adversarial direction vs. a random direction of equal norm, in R³⁰⁷², to show random directions are near-orthogonal to the gradient while the adversarial direction pushes loss up sharply.

## Architecture

Three Conv → BatchNorm → ReLU → MaxPool blocks, uniform 3×3 filters, followed by a linear classifier outputting raw logits (no softmax, to avoid gradient masking):

```
R^(32×32×3) → R^(16×16×32) → R^(8×8×64) → R^(4×4×128) → 2048 → 10
```

Sized deliberately to fit a Colab T4 budget rather than to chase state-of-the-art accuracy — see [Limitations](#limitations).

## Results

| | Clean | FGSM (ε=0.03) | PGD, 20 steps (ε=0.03) |
|---|---|---|---|
| Test accuracy | 78.5% | 0.6% | 0.0% |

See the report for the full epsilon sweep (ε ∈ [0.01, 0.10]), gradient-norm trajectory over PGD iterations, the 1D loss-slice comparison, and visual triplets (original / perturbation / adversarial) across five classes.

## Repo structure

```
├── CNN_2.ipynb        # full training + attack + analysis pipeline
├── report.pdf          # write-up: methodology, results, limitations
└── README.md
```

## Running it

Built for Google Colab (T4 GPU). To reproduce:

1. Open `CNN_2.ipynb` in Colab, set runtime to GPU.
2. Run all cells — CIFAR-10 downloads automatically via `torchvision.datasets`.
3. Training takes ~10–15 min on a T4; the full epsilon sweep (10,000 images × 2 attacks × 5 epsilons) is the slower step.

**Note on reproducibility:** a fixed seed (42) is set, but `torch.use_deterministic_algorithms` is not enabled, so exact metric values may drift slightly (~0.1%) between runs on different hardware.

## Limitations

- Geometric statistics (cosine similarity, Taylor error) are computed on a single correctly-classified sample as an illustrative case study, not averaged across the full test set.
- The architecture is intentionally small (78.5% clean accuracy); results may not transfer directly to deeper networks with different loss-surface curvature.
- All attacks are white-box (full gradient access) — this says nothing about black-box or transfer-attack robustness.
- No adversarial training was actually implemented; the min-max defence formulation is discussed theoretically, not tested empirically.

## References

Goodfellow et al. (2015), Madry et al. (2018), Carlini & Wagner (2017), Santurkar et al. (2018), Montufar et al. (2014), Simonyan & Zisserman (2014), Kingma & Ba (2014), Krizhevsky (2009) — full citations in the report.
