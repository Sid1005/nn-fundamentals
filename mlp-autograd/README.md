# micrograd

A scalar-valued autograd engine and interactive browser visualizer, built on top of [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd).

## How it works

Training a neural network reduces to three repeated steps:

**1. Forward pass** — inputs flow through the network layer by layer. Each neuron computes a weighted sum of its inputs plus a bias, then squashes the result through a nonlinearity (`tanh`). Every operation (`+`, `*`, `tanh`) is tracked in a computation graph so gradients can flow back through it.

**2. Backpropagation** — starting from the loss, the chain rule is applied in reverse topological order across the computation graph. Each node accumulates `∂loss/∂self` from its downstream consumers. No external autodiff library — the graph is built dynamically during the forward pass and unwound here.

**3. Gradient descent** — each parameter nudges itself opposite to its gradient: `p.data -= lr * p.grad`. Repeat.

## Files

| File | What it is |
|---|---|
| `micrograd.py` | The engine: `Value` (scalar node with autograd), `Neuron`, `Layer`, `MLP` |
| `train.py` | Python trainer — 4-point binary classification, MSE loss, 100 epochs |
| `visualizer.html` | Self-contained browser app — JS port of the engine, interactive canvas, live decision boundary heatmap |

## Live demo

[**Open visualizer →**](YOUR_VERCEL_URL)

## Run locally

```bash
python train.py
```

Or open `visualizer.html` directly in any browser — no server needed.
