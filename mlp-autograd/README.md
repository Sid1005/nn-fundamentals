# mlp-autograd

A scalar-valued autograd engine and browser visualizer, built on top of [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd).

## How it works

Training a neural network repeats three steps:

**1. Forward pass.** Inputs flow through the network layer by layer. Each neuron computes a weighted sum of its inputs plus a bias, then runs the result through a tanh nonlinearity. Every operation is tracked in a computation graph so gradients can flow back through it.

**2. Backpropagation.** Starting from the loss, the chain rule is applied in reverse order across the computation graph. Each node accumulates its share of the gradient from the nodes downstream of it. No external library. The graph is built during the forward pass and unwound here.

**3. Gradient descent.** Each parameter updates itself: `p.data -= lr * p.grad`. Repeat.

## Files

| File | What it is |
|---|---|
| `micrograd.py` | The engine: `Value` (scalar node with autograd), `Neuron`, `Layer`, `MLP` |
| `train.py` | Python trainer. 4-point binary classification, MSE loss, 100 epochs |
| `visualizer.html` | Self-contained browser app. JS port of the engine with an interactive canvas and live decision boundary |

## Live demo

[**Open visualizer**](https://mlp-autograd.vercel.app)

## Run locally

```bash
python train.py
```

Or open `visualizer.html` in any browser. No server needed.
