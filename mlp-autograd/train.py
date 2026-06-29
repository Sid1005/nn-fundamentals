import random
import matplotlib.pyplot as plt
import numpy as np
from micrograd import MLP, Value

def train_simple_binary():
    print("========================================")
    print("Task 1: Simple Binary Prediction (using +1 / -1)")
    print("========================================")
    
    # 4 training inputs (3-dimensional) and their labels (1.0 or -1.0)
    X = [
      [2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0]
    ]
    y = [1.0, -1.0, -1.0, 1.0] # Targets: +1 or -1
    
    # Initialize a simple MLP: 3 inputs, 2 hidden layers of 4 neurons, 1 output neuron
    # Now all layers use tanh, including the final output layer
    model = MLP(3, [4, 4, 1])
    
    # Training loop
    epochs = 100
    learning_rate = 0.05
    
    print(f"Training MLP for {epochs} epochs...")
    for k in range(epochs):
        # Forward pass
        ypred = [model(x) for x in X]
        
        # Mean Squared Error loss
        loss = sum((yout - ygt)**2 for ygt, yout in zip(y, ypred)) / len(y)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Gradient descent update
        for p in model.parameters():
            p.data -= learning_rate * p.grad
            
        if k % 10 == 0 or k == epochs - 1:
            print(f"  Epoch {k:2d} | Loss: {loss.data:.6f}")
            
    print("\nFinal Predictions:")
    for x, ygt, ypred_val in zip(X, y, ypred):
        print(f"  Input: {x} -> Target: {ygt:+.1f} | Predicted: {ypred_val.data:+.4f}")
    print("========================================\n")


if __name__ == "__main__":
    train_simple_binary()
