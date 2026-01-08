# Optimizer Documentation

This document provides a detailed overview of the Optimizer subsystem within the `cgadimpl` codebase. It outlines the class hierarchy, functionality, and specific details of the implemented optimization algorithms.

## Overview

The `optim` module provides a class-based framework to update model parameters to minimize a loss function. Unlike the initial functional implementation, the current system uses an `Optimizer` base class, allowing for stateful optimization (e.g., tracking moments in Adam).

## API Overview

The following classes are defined in `include/ad/optimizer/optim.hpp`:

### Base Class: `Optimizer`
The base class for all optimizers. It manages a list of parameters and handles mixed-precision updates via master weights.

```cpp
class Optimizer {
public:
    Optimizer(const std::vector<Value>& params);
    virtual void step() = 0;
    void zero_grad();
};
```

### `SGDOptimizer`
Implements standard Stochastic Gradient Descent.

```cpp
class SGDOptimizer : public Optimizer {
public:
    SGDOptimizer(const std::vector<Value>& params, float learning_rate = 0.01);
    void step() override;
};
```

### `Adam`
Implements the Adam (Adaptive Moment Estimation) optimization algorithm.

```cpp
class Adam : public Optimizer {
public:
    Adam(const std::vector<Value>& params, 
         float alpha = 0.001, 
         float beta1 = 0.9, 
         float beta2 = 0.999, 
         float epsilon = 1e-8);
    void step() override;
};
```

## Adam Optimizer Details

Adam is an adaptive learning rate optimization algorithm that maintains two moving averages for each parameter:

### 1. First Moment (`m`) - Momentum
The first moment `m` is the exponentially decaying average of past gradients. It helps the optimizer continue moving in the same direction, effectively adding "inertia" to the updates.
- **Formula**: $m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$
- **Role**: Smooths the gradient updates and helps navigate through noisy gradients or local minima.

### 2. Second Moment (`v`) - Adaptive Scaling
The second moment `v` is the exponentially decaying average of past squared gradients.
- **Formula**: $v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$
- **Role**: Provides a measure of the "uncentered variance" of the gradients. It is used to scale the learning rate for each parameter individually. Parameters with large, frequent gradients get smaller updates, while those with small, sparse gradients get larger updates.

### 3. Bias Correction
Since `m` and `v` are initialized to zero, they are biased towards zero, especially during the initial time steps. Adam applies bias correction to counteract this:
- $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$
- $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$

## L2 Regularization vs. Weight Decay

In the context of Adam, there is a subtle but important distinction between L2 Regularization and Weight Decay:

### L2 Regularization (Penalty to Loss)
Standard L2 regularization adds a penalty term directly to the loss function:
$Loss_{total} = Loss_{data} + \frac{\lambda}{2} \sum w^2$
This results in a gradient term $\lambda w$ being added to the original gradient. In Adam, this "regularized gradient" then goes through the `m` and `v` moving average calculations.

### Weight Decay (Direct Penalty)
Weight decay (as seen in **AdamW**) applies the penalty directly to the weight update, *after* the adaptive scaling:
$w_{t+1} = w_t - \alpha \cdot (\text{Adam Update}) - \lambda w_t$

> [!NOTE]
> The current `cgadimpl` implementation of Adam **does not** include built-in weight decay. If you wish to use L2 regularization, you should add the penalty term directly to your loss function before calling `backward()`. This will add the penalty term directly to the gradient.

## Analysis: Pros and Cons

### Pros
*   **Stateful Optimization**: The class-based API allows for tracking moments (Adam) and other states across iterations.
*   **Mixed Precision Support**: Automatically handles master weights in Float32 for parameters stored in lower precision (e.g., Float16/BFloat16).
*   **Efficient Parameter Access**: Iterates over a pre-defined list of parameters (`O(P)`) rather than traversing the entire graph (`O(N)`).

### Cons
*   **Manual Parameter Management**: Requires the user to manually pass the list of parameters to the optimizer constructor.
*   **No Built-in Regularization**: L2 regularization or Weight Decay must be implemented manually in the loss calculation.
