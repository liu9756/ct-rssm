
# Reading Notes — Neural Ordinary Differential Equations

**Paper**: Neural Ordinary Differential Equations — Chen et al., NeurIPS 2018
**Date**: 2026-03-06
**Section**: Section 1 — Introduction

---

## Core Question

> What happens if, instead of stacking discrete layers, we directly model the **continuous dynamics** of a hidden state?

---

## Background: The Problem with Discrete Networks

Models like ResNets and RNNs build transformations by repeatedly applying:

$$h_{t+1} = h_t + f(h_t, \theta_t)$$

This is essentially an **Euler discretization** of a continuous transformation. It works, but comes with structural limitations:

- **Memory cost scales with depth** — backpropagation requires storing all intermediate activations: $O(L)$ memory where $L$ = number of layers
- **Fixed depth** — the number of computation steps is a static hyperparameter, not adaptive to input complexity
- **Normalizing flows are expensive** — the change-of-variables formula requires computing a Jacobian determinant at $O(D^3)$ cost

---

## The Neural ODE Idea

Instead of specifying a discrete sequence of layers, parameterize the **derivative** of the hidden state with a neural network:

$$\frac{dh(t)}{dt} = f(h(t), t, \theta)$$

The output is then the solution to this ODE initial value problem at time $T$:

$$h(T) = h(0) + \int_0^T f(h(t), t, \theta)\, dt = \texttt{ODESolve}(h(0), f, \theta)$$

> **Intuition**: ResNet gives you discrete snapshots of where a particle is at each timestep. Neural ODE gives you the particle's **velocity field** at every moment — the trajectory is then recovered by integration.

---

## Four Key Benefits (from Section 1)

### 1. Memory Efficiency
Gradients are computed via the **adjoint sensitivity method** — a second ODE solved backward in time — rather than backpropagating through each solver step. No intermediate activations need to be stored.

$$\text{Memory: } O(L) \;\longrightarrow\; O(1)$$

### 2. Adaptive Computation
Modern ODE solvers monitor numerical error and **adaptively adjust step size**:
- Simple inputs → fewer function evaluations (NFE) → less compute
- Complex inputs → more function evaluations → more compute

After training, the tolerance can be **lowered at inference time** for real-time or low-power applications — without retraining.

> **Intuition**: Like a GPS that adjusts routing dynamically based on traffic, rather than following a fixed path regardless of conditions.

### 3. Scalable and Invertible Normalizing Flows
Continuous transformations simplify the change-of-variables formula. The expensive log-determinant collapses to a **trace**:

$$\underbrace{\log\left|\det \frac{\partial f}{\partial z_0}\right|}_{O(D^3)\text{ — standard NF}} \;\longrightarrow\; \underbrace{-\operatorname{tr}\left(\frac{df}{dz(t)}\right)}_{O(D)\text{ — CNF}}$$

This is formalized in **Theorem 1 (Section 4)**: the Instantaneous Change of Variables.

### 4. Continuous-Time Series Models
Unlike RNNs, which require observations at fixed, evenly-spaced intervals, Neural ODEs define dynamics continuously. The latent state can be queried at **any arbitrary time point** — making them naturally suited for irregularly-sampled data (e.g., medical records, event logs).

> **Intuition**: An RNN is like a clock that only ticks at whole hours. A Neural ODE is like an analog watch — you can read it at any moment.

---

## Conceptual Summary

| | Discrete (ResNet/RNN) | Continuous (Neural ODE) |
|---|---|---|
| Hidden state update | $h_{t+1} = h_t + f(h_t, \theta_t)$ | $\frac{dh}{dt} = f(h(t), t, \theta)$ |
| Depth | Fixed integer $L$ | Continuous time $T$ |
| Memory (training) | $O(L)$ | $O(1)$ |
| Backpropagation | Store all activations | Solve adjoint ODE backward |
| Time-series | Fixed intervals only | Arbitrary timestamps |
| Normalizing flows | $O(D^3)$ Jacobian det | $O(D)$ trace |

---

## Key Terminology

| Term | Definition |
|---|---|
| **Transformation** | A function mapping data from one space to another; each layer in a network is one transformation |
| **Gradient** | $\partial L / \partial \theta$ — how much the loss changes with respect to a parameter; tells the optimizer which direction to update |
| **Backpropagation** | Chain-rule-based algorithm to compute gradients layer by layer in reverse |
| **Adjoint** $a(t)$ | $\partial L / \partial z(t)$ — gradient of the loss w.r.t. the hidden state at time $t$; the continuous analog of backprop |
| **NFE** | Number of Function Evaluations — how many times the ODE solver calls $f$; acts as an implicit measure of "depth" |

---

## Open Questions Going into Section 2

- How exactly does the adjoint sensitivity method work mathematically?
- How is the backward ODE constructed, and what is the augmented state?
- What does Algorithm 1 actually compute step by step?

---

*Next section: **Section 2 — Reverse-mode Automatic Differentiation of ODE Solutions***