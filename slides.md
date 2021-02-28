---
title: Automatic differentiation with JAX
author: Hans Dembinski
institute: TU Dortmund, Germany
---

# What is [JAX](https://jax.readthedocs.io)

![](https://jax.readthedocs.io/en/latest/_static/jax_logo_250px.png){width=30%}

* For vector-valued functions expressible in NumPy code ($\mathcal{R}^m \to \mathcal{R}^n$)
  * Compute derivatives with [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) (gradient, Jacobian, Hessian, ...)
  * JIT compile with XLA (Accelerated Linear Algebra) into optimised code for CPU,GPU,TPU
    * XLA also used by [TensorFlow](https://www.tensorflow.org/xla), [Julia](https://github.com/JuliaTPU/XLA.jl), [PyTorch](https://github.com/pytorch/xla), [Nx](https://github.com/elixir-nx/nx)
    * XLA compiler uses [LLVM](https://llvm.org) to generate low-level code for CPU,GPU
  * Vectorise code
* Successor of [autograd](https://github.com/hips/autograd)
* Mirrors [NumPy](https://numpy.org/) API
* Powered by [Google](https://github.com/google/jax)

# Why JAX?

:::::: columns
::: column

* Easy to learn if you already know NumPy

* Laser-focussed on one task

* Great low-level building block to build your own custom solutions

* [Supported and further developed by Google](https://cloud.google.com/blog/products/ai-machine-learning/google-breaks-ai-performance-records-in-mlperf-with-worlds-fastest-training-supercomputer)

* Replacement for Numba? Not yet... (will come back to that)

:::
::: column

![](ena-begcevic-cO_O1YKWZbY-unsplash.jpg){ width=80% }

Photo by [Ena Begcevic](https://unsplash.com/@enabegcevuc?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/building-block?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText)

:::
::::::

# Source material and further reading

* [Boris Settinger's gentle introduction to automatic differentiation](https://www.kaggle.com/borisettinger/gentle-introduction-to-automatic-differentiation)
  * Great Jupyter notebook with animations, references to prior work (go read this)
* [Baydin et al., *Automatic differentiation in machine learning: a survey*, Journal of Machine Learning Research, 18(153):1--43, 2018](https://arxiv.org/abs/1502.05767)
* JAX's [beginner- and poweruser-friendly documentation](https://jax.readthedocs.io/)

# Derivatives in statistics are everywhere

- Minimisation with Newton-Raphson method (e.g. Minuit's Migrad algorithm), in 1-dim:
  $$
  x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)}
  $$

- Error propagation for vector-valued function $\vec y = f(\vec x)$ that maps from $\mathcal{R}^m \to \mathcal{R}^n$
  $$
  C_y = J \, C_x \, J^T
  $$
  with $J_{ij} = \partial y_i / \partial x_j$

# Three ways to a compute derivative

- $\mathcal{R} \to \mathcal{R}$
  $$
  \frac{\partial y}{\partial x} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
  $$

- $\mathcal{R}^m \to \mathcal{R}^n$:
  $$
  \frac{\partial \vec y}{\partial x_j} = \lim_{h \to 0} \frac{f(\vec x + \vec{e}_j h) - f(\vec x)}{h}
  $$

- Three ways to compute derivatives on a computer
  - Numerical differentiation
  - Symbolic differentiation
  - Automatic differentiation

# Numerical differentiation

$$
\frac{\partial y}{\partial x} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h} \approx \frac{f(x + h) - f(x)}{h} + O(h^2)
$$
\small \hspace{0.3cm} (It is almost always better to use the symmetric version $\frac{\partial y}{\partial x} \approx \frac{f(x + h) - f(x - h)}{2h}$)

- If you choose $h$ small enough, then error is negligible
- Choose $h$ too small and you get numerical issues `1e16 - 1 == 1e16`
- **Pro**
  - Works on any $f(\vec x)$, even if $f(\vec x)$ calls into foreign (C++) code
  - *Pro-ish*: Requires $2 \times m$ function evaluations for $\mathcal{R}^m \to \mathcal{R}^n$
- **Contra**
  - Difficult to chose $h$ optimally/safely when nothing is known about $f(\vec x)$
  - Lots of research on doing this, but it remains complicated and fragile
  - Accuracy well below machine precision

# Symbolic differentiation

- Calculating derivatives is mechanical, we know all rules
  $(f(x) + g(x))' = f'(x) + g'(x), \; (x^a)' = a \, x^{a-1}, \; (\sin(x))' = \cos(x), \; \dots$
- Computer algebra systems (CAS) can apply these rules, e.g. [SymPy](https://www.sympy.org/en/index.html)
  ```py
  from sympy import diff
  from sympy.abc import x
  diff(3 * x ** 2 + 2 * x, x) # returns 6*x + 2
  ```
- **Pro**
  - Great if you need to know formula, CAS can simplify expression
  - Compute once (slow), evaluate many times (fast)
  - Exact within machine precision
- **Contra**
  - Only works if $f(\vec x)$ is analytical and symbolically representable
  - Does not work if function does computer things, `if`, `while`, ...

# Automatic differentiation

- Alternative approach to derivatives using [hyperreal numbers](https://en.wikipedia.org/wiki/Hyperreal_number)
  - Extension of real numbers to include infinite and **infinitesimal** quantities
  - That sounds cool (or scary) but is actually very simple
  - Physicists have intuitively used that for a long time
  - Can be easily implemented on computer
- **Pro**
  - Works (in principle) for any pure function one can write in code
  - Exact within machine precision
  - About same cost as (lowest order) numerical differentiation
- **Contra**
  - Fails if $f(\vec x)$ calls into foreign (C++) code

# Motivation of hyperreal numbers

- Start with a Taylor series
  $$
  f(x + \epsilon) = f(x) + f'(x) \epsilon + \frac 12 f''(x) \epsilon^2 + \dots
  $$
- Now define $\epsilon^2 = 0$, this truncates series to exact result
  $$
  f(x + \epsilon) = f(x) + f'(x) \epsilon
  $$
- Note that you can *read off* derivative as factor in front of $\epsilon$
- Turns out one can use this to algebraically compute derivatives

# Hyperreal numbers

- Think of hyperreal number $x + y \epsilon$ analog to complex number $x + y i$
  - $\epsilon^2 = 0$ instead of $i^2 = -1$
  - $\text{st}( \dots )$ extracts real number, analog to $\text{Re}(\dots)$ for complex number
- Laws for addition, multiplication, inverse, power can be easily derived

  $(x + a \epsilon) + (y + b \epsilon) = (x + y) + (a + b) \epsilon$

  $(x + a \epsilon) (y + b \epsilon) = xy + (xb + ya)\epsilon$

  $\dfrac{1}{x + a \epsilon} = \dfrac{x - a \epsilon}{(x + a \epsilon) (x - a \epsilon)} = \dfrac{x - a \epsilon}{x^2} = \dfrac 1x - \dfrac a {x^2} \epsilon$

  $(x + a \epsilon)^n = (x + a \epsilon) (x + a \epsilon) \dots =  x^n + n x^{n-1} a \epsilon$
- With these rules can go on to compute polynomials, transcendental functions expressible as infinite series of polynomials and so on

---

* We can program these dual numbers into computers
* Any code that works on real numbers also works on hyperreal numbers
* Derivatives are computed automatically as a side-product
* Also works if code uses computer things like `if`, `while`, ...
