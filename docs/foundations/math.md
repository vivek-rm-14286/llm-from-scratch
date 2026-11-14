# Math You Actually Need

Deep learning uses a surprisingly small slice of mathematics. You need to multiply matrices and know what the shapes mean, take derivatives and chain them together, and reason about probability distributions well enough to understand softmax, cross-entropy, and KL divergence. That is most of it. This page covers exactly that slice, each idea with a short numpy example you can run.

!!! abstract "What you'll learn"
    - Vectors, matrices, dot products, and matrix multiplication, with a focus on shapes
    - Derivatives, partial derivatives, gradients, and the chain rule
    - Why the gradient points uphill and how gradient descent uses that
    - Probability distributions, expectation, and log-probabilities
    - Softmax (and its numerically stable form), cross-entropy, and why they pair so well
    - Entropy, KL divergence, and perplexity, the information theory that appears in LLM training and RL

All examples need only numpy:

```python
import numpy as np
np.set_printoptions(precision=4, suppress=True)
```

## Linear algebra

### Vectors and the dot product

A **vector** is a list of numbers, such as $x = [x_1, x_2, \ldots, x_n]$. In this site vectors usually represent things: a token's embedding, a hidden state, a game observation.

The **dot product** of two vectors of the same length multiplies them elementwise and sums:

$$
a \cdot b = \sum_{i=1}^{n} a_i b_i
$$

Intuition: the dot product is large and positive when the vectors point in similar directions, near zero when they are unrelated (orthogonal), and negative when they point in opposite directions. Attention scores in a transformer are dot products between a query and a key: "how relevant is this token to that one?"

$$
a \cdot b = \|a\| \, \|b\| \cos\theta
$$

Dividing out the lengths gives **cosine similarity**, the standard way to compare embeddings in [RAG](../rag/fundamentals.md).

```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([2.0, 0.0, 1.0])
print(a @ b)                                         # 5.0
cos = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(cos)                                           # ~0.598
```

### Matrices and matrix multiplication

A **matrix** is a 2D grid of numbers with shape `(rows, cols)`. Multiplying matrix $A$ of shape $(m, k)$ by $B$ of shape $(k, n)$ gives $C$ of shape $(m, n)$, where each entry is a dot product of a row of $A$ with a column of $B$:

$$
C_{ij} = \sum_{p=1}^{k} A_{ip} B_{pj}
$$

The inner dimensions must match: $(m, \mathbf{k}) \times (\mathbf{k}, n) \to (m, n)$. Keeping track of shapes is most of the battle in deep learning code.

A **linear layer** in a neural network is just a matrix multiply plus a bias. With a batch of $B$ inputs, each of dimension $d_\text{in}$, stacked as rows of $X$:

$$
Y = X W + b, \qquad X: (B, d_\text{in}),\; W: (d_\text{in}, d_\text{out}),\; b: (d_\text{out}),\; Y: (B, d_\text{out})
$$

```python
rng = np.random.default_rng(0)
X = rng.normal(size=(4, 3))     # batch of 4 inputs, 3 features each
W = rng.normal(size=(3, 5))     # maps 3 -> 5 features
b = np.zeros(5)
Y = X @ W + b                   # b broadcasts across the batch
print(Y.shape)                  # (4, 5)
```

!!! note "Row vs column convention"
    Math texts often write $y = Wx$ with column vectors. Code (numpy, PyTorch) usually stores a batch as rows, so you see `X @ W`. PyTorch's `nn.Linear` stores its weight as `(out, in)` and computes `X @ W.T + b`. Same math, transposed layout.

### Transpose and a few useful facts

The **transpose** $A^\top$ swaps rows and columns: $(A^\top)_{ij} = A_{ji}$. Two facts you will use:

- $(AB)^\top = B^\top A^\top$
- Matrix multiplication is associative, $(AB)C = A(BC)$, but **not** commutative, $AB \neq BA$ in general.

The cost of multiplying $(m, k)$ by $(k, n)$ is about $2mkn$ floating-point operations (one multiply and one add per term). This is where the "2 FLOPs per parameter per token" rule on [Scaling Laws & Compute](../llm/scaling-laws.md) comes from.

### Norms

The **L2 norm** (length) of a vector is $\|x\|_2 = \sqrt{\sum_i x_i^2}$. You will see it in weight decay (penalizing large weights), gradient clipping (capping $\|\nabla\|$), and RMSNorm (dividing by $\sqrt{\text{mean}(x^2)}$).

## Calculus

### Derivatives

The **derivative** $f'(x)$ tells you how fast $f$ changes as $x$ changes, at a given point. If $f'(x) = 3$, a tiny increase $\epsilon$ in $x$ increases $f$ by about $3\epsilon$:

$$
f(x + \epsilon) \approx f(x) + f'(x)\,\epsilon
$$

The derivatives you need:

| $f(x)$ | $f'(x)$ |
|--------|---------|
| $x^n$ | $n x^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln x$ | $1/x$ |
| $\sigma(x) = 1/(1+e^{-x})$ | $\sigma(x)(1-\sigma(x))$ |
| $\tanh x$ | $1 - \tanh^2 x$ |
| $\text{ReLU}(x) = \max(0, x)$ | $1$ if $x > 0$ else $0$ |

You can always check a derivative **numerically** with a finite difference. This is how you will test your backprop code:

```python
def numerical_derivative(f, x, eps=1e-5):
    return (f(x + eps) - f(x - eps)) / (2 * eps)

sigmoid = lambda x: 1 / (1 + np.exp(-x))
x = 0.7
analytic = sigmoid(x) * (1 - sigmoid(x))
print(analytic, numerical_derivative(sigmoid, x))   # both ~0.2217
```

### Partial derivatives and gradients

A neural network's loss depends on millions of parameters. A **partial derivative** $\partial f / \partial x_i$ is the derivative with respect to one input while holding the others fixed. The **gradient** collects all of them into a vector:

$$
\nabla f(x) = \left[ \frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n} \right]
$$

The gradient has the same shape as $x$, and it points in the direction in which $f$ increases fastest. So to *decrease* a loss, step the opposite way. That is **gradient descent**:

$$
\theta \leftarrow \theta - \eta \, \nabla_\theta L(\theta)
$$

where $\eta$ is the **learning rate**.

```python
# Minimize f(x, y) = (x - 3)^2 + 10 * (y + 1)^2
def grad(p):
    x, y = p
    return np.array([2 * (x - 3), 20 * (y + 1)])

p = np.array([0.0, 0.0])
for step in range(100):
    p = p - 0.04 * grad(p)
print(p)  # close to [3, -1]
```

Try `lr = 0.11` instead of `0.04`. The `y` coordinate oscillates and blows up because its curvature (the 10 in front) is high. A learning rate that is too large for the steepest direction diverges. This is the same reason LLM training uses warmup and gradient clipping.

### The chain rule

The **chain rule** is the single most important idea for training neural networks. If $y = f(u)$ and $u = g(x)$, then:

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

Intuition: if $u$ changes 2x as fast as $x$, and $y$ changes 3x as fast as $u$, then $y$ changes 6x as fast as $x$. A neural network is a long chain of simple functions, and **backpropagation** is just the chain rule applied from the output backward, reusing intermediate results.

When a variable feeds into several paths, you **sum** the contributions from each path:

$$
\frac{\partial L}{\partial x} = \sum_{k} \frac{\partial L}{\partial u_k} \frac{\partial u_k}{\partial x}
$$

This "sum over paths" rule is why gradients *accumulate* in PyTorch (`+=`), and it is the reason you call `optimizer.zero_grad()` each step.

```python
# L = (w*x + b - y)^2 ; compute dL/dw by the chain rule
w, x, b, y = 2.0, 3.0, 1.0, 5.0
u = w * x + b          # 7
e = u - y              # 2
L = e ** 2             # 4
dL_de = 2 * e          # 4
de_du = 1.0
du_dw = x              # 3
dL_dw = dL_de * de_du * du_dw
print(dL_dw)           # 12.0
print(numerical_derivative(lambda w_: (w_ * x + b - y) ** 2, w))  # ~12.0
```

### Gradients of matrix expressions

For a linear layer $Y = XW$, if you know the gradient of the loss with respect to the output, $G = \partial L / \partial Y$ (same shape as $Y$), then:

$$
\frac{\partial L}{\partial W} = X^\top G, \qquad \frac{\partial L}{\partial X} = G W^\top
$$

A trick to remember these: the result must have the same shape as the thing you differentiate with respect to. $W$ is $(d_\text{in}, d_\text{out})$, $X^\top$ is $(d_\text{in}, B)$, $G$ is $(B, d_\text{out})$, so $X^\top G$ is the only arrangement that works. You will use these two formulas in the [from-scratch MLP](neural-networks.md).

## Probability

### Distributions

A **probability distribution** over a discrete set (such as a vocabulary of 50,000 tokens) is a list of non-negative numbers that sum to 1. A language model's output at each position is exactly this: $p(\text{next token} \mid \text{previous tokens})$.

The **expectation** of a function under a distribution is its probability-weighted average:

$$
\mathbb{E}_{x \sim p}[f(x)] = \sum_x p(x) f(x)
$$

In RL, the objective is an expected return; in training, the loss is an expected value over the data, which you approximate by averaging over a mini-batch.

### Joint, conditional, and the chain rule of probability

The probability of a whole sequence factorizes into a product of conditionals:

$$
p(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \ldots, x_{t-1})
$$

This identity is the whole basis of **autoregressive** language modeling: predict one token at a time, conditioned on everything before it.

### Why we work with logs

Multiplying thousands of probabilities, each less than 1, underflows to zero in floating point. Taking logs turns products into sums, which are numerically safe:

$$
\log p(x_1, \ldots, x_T) = \sum_{t=1}^{T} \log p(x_t \mid x_{<t})
$$

```python
probs = np.full(1000, 0.01)
print(np.prod(probs))           # 0.0  (underflow)
print(np.sum(np.log(probs)))    # -4605.17 (fine)
```

## Softmax

A network produces raw scores called **logits**, any real numbers. **Softmax** turns a vector of logits $z$ into a probability distribution:

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Exponentiating makes everything positive; dividing by the sum makes it sum to 1. Larger logits get disproportionately more probability.

### The numerically stable version

$e^{1000}$ overflows to infinity. Softmax is unchanged if you subtract any constant from all logits (it cancels in the ratio), so subtract the maximum first:

```python
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)   # stability
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

z = np.array([1000.0, 1001.0, 1002.0])
print(np.exp(z) / np.exp(z).sum())   # [nan nan nan] with overflow warning
print(softmax(z))                    # [0.0900 0.2447 0.6652]
```

### Temperature

Dividing logits by a **temperature** $T$ before softmax controls how peaked the distribution is. $T < 1$ sharpens it (more deterministic), $T > 1$ flattens it (more random). This is the temperature knob in [sampling](../inference/inference.md).

```python
z = np.array([2.0, 1.0, 0.1])
for T in [0.5, 1.0, 2.0]:
    print(T, softmax(z / T))
```

### Log-softmax

Often you need $\log \text{softmax}(z)$ directly. Computing it as `log(softmax(z))` loses precision when a probability is tiny. The stable form is:

$$
\log \text{softmax}(z)_i = z_i - \log \sum_j e^{z_j}
$$

where the second term is the **log-sum-exp**, computed stably by factoring out the max:

```python
def log_softmax(z, axis=-1):
    m = z.max(axis=axis, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=axis, keepdims=True))
```

## Cross-entropy

**Cross-entropy** measures how well a predicted distribution $q$ matches a true distribution $p$:

$$
H(p, q) = -\sum_x p(x) \log q(x)
$$

In classification and language modeling, the "true" distribution is one-hot: all probability on the correct class $y$. The sum collapses to a single term, the **negative log-likelihood** of the correct answer:

$$
L = -\log q(y) = -\log \text{softmax}(z)_y
$$

Intuition: if the model gives the right token probability 1, the loss is 0. If it gives it 0.5, the loss is $\ln 2 \approx 0.69$. If it gives it 0.001, the loss is about 6.9. Confident wrong answers are punished heavily.

```python
def cross_entropy(logits, targets):
    # logits: (N, C), targets: (N,) integer class indices
    lp = log_softmax(logits, axis=-1)
    return -lp[np.arange(len(targets)), targets].mean()

logits = np.array([[2.0, 0.5, -1.0],
                   [0.1, 0.2, 3.0]])
targets = np.array([0, 2])
print(cross_entropy(logits, targets))   # ~0.175
```

!!! tip "A baseline worth memorizing"
    A model that predicts uniformly over $V$ classes has cross-entropy $\ln V$. For a 50,000-token vocabulary that is about 10.8. A freshly initialized language model should start near this value. If your initial loss is much higher, something is wrong with initialization or the output layer.

### The softmax + cross-entropy gradient

This combination has a beautifully simple gradient with respect to the logits:

$$
\frac{\partial L}{\partial z_i} = \text{softmax}(z)_i - \mathbb{1}[i = y]
$$

In words: predicted probabilities minus the one-hot target. If the model puts 0.7 on the right answer, the gradient on that logit is $-0.3$ (push it up); every wrong class gets its own probability as gradient (push it down). This is why frameworks fuse softmax and cross-entropy into one operation, and why PyTorch's `F.cross_entropy` takes raw logits, not probabilities.

```python
z = np.array([2.0, 0.5, -1.0]); y = 0
p = softmax(z)
grad_analytic = p.copy(); grad_analytic[y] -= 1
f = lambda z_: -log_softmax(z_)[y]
grad_numeric = np.array([
    (f(z + eps_vec) - f(z - eps_vec)) / 2e-5
    for eps_vec in np.eye(3) * 1e-5
])
print(grad_analytic, grad_numeric)   # match
```

## Information theory

### Entropy

The **entropy** of a distribution measures its uncertainty:

$$
H(p) = -\sum_x p(x) \log p(x)
$$

A one-hot distribution has entropy 0 (no uncertainty). A uniform distribution over $V$ outcomes has the maximum, $\log V$. With natural log the unit is *nats*; with $\log_2$ it is *bits*. In RL, adding an entropy bonus to the objective keeps a policy from collapsing to deterministic actions too early ([PPO](../decision/policy-methods.md)).

### KL divergence

The **Kullback-Leibler (KL) divergence** measures how different distribution $q$ is from $p$:

$$
D_\text{KL}(p \,\|\, q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)
$$

Key properties:

- $D_\text{KL} \geq 0$, and it is 0 only when $p = q$.
- It is **not symmetric**: $D_\text{KL}(p\|q) \neq D_\text{KL}(q\|p)$ in general.
- Because $H(p)$ does not depend on the model $q$, minimizing cross-entropy $H(p, q)$ is the same as minimizing $D_\text{KL}(p\|q)$. Training a language model with cross-entropy pushes its distribution toward the data distribution.

KL divergence shows up repeatedly later: as a penalty that keeps an RLHF-tuned model close to its starting point ([Preference Optimization](../post-training/preference.md)), in PPO's trust region, and in knowledge distillation.

```python
def kl(p, q):
    return np.sum(p * (np.log(p) - np.log(q)))

p = np.array([0.7, 0.2, 0.1])
q = np.array([0.4, 0.4, 0.2])
print(kl(p, q), kl(q, p))   # ~0.184 vs ~0.192: not symmetric
```

### Perplexity

**Perplexity** is the exponential of the average cross-entropy per token:

$$
\text{PPL} = \exp\left( -\frac{1}{T}\sum_{t=1}^{T} \log q(x_t \mid x_{<t}) \right)
$$

Intuition: a perplexity of 20 means the model is, on average, as uncertain as if it were choosing uniformly among 20 tokens. Lower is better. Perplexity is only comparable between models that use the same tokenizer, because a different tokenizer changes what "one token" means. See [Evaluation](../pretraining/evaluation.md).

```python
token_logprobs = np.log(np.array([0.2, 0.05, 0.5, 0.1]))
print(np.exp(-token_logprobs.mean()))   # ~6.7
```

## Putting it together

Here is one training step of a linear softmax classifier in pure numpy, using only ideas from this page: matrix multiply, softmax, cross-entropy, the softmax-cross-entropy gradient, the matrix gradient rule, and gradient descent.

```python title="softmax_regression.py"
rng = np.random.default_rng(0)
N, D, C = 200, 2, 3
centers = np.array([[2, 0], [-1, 2], [-1, -2]])
y = rng.integers(0, C, size=N)
X = centers[y] + rng.normal(scale=0.7, size=(N, D))

W = np.zeros((D, C)); b = np.zeros(C)
for step in range(200):
    logits = X @ W + b                      # (N, C)
    P = softmax(logits)                     # (N, C)
    loss = -np.log(P[np.arange(N), y]).mean()
    G = P.copy(); G[np.arange(N), y] -= 1   # dL/dlogits
    G /= N                                  # because loss is a mean
    dW = X.T @ G                            # (D, C)
    db = G.sum(axis=0)                      # (C,)
    W -= 0.5 * dW; b -= 0.5 * db
    if step % 50 == 0:
        print(step, round(loss, 4))

acc = (np.argmax(X @ W + b, axis=1) == y).mean()
print("accuracy", acc)
```

The loss starts at $\ln 3 \approx 1.0986$ (uniform predictions, since $W = 0$) and falls. A neural network is this same loop with more layers in the middle; the [Neural Networks](neural-networks.md) page adds them.

!!! warning "Common pitfalls"
    - **Unstable softmax.** Always subtract the max. In PyTorch, use `F.softmax`, `F.log_softmax`, and `F.cross_entropy`, which do this for you.
    - **Passing probabilities to `F.cross_entropy`.** It expects logits and applies log-softmax internally. Passing softmax outputs computes the wrong loss and trains poorly.
    - **Forgetting the $1/N$ when the loss is a mean.** The gradient scales with it. Mismatched scaling silently changes your effective learning rate.
    - **Mixing log bases.** Losses in PyTorch use natural log (nats). Perplexity is $e^{\text{loss}}$, not $2^{\text{loss}}$.
    - **Reading KL as a distance.** It is asymmetric; which direction you use changes behavior (mode-seeking vs mass-covering).

## Check your understanding

??? question "What is the shape of $\partial L/\partial W$ for $W$ of shape (512, 2048)?"
    (512, 2048). A gradient always has the same shape as the tensor it is taken with respect to.

??? question "A 32,000-token vocabulary model has an initial loss of 10.4. Is that reasonable?"
    Yes. Uniform predictions give $\ln 32000 \approx 10.37$. A freshly initialized model with small output weights should be close to that.

??? question "Why does softmax give the same result if you add 100 to every logit?"
    $e^{z_i + c} = e^c e^{z_i}$, and the factor $e^c$ cancels between numerator and denominator. That is also why subtracting the max is safe.

??? question "The model gives the correct class probability 0.25. What is the gradient of the cross-entropy loss with respect to that class's logit?"
    $0.25 - 1 = -0.75$. Gradient descent subtracts the gradient, so this logit increases.

??? question "Why is minimizing cross-entropy equivalent to minimizing KL divergence from the data distribution?"
    $H(p, q) = H(p) + D_\text{KL}(p\|q)$, and $H(p)$ is a constant that does not depend on the model. So the two objectives differ by a constant and have the same minimizer and gradients.

## Further reading

- 3Blue1Brown, [Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra): visual intuition for vectors, matrices, and transforms
- Goodfellow, Bengio, Courville, [Deep Learning](https://www.deeplearningbook.org/), chapters 2-4 (linear algebra, probability and information theory, numerical computation)
- [Dive into Deep Learning](https://d2l.ai/), the appendix on mathematics for deep learning
- Christopher Olah, [Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/)

## Next

Next: [Python & PyTorch](pytorch.md)
