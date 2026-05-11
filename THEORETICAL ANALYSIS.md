# FedAD Theoretical Analysis

This document analyzes the training dynamics of `FedAD.py` under the fixed setting `UPLOAD_MODE="baseline"`. More specifically, we adopt the following modeling conventions:

- the local optimizer is Adam, and its state is reinitialized at the beginning of each round of local training, so both the momentum and the second moment are cold-started round by round;
- the shared parameters consist of the encoder and the task head, while the private parameters consist of the local adapter; the server aggregates only the shared parameters, and the private parameters always remain on the client side;
- a single learning rate is used for all trainable parameter blocks in the analysis, so it is uniformly denoted by `\eta`;
- in every round, all clients participate in training and all have valid samples, so the normalized weights $\widehat\omega_n^{(r)}$ are always positive.

The goal here is not to expand the internal Adam recursion term by term. Instead, under a set of explicit assumptions, we establish a conditional average-stationarity result for this surrogate training dynamics.

---

## 1. Problem Setup and an Analyzable Mathematical Update System

Let the shared parameters be

```math
\theta \in \mathbb R^{d_\theta},
```

and let the private parameters of client $n$ be

```math
\phi_n \in \mathbb R^{d_{\phi_n}}.
```

The dynamic joint objective at round $r$ is defined as

```math
\Psi^{(r)}(\theta,\phi_1,\dots,\phi_N)
:=
\sum_{n=1}^N \widehat\omega_n^{(r)} F_n^{(r)}(\theta,\phi_n),
\qquad
\sum_{n=1}^N \widehat\omega_n^{(r)} = 1,
\quad
\widehat\omega_n^{(r)} > 0.
```

Here $\widehat\omega_n^{(r)}$ denotes the normalized client weight at round $r$, satisfying $\sum_n \widehat\omega_n^{(r)}=1$.

The global state is denoted by

```math
z^{(r)} := (\theta^{(r)},\phi_1^{(r)},\dots,\phi_N^{(r)}).
```

The local state of client $n$ at local step $t$ of round $r$ is defined as

```math
x_n^{(r,t)} := (\theta_n^{(r,t)},\phi_n^{(r,t)}),
\qquad
x_n^{(r,0)} := (\theta^{(r)},\phi_n^{(r)}).
```

Let the common number of local steps executed by all clients at round $r$ be

```math
\tau_r.
```

Assume that there exists a uniform upper bound

```math
\tau_{\max} := \sup_r \tau_r < \infty,
\qquad
\tau_r \ge 1.
```

For any parameter block $b$, let its local Adam direction be

```math
u_{n,b}^{(r,t)}
:=
\frac{\hat m_{n,b}^{(r,t+1)}}{\sqrt{\hat v_{n,b}^{(r,t+1)}}+\varepsilon},
\qquad
t=0,\dots,\tau_r-1.
```

Here $\varepsilon>0$ is the numerical stabilization constant of Adam. We do not use the symbol $\tau$ for this quantity in order to avoid confusion with the local-step count $\tau_r$.

Since the local optimizer is cold-started at the beginning of every round, we have

```math
m_{n,b}^{(r,0)} = 0,
\qquad
v_{n,b}^{(r,0)} = 0.
```

Accordingly, the aggregated shared direction at local step $t$ of round $r$ is defined as

```math
U_\theta^{(r,t)}
:=
\sum_{n=1}^N \widehat\omega_n^{(r)} u_{n,\theta}^{(r,t)},
\qquad
t=0,\dots,\tau_r-1.
```

Hence, after one full communication round, the parameter updates can be written as

```math
\theta^{(r+1)}
:=
\theta^{(r)} - \eta \sum_{t=0}^{\tau_r-1} U_\theta^{(r,t)},
```

```math
\phi_n^{(r+1)}
:=
\phi_n^{(r)} - \eta \sum_{t=0}^{\tau_r-1} u_{n,\phi}^{(r,t)}.
```

Define

```math
\Delta_\theta^{(r)} := -\eta \sum_{t=0}^{\tau_r-1} U_\theta^{(r,t)},
\qquad
\Delta_{\phi_n}^{(r)} := -\eta \sum_{t=0}^{\tau_r-1} u_{n,\phi}^{(r,t)},
```

```math
\Delta z^{(r)} := (\Delta_\theta^{(r)},\Delta_{\phi_1}^{(r)},\dots,\Delta_{\phi_N}^{(r)}),
\qquad
z^{(r+1)} = z^{(r)} + \Delta z^{(r)}.
```

To standardize the notation for conditional expectations below, let $\mathcal F_{r,0}$ be the conditional information available before the beginning of round $r$, and define

```math
\mathbb E_r[\cdot] := \mathbb E[\cdot \mid \mathcal F_{r,0}].
```

For the local step $t$ within round $r$, let $\mathcal F_{r,t}$ denote the conditional information available before that step, and assume by default that $\mathcal F_{r,0}\subseteq \mathcal F_{r,t}$
holds. Therefore, any estimate involving $\mathbb E[\cdot\mid \mathcal F_{r,t}]$
that is derived later from the first-step lemma and Assumption A2 can be converted into an estimate involving $\mathbb E_r[\cdot]$
by the tower property. All intra-round random quantities in this document are interpreted under this convention.

Whenever sums over the local-step index $t$ appear below, they are understood by default to range over the common local-step interval $t=0,\dots,\tau_r-1$ for that round. If a certain expression requires client aggregation at local step $t$, then the summation is understood to be over all clients by default.

---

## 2. Weighted Geometry, Drift Quantities, and Stationarity Measures

To simultaneously characterize shared variables and private variables in a unified product space, for each round $r$ we introduce the weighted inner product and weighted norm induced by $\widehat\omega^{(r)}=(\widehat\omega_1^{(r)},\dots,\widehat\omega_N^{(r)})$:

```math
\langle z, z' \rangle_{w^{(r)}}
:=
\langle \theta,\theta' \rangle
+
\sum_{n=1}^N \widehat\omega_n^{(r)} \langle \phi_n,\phi_n' \rangle,
\qquad
z=(\theta,\phi_1,\dots,\phi_N),
\ z'=(\theta',\phi_1',\dots,\phi_N').
```

```math
\|z\|_{w^{(r)}}^2
:=
\|\theta\|^2 + \sum_{n=1}^N \widehat\omega_n^{(r)} \|\phi_n\|^2,
\qquad
z=(\theta,\phi_1,\dots,\phi_N).
```

Since $\widehat\omega_n^{(r)} > 0$ always holds, the above indeed defines a valid weighted geometric structure for every fixed round $r$. When round $r$ is fixed in the discussion below, we still write $\langle \cdot,\cdot\rangle_{w^{(r)}}$
and $\|\cdot\|_{w^{(r)}}$
simply as $\langle \cdot,\cdot\rangle_w$
and $\|\cdot\|_w$,
respectively, to simplify notation. Whenever a comparison across rounds is involved, it should be understood that the weighted geometry corresponding to that round is used. Unless otherwise stated, sums over client indices below always range over $n=1,\dots,N$ by default; if an expression depends on local step $t$, then the summation is still understood to be over all clients.

Under this weighted inner product at round $r$, we have

```math
\nabla_w \Psi^{(r)}(z)
=
\bigl(
\nabla_\theta \Psi^{(r)}(z),
\nabla_{\phi_1}F_1^{(r)}(\theta,\phi_1),
\dots,
\nabla_{\phi_N}F_N^{(r)}(\theta,\phi_N)
\bigr),
```

because under the standard Euclidean geometry, $\nabla_{\phi_n}\Psi^{(r)}(z) = \widehat\omega_n^{(r)}\nabla_{\phi_n}F_n^{(r)}(\theta,\phi_n)$,
and the weighted inner product absorbs this weight in the Riesz representation. Therefore, whenever the notation $\nabla \Psi^{(r)}(z)$
is used below, it refers by default to the gradient under the weighted geometry above.

To quantify the deviation of intra-round local states from the round-initial state, define the trajectory drift of client $n$ at local step $t$ of round $r$ as

```math
\rho_n^{(r,t)}
:=
\|\theta_n^{(r,t)}-\theta^{(r)}\|^2
+
\|\phi_n^{(r,t)}-\phi_n^{(r)}\|^2,
```

and further define, for any $t=0,\dots,\tau_r-1$,
the weighted drift quantity at local step $t$:

```math
D_t^{(r)}
:=
\sum_{n=1}^N \widehat\omega_n^{(r)} \rho_n^{(r,t)}.
```

We use the following quantities to measure the stationarity of the shared block and the private blocks at the round-initial point:

```math
G_\theta^{(r)} := \|\nabla_\theta \Psi^{(r)}(z^{(r)})\|^2,
```

```math
G_\phi^{(r)}
:=
\sum_{n=1}^N \widehat\omega_n^{(r)}
\|\nabla_{\phi_n} F_n^{(r)}(\theta^{(r)},\phi_n^{(r)})\|^2.
```

All descent analysis for the private block below is uniformly based on the round-initial stationarity quantity $G_\phi^{(r)}$, so no additional step-indexed private stationarity notation is introduced.

---

## 3. Basic Assumptions and the Fundamental Drift Lemma

The following explicit assumptions, together with the first-step lemma immediately after them, form the basis of all subsequent lemmas and the main theorem. What had previously been written directly as A4, namely the intra-round drift bound, is no longer presented as an independent assumption. Instead, it is derived as a basic lemma from A1, A2, A3, the update formulas in Section 1, and the condition $\tau_{\max}<\infty$.

### A1. Local Smoothness and Global Smoothness

For any $r,n$, the local objective $F_n^{(r)}(\theta,\phi_n)$ is differentiable with respect to $(\theta,\phi_n)$, and there exists a constant $L_F>0$ such that for any local states

```math
x=(\theta,\phi_n),
\qquad
x'=(\theta',\phi_n'),
```

we have

```math
\|\nabla F_n^{(r)}(x)-\nabla F_n^{(r)}(x')\|
\le
L_F \|x-x'\|.
```

At the same time, there exists a constant $L_\Psi>0$ such that for any round $r$ and any global states $z,z' \in \mathcal Z$,

```math
\Psi^{(r)}(z')
\le
\Psi^{(r)}(z)
+
\langle \nabla \Psi^{(r)}(z), z'-z \rangle_{w^{(r)}}
+
\frac{L_\Psi}{2}\|z'-z\|_{w^{(r)}}^2.
```

When a fixed round $r$ is under discussion below, the above can still be written in the simplified form using $\langle \cdot,\cdot\rangle_w$
and $\|\cdot\|_w$
according to the convention in Section 2.

### A2. Coordinatewise Gradient Boundedness at the Start of Each Round

There exist constants

```math
G_{\infty,\theta},G_{\infty,\phi}>0,
```

such that for any $n,r$,

```math
\|\nabla_\theta F_n^{(r)}(x_n^{(r,0)})\|_\infty \le G_{\infty,\theta},
\qquad
\|\nabla_{\phi_n} F_n^{(r)}(x_n^{(r,0)})\|_\infty \le G_{\infty,\phi}.
```

This condition is used only to express the first local Adam step as an explicit constant bound under cold start; the analysis of later steps does not depend on this assumption.

### Lemma A. Provable Bounds for the First Step of Cold-Started Adam

We first treat the first local step $t=0$ separately from the later local steps $t\ge 1$. The reason why the first step can be isolated is that Section 1 fixes the cold start of Adam at the beginning of every round: $m_{n,b}^{(r,0)}=v_{n,b}^{(r,0)}=0$. At this moment, both the bias-corrected first moment and second moment of Adam can be written explicitly.

Under the mathematical update system analyzed in Section 1, for a parameter block $b\in\{\theta,\phi_n\}$, let the gradient fed into Adam at the first step be

```math
g_{n,b}^{(r,0)}:=\nabla_b F_n^{(r)}(x_n^{(r,0)}).
```

Then the first-step direction satisfies the coordinatewise explicit formula

```math
u_{n,b}^{(r,0)}
=
\frac{g_{n,b}^{(r,0)}}{|g_{n,b}^{(r,0)}|+\varepsilon},
```

where the absolute value and the fraction are understood coordinatewise. If A2 holds, then there exist constants

```math
\underline c_{\theta}^{(0)}:=\frac{1}{G_{\infty,\theta}+\varepsilon},
\qquad
\underline c_{\phi}^{(0)}:=\frac{1}{G_{\infty,\phi}+\varepsilon},
\qquad
\overline c^{(0)}:=\frac{1}{\varepsilon^2},
```

such that for any $n,r$,

```math
\left\langle
\nabla_\theta F_n^{(r)}(x_n^{(r,0)}),
u_{n,\theta}^{(r,0)}
\right\rangle
\ge
\underline c_{\theta}^{(0)}
\left\|
\nabla_\theta F_n^{(r)}(x_n^{(r,0)})
\right\|^2,
```

```math
\left\langle
\nabla_{\phi_n} F_n^{(r)}(x_n^{(r,0)}),
u_{n,\phi}^{(r,0)}
\right\rangle
\ge
\underline c_{\phi}^{(0)}
\left\|
\nabla_{\phi_n} F_n^{(r)}(x_n^{(r,0)})
\right\|^2,
```

and also

```math
\|u_{n,\theta}^{(r,0)}\|^2
\le
\overline c^{(0)}
\left\|
\nabla_\theta F_n^{(r)}(x_n^{(r,0)})
\right\|^2,
```

```math
\|u_{n,\phi}^{(r,0)}\|^2
\le
\overline c^{(0)}
\left\|
\nabla_{\phi_n} F_n^{(r)}(x_n^{(r,0)})
\right\|^2.
```

### Proof

We use the shared block as an example. Under cold start and bias-corrected Adam,

```math
\hat m_{n,\theta}^{(r,1)}=g_{n,\theta}^{(r,0)},
\qquad
\hat v_{n,\theta}^{(r,1)}=\bigl(g_{n,\theta}^{(r,0)}\bigr)^{\odot 2},
```

so coordinatewise,

```math
u_{n,\theta,i}^{(r,0)}
=
\frac{g_{n,\theta,i}^{(r,0)}}{|g_{n,\theta,i}^{(r,0)}|+\varepsilon}.
```

Therefore,

```math
\left\langle
g_{n,\theta}^{(r,0)},
u_{n,\theta}^{(r,0)}
\right\rangle
=
\sum_i
\frac{\bigl(g_{n,\theta,i}^{(r,0)}\bigr)^2}{|g_{n,\theta,i}^{(r,0)}|+\varepsilon}
\ge
\frac{1}{G_{\infty,\theta}+\varepsilon}
\|g_{n,\theta}^{(r,0)}\|^2,
```

where the last step uses the coordinatewise boundedness in A2, $|g_{n,\theta,i}^{(r,0)}|\le G_{\infty,\theta}$.

Similarly,

```math
\|u_{n,\theta}^{(r,0)}\|^2
=
\sum_i
\frac{\bigl(g_{n,\theta,i}^{(r,0)}\bigr)^2}{(|g_{n,\theta,i}^{(r,0)}|+\varepsilon)^2}
\le
\frac{1}{\varepsilon^2}
\|g_{n,\theta}^{(r,0)}\|^2.
```

The private block is exactly analogous; one only needs to replace $G_{\infty,\theta}$ by $G_{\infty,\phi}$. $\square$

### A3. Stability Assumption for Later Intra-Round Steps

For later local steps $t=1,\dots,\tau_r-1$, let $\mathcal F_{r,t}$ be the conditional information available before the $t$-th local step of round $r$. There exist constants

```math
c_{\mathrm{al},\theta},\ c_{\mathrm{ub},\theta},\ c_{\mathrm{al},\phi},\ c_{\mathrm{ub},\phi} > 0,
```

and noise constants $\sigma_\theta,\sigma_\phi \ge 0$, such that for all $n,r$ and $t=1,\dots,\tau_r-1$,

```math
\mathbb E\!\left[
\left\langle
\nabla_\theta F_n^{(r)}(x_n^{(r,t)}),
u_{n,\theta}^{(r,t)}
\right\rangle
\middle| \mathcal F_{r,t}
\right]
\ge
c_{\mathrm{al},\theta}
\|\nabla_\theta F_n^{(r)}(x_n^{(r,t)})\|^2,
```

```math
\mathbb E\!\left[
\|u_{n,\theta}^{(r,t)}\|^2
\middle| \mathcal F_{r,t}
\right]
\le
c_{\mathrm{ub},\theta}
\bigl(
\|\nabla_\theta F_n^{(r)}(x_n^{(r,t)})\|^2 + \sigma_\theta^2
\bigr),
```

```math
\mathbb E\!\left[
\left\langle
\nabla_{\phi_n} F_n^{(r)}(x_n^{(r,t)}),
u_{n,\phi}^{(r,t)}
\right\rangle
\middle| \mathcal F_{r,t}
\right]
\ge
c_{\mathrm{al},\phi}
\|\nabla_{\phi_n} F_n^{(r)}(x_n^{(r,t)})\|^2,
```

```math
\mathbb E\!\left[
\|u_{n,\phi}^{(r,t)}\|^2
\middle| \mathcal F_{r,t}
\right]
\le
c_{\mathrm{ub},\phi}
\bigl(
\|\nabla_{\phi_n} F_n^{(r)}(x_n^{(r,t)})\|^2 + \sigma_\phi^2
\bigr).
```

This assumption covers only the intra-round dynamics after the first step. It abstracts the preconditioned update direction induced by cold-started Adam after the first local step as an effective descent direction that stays quantitatively aligned with the true gradient and has bounded second moment. In contrast, the bounds at $t=0$ are provided separately by the first-step lemma above.

To avoid repeatedly switching between "use Lemma A at $t=0$" and "use A2 at $t\ge 1$" later, we unify the two into one set of constants valid for all local steps. Define

```math
\underline c_\theta := \min\!\left\{\underline c_\theta^{(0)},\, c_{\mathrm{al},\theta}\right\},
\qquad
\underline c_\phi := \min\!\left\{\underline c_\phi^{(0)},\, c_{\mathrm{al},\phi}\right\},
```

```math
\overline c_\theta := \max\!\left\{\overline c^{(0)},\, c_{\mathrm{ub},\theta}\right\},
\qquad
\overline c_\phi := \max\!\left\{\overline c^{(0)},\, c_{\mathrm{ub},\phi}\right\}.
```

Then for any $n,r$ and any $t=0,\dots,\tau_r-1$, we have the unified one-step alignment / second-moment bounds

```math
\mathbb E\!\left[
\left\langle
\nabla_\theta F_n^{(r)}(x_n^{(r,t)}),
u_{n,\theta}^{(r,t)}
\right\rangle
\middle| \mathcal F_{r,t}
\right]
\ge
\underline c_\theta
\|\nabla_\theta F_n^{(r)}(x_n^{(r,t)})\|^2,
```

```math
\mathbb E\!\left[
\|u_{n,\theta}^{(r,t)}\|^2
\middle| \mathcal F_{r,t}
\right]
\le
\overline c_\theta
\bigl(
\|\nabla_\theta F_n^{(r)}(x_n^{(r,t)})\|^2 + \sigma_\theta^2
\bigr),
```

```math
\mathbb E\!\left[
\left\langle
\nabla_{\phi_n} F_n^{(r)}(x_n^{(r,t)}),
u_{n,\phi}^{(r,t)}
\right\rangle
\middle| \mathcal F_{r,t}
\right]
\ge
\underline c_\phi
\|\nabla_{\phi_n} F_n^{(r)}(x_n^{(r,t)})\|^2,
```

```math
\mathbb E\!\left[
\|u_{n,\phi}^{(r,t)}\|^2
\middle| \mathcal F_{r,t}
\right]
\le
\overline c_\phi
\bigl(
\|\nabla_{\phi_n} F_n^{(r)}(x_n^{(r,t)})\|^2 + \sigma_\phi^2
\bigr).
```

When $t=0$, these four inequalities follow directly from Lemma A together with $\sigma_\theta,\sigma_\phi\ge 0$;
when $t\ge 1$, they follow immediately from A2 together with the definitions of the minima and maxima above. Whenever the phrase "unified one-step alignment / second-moment bounds" is used below, it refers to these four inequalities valid for all local steps.

### A4. Bounded Shared Heterogeneity at the Start of Each Round

There exists a constant $\mathcal H_\theta^2 < \infty$ such that for any $r$,

```math
\sum_{n=1}^N \widehat\omega_n^{(r)}
\left\|
\nabla_\theta F_n^{(r)}(\theta^{(r)},\phi_n^{(r)})
-
\nabla_\theta \Psi^{(r)}(z^{(r)})
\right\|^2
\le
\mathcal H_\theta^2.
```

### Lemma 0. Controlled Drift of Intra-Round Local Trajectories

There exist constants $\overline C_{D,0},\overline C_{D,1}>0$, depending only on $L_F,\ \overline c_\theta,\ \overline c_\phi$
and $\tau_{\max}$, such that for any $r$ and any $t=0,\dots,\tau_r-1$,

```math
\mathbb E_r[D_t^{(r)}]
\le
\overline C_{D,0}\, t\, \eta^2
\Bigl(
G_\theta^{(r)} + G_\phi^{(r)} + \mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\Bigr)
+
\overline C_{D,1}\, \eta^2
\sum_{s=0}^{t-1}\mathbb E_r[D_s^{(r)}].
```

Therefore, there exist constants $\eta_D>0$ and $C_D>0$, again depending only on the constants above and on $\tau_{\max}$, such that whenever $0<\eta\le \eta_D$,

```math
\mathbb E_r[D_t^{(r)}]
\le
C_D\, t\, \eta^2
\Bigl(
G_\theta^{(r)} + G_\phi^{(r)} + \mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\Bigr).
```

### Proof

For any client $n$ and any local step $t\ge 1$, the update formulas in Section 1 give

```math
\theta_n^{(r,t)}-\theta^{(r)}
=
-\eta \sum_{s=0}^{t-1} u_{n,\theta}^{(r,s)},
\qquad
\phi_n^{(r,t)}-\phi_n^{(r)}
=
-\eta \sum_{s=0}^{t-1} u_{n,\phi}^{(r,s)}.
```

Hence, by Jensen's inequality,

```math
\rho_n^{(r,t)}
\le
t\eta^2\sum_{s=0}^{t-1}
\Bigl(
\|u_{n,\theta}^{(r,s)}\|^2
+
\|u_{n,\phi}^{(r,s)}\|^2
\Bigr).
```

Summing over all clients with weights gives

```math
D_t^{(r)}
\le
t\eta^2\sum_{s=0}^{t-1}\sum_{n=1}^N \widehat\omega_n^{(r)}
\Bigl(
\|u_{n,\theta}^{(r,s)}\|^2
+
\|u_{n,\phi}^{(r,s)}\|^2
\Bigr).
```

Taking $\mathbb E_r[\cdot]$ on both sides and applying the unified one-step second-moment bounds above yields

```math
\begin{aligned}
\mathbb E_r[D_t^{(r)}]
\le\;&
t\eta^2\sum_{s=0}^{t-1}\sum_{n=1}^N \widehat\omega_n^{(r)}
\Bigl[
\overline c_\theta\,
\mathbb E_r\|\nabla_\theta F_n^{(r)}(x_n^{(r,s)})\|^2 \\
&\qquad\qquad\qquad\qquad
+
\overline c_\phi\,
\mathbb E_r\|\nabla_{\phi_n} F_n^{(r)}(x_n^{(r,s)})\|^2
\Bigr] \\
&+
t\eta^2\sum_{s=0}^{t-1}
\bigl(
\overline c_\theta\sigma_\theta^2
+
\overline c_\phi\sigma_\phi^2
\bigr).
\end{aligned}
```

We next estimate the squared gradient terms of the shared block and the private block separately. Consider the shared block first. By the local smoothness in A1,

```math
\|\nabla_\theta F_n^{(r)}(x_n^{(r,s)})\|^2
\le
2\|\nabla_\theta F_n^{(r)}(x_n^{(r,0)})\|^2
+
2L_F^2 \rho_n^{(r,s)}.
```

Using further that $\nabla_\theta F_n^{(r)}(x_n^{(r,0)}) = \nabla_\theta \Psi^{(r)}(z^{(r)}) + \bigl[\nabla_\theta F_n^{(r)}(x_n^{(r,0)})-\nabla_\theta \Psi^{(r)}(z^{(r)})\bigr]$
and applying A4, we obtain

```math
\sum_{n=1}^N \widehat\omega_n^{(r)}
\|\nabla_\theta F_n^{(r)}(x_n^{(r,0)})\|^2
\le
2G_\theta^{(r)} + 2\mathcal H_\theta^2.
```

Hence,

```math
\sum_{n=1}^N \widehat\omega_n^{(r)}
\mathbb E_r\|\nabla_\theta F_n^{(r)}(x_n^{(r,s)})\|^2
\le
4G_\theta^{(r)} + 4\mathcal H_\theta^2 + 2L_F^2 \mathbb E_r[D_s^{(r)}].
```

Now consider the private block. By A1,

```math
\|\nabla_{\phi_n}F_n^{(r)}(x_n^{(r,s)})\|^2
\le
2\|\nabla_{\phi_n}F_n^{(r)}(x_n^{(r,0)})\|^2
+
2L_F^2 \rho_n^{(r,s)}.
```

Weighting and summing over all clients, and using the definitions of $G_\phi^{(r)}$ and $D_s^{(r)}$, yields

```math
\sum_{n=1}^N \widehat\omega_n^{(r)}
\mathbb E_r\|\nabla_{\phi_n}F_n^{(r)}(x_n^{(r,s)})\|^2
\le
2G_\phi^{(r)} + 2L_F^2 \mathbb E_r[D_s^{(r)}].
```

Substituting these bounds back in, there exist constants $\widetilde C_{D,0},\widetilde C_{D,1}>0$, depending only on $L_F,\ \overline c_\theta,\ \overline c_\phi$,
such that

```math
\mathbb E_r[D_t^{(r)}]
\le
t\eta^2\sum_{s=0}^{t-1}
\Bigl[
\widetilde C_{D,0}
\bigl(
G_\theta^{(r)} + G_\phi^{(r)} + \mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr)
+
\widetilde C_{D,1}\mathbb E_r[D_s^{(r)}]
\Bigr].
```

Using next the bound $t\le \tau_r\le \tau_{\max}$, the above can be rewritten as

```math
\mathbb E_r[D_t^{(r)}]
\le
\overline C_{D,0}\, t\, \eta^2
\Bigl(
G_\theta^{(r)} + G_\phi^{(r)} + \mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\Bigr)
+
\overline C_{D,1}\, \eta^2
\sum_{s=0}^{t-1}\mathbb E_r[D_s^{(r)}],
```

which is the recursive drift bound.

We now close it into an explicit $O(t\eta^2)$ upper bound. Let

```math
\mathcal A_r
:=
G_\theta^{(r)} + G_\phi^{(r)} + \mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2.
```

Choose

```math
C_D := 2\overline C_{D,0},
\qquad
\eta_D := \min\left\{1,\ \frac{1}{\tau_{\max}\sqrt{\overline C_{D,1}}}\right\}.
```

Proceed by induction on $t$. The case $t=0$ is immediate, since $D_0^{(r)}=0$. Assume that for all $s<t$, $\mathbb E_r[D_s^{(r)}]\le C_D s\eta^2\mathcal A_r$.
Then

```math
\sum_{s=0}^{t-1}\mathbb E_r[D_s^{(r)}]
\le
C_D\eta^2\mathcal A_r \sum_{s=0}^{t-1}s
\le
\frac{C_D}{2}\tau_{\max} t\,\eta^2\mathcal A_r.
```

Plugging this back into the recursive bound gives

```math
\mathbb E_r[D_t^{(r)}]
\le
\left(
\overline C_{D,0}
+
\frac{\overline C_{D,1}C_D\tau_{\max}\eta^2}{2}
\right)
t\eta^2\mathcal A_r.
```

When $0<\eta\le \eta_D$, $\overline C_{D,1}\tau_{\max}^2\eta^2\le 1$,
so

```math
\frac{\overline C_{D,1}C_D\tau_{\max}\eta^2}{2}
=
\overline C_{D,0}\,\overline C_{D,1}\tau_{\max}\eta^2
\le
\overline C_{D,0}.
```

Therefore,

```math
\mathbb E_r[D_t^{(r)}]
\le
2\overline C_{D,0}\, t\, \eta^2\mathcal A_r
=
C_D\, t\, \eta^2\mathcal A_r.
```

The induction is complete, and the lemma follows. $\square$

### A5. Uniformly Bounded Cross-Round Function Drift

Let $\mathcal Z_{\mathrm{tr}} \subset \mathcal Z$ be a common bounded region containing all actual iterates, and define

```math
B_\Psi
:=
\sup_r
\left(
\sup_{z \in \mathcal Z_{\mathrm{tr}}}
|\Psi^{(r+1)}(z)-\Psi^{(r)}(z)|
+
\left|
\inf_{z \in \mathcal Z_{\mathrm{tr}}}\Psi^{(r+1)}(z)
-
\inf_{z \in \mathcal Z_{\mathrm{tr}}}\Psi^{(r)}(z)
\right|
\right)
< \infty.
```

### A6. Lower Bounded Objective

Define

```math
\underline\Psi := \inf_{r,\;z \in \mathcal Z_{\mathrm{tr}}} \Psi^{(r)}(z) > -\infty.
```

---

## 4. Descent Properties of the Shared Direction

### Lemma 1. Lower Bound on the Inner Product of the Shared Aggregated Direction

There exist constants $\alpha_\theta,\beta_\theta>0$ such that for any $r$ and any $t=0,\dots,\tau_r-1$,

```math
\mathbb E_r\!\left[
\left\langle
\nabla_\theta \Psi^{(r)}(z^{(r)}),
U_\theta^{(r,t)}
\right\rangle
\right]
\ge
\alpha_\theta G_\theta^{(r)}
-
\beta_\theta
\bigl(
\mathcal H_\theta^2 + L_F^2 \mathbb E_r[D_t^{(r)}] + \sigma_\theta^2
\bigr).
```

### Proof

To simplify notation, write

```math
a^{(r)} := \nabla_\theta \Psi^{(r)}(z^{(r)}),
\qquad
g_n^{(r,t)} := \nabla_\theta F_n^{(r)}(x_n^{(r,t)}).
```

By the definition of the shared aggregated direction,

```math
\left\langle a^{(r)}, U_\theta^{(r,t)} \right\rangle
=
\sum_{n=1}^N \widehat\omega_n^{(r)}
\left\langle a^{(r)}, u_{n,\theta}^{(r,t)} \right\rangle.
```

For any client $n$, decompose $a^{(r)}$ as

```math
a^{(r)} = g_n^{(r,t)} + \bigl(a^{(r)}-g_n^{(r,t)}\bigr),
```

and apply Young's inequality $\langle p,q\rangle \ge -\frac{\gamma}{2}\|q\|^2-\frac{1}{2\gamma}\|p\|^2$
with $\gamma=\underline c_\theta/\overline c_\theta$.
Combining this with the unified one-step alignment / second-moment bounds yields

```math
\mathbb E_r\!\left[
\left\langle a^{(r)}, U_\theta^{(r,t)} \right\rangle
\right]
\ge
\frac{\underline c_\theta}{2}
\sum_{n=1}^N \widehat\omega_n^{(r)} \mathbb E_r\|g_n^{(r,t)}\|^2
-
\frac{\overline c_\theta}{2\underline c_\theta}
\sum_{n=1}^N \widehat\omega_n^{(r)} \mathbb E_r\|a^{(r)}-g_n^{(r,t)}\|^2
-
\frac{\underline c_\theta}{2} \sigma_\theta^2.
```

On the other hand, for any vectors $x,y$,

```math
\|y\|^2
=
\|x+(y-x)\|^2
\le
2\|x\|^2 + 2\|y-x\|^2,
```

which is equivalent to

```math
\|x\|^2
\ge
\frac12\|y\|^2 - \|x-y\|^2.
```

Taking $x=g_n^{(r,t)}$ and $y=a^{(r)}$, then weighting and summing over all clients, we obtain

```math
\sum_{n=1}^N \widehat\omega_n^{(r)} \mathbb E_r\|g_n^{(r,t)}\|^2
\ge
\frac12 G_\theta^{(r)}
-
\sum_{n=1}^N \widehat\omega_n^{(r)} \mathbb E_r\|g_n^{(r,t)}-a^{(r)}\|^2.
```

We next decompose this deviation term into the sum of the intra-round trajectory-drift error and the round-initial shared-heterogeneity error:

```math
g_n^{(r,t)} - a^{(r)}
=
\bigl[
\nabla_\theta F_n^{(r)}(x_n^{(r,t)})
-
\nabla_\theta F_n^{(r)}(x_n^{(r,0)})
\bigr]
+
\bigl[
\nabla_\theta F_n^{(r)}(x_n^{(r,0)}) - a^{(r)}
\bigr].
```

By the smoothness in A1 and the heterogeneity control in A4,

```math
\sum_{n=1}^N \widehat\omega_n^{(r)} \mathbb E_r\|g_n^{(r,t)}-a^{(r)}\|^2
\le
2L_F^2 \mathbb E_r[D_t^{(r)}] + 2\mathcal H_\theta^2.
```

Therefore, there exists a constant $\widetilde\beta_\theta>0$, depending only on $\underline c_\theta,\overline c_\theta$,
such that

```math
\mathbb E_r\!\left[
\left\langle a^{(r)}, U_\theta^{(r,t)} \right\rangle
\right]
\ge
\frac{\underline c_\theta}{4} G_\theta^{(r)}
-
\widetilde\beta_\theta
\bigl(
\mathcal H_\theta^2 + L_F^2 \mathbb E_r[D_t^{(r)}]
\bigr)
-
\frac{\underline c_\theta}{2} \sigma_\theta^2.
```

Renaming constants gives the stated result. $\square$

### Lemma 2. Upper Bound on the Second Moment of the Shared Aggregated Direction

There exists a constant $\kappa_\theta>0$ such that for any $r$ and any $t=0,\dots,\tau_r-1$,

```math
\mathbb E_r \|U_\theta^{(r,t)}\|^2
\le
\kappa_\theta
\bigl(
G_\theta^{(r)} + \mathcal H_\theta^2 + L_F^2 \mathbb E_r[D_t^{(r)}] + \sigma_\theta^2
\bigr).
```

### Proof

By the Jensen-type estimate for convex combinations, $\|\sum_i a_i x_i\|^2 \le (\sum_i a_i)\sum_i a_i\|x_i\|^2$
and $\sum_{n=1}^N \widehat\omega_n^{(r)} = 1$,

```math
\|U_\theta^{(r,t)}\|^2
\le
\sum_{n=1}^N \widehat\omega_n^{(r)} \|u_{n,\theta}^{(r,t)}\|^2.
```

Applying the unified one-step second-moment bounds and the gradient-deviation decomposition from A1 and A3, we obtain

```math
\mathbb E_r \|U_\theta^{(r,t)}\|^2
\le
\overline c_\theta
\sum_{n=1}^N \widehat\omega_n^{(r)} \mathbb E_r\|\nabla_\theta F_n^{(r)}(x_n^{(r,t)})\|^2
+
\overline c_\theta\sigma_\theta^2
\le
\kappa_\theta
\bigl(
G_\theta^{(r)} + \mathcal H_\theta^2 + L_F^2 \mathbb E_r[D_t^{(r)}] + \sigma_\theta^2
\bigr).
```

$\square$

---

## 5. Descent Properties of the Private Direction

### Lemma 3. Lower Bound on the Inner Product Along the Private Direction

There exist constants $\alpha_\phi,\beta_\phi>0$ such that for any $r$ and any $t=0,\dots,\tau_r-1$,

```math
\sum_{n=1}^N \widehat\omega_n^{(r)}
\mathbb E_r\!\left[
\left\langle
\nabla_{\phi_n} F_n^{(r)}(x_n^{(r,0)}),
u_{n,\phi}^{(r,t)}
\right\rangle
\right]
\ge
\alpha_\phi G_\phi^{(r)}
-
\beta_\phi
\bigl(
L_F^2 \mathbb E_r[D_t^{(r)}] + \sigma_\phi^2
\bigr).
```

### Proof

To simplify notation, let

```math
b_n^{(r)} := \nabla_{\phi_n} F_n^{(r)}(x_n^{(r,0)}),
\qquad
h_n^{(r,t)} := \nabla_{\phi_n} F_n^{(r)}(x_n^{(r,t)}).
```

Then

```math
\langle b_n^{(r)},u_{n,\phi}^{(r,t)}\rangle
=
\langle h_n^{(r,t)},u_{n,\phi}^{(r,t)}\rangle
+
\langle b_n^{(r)}-h_n^{(r,t)},u_{n,\phi}^{(r,t)}\rangle.
```

Apply Young's inequality $\langle p,q\rangle \ge -\frac{\gamma}{2}\|q\|^2-\frac{1}{2\gamma}\|p\|^2$
with $\gamma=\underline c_\phi/\overline c_\phi$,
then combine it with the unified one-step alignment / second-moment bounds. This yields

```math
\mathbb E_r\!\left[
\langle b_n^{(r)},u_{n,\phi}^{(r,t)}\rangle
\right]
\ge
\frac{\underline c_\phi}{2}\mathbb E_r\|h_n^{(r,t)}\|^2
-
\frac{\overline c_\phi}{2\underline c_\phi}
\mathbb E_r\|b_n^{(r)}-h_n^{(r,t)}\|^2
-
\frac{\underline c_\phi}{2}\sigma_\phi^2.
```

On the other hand, using the identity

```math
\|h_n^{(r,t)}\|^2
\ge
\frac12 \|b_n^{(r)}\|^2 - \|h_n^{(r,t)}-b_n^{(r)}\|^2,
```

we obtain

```math
\frac{\underline c_\phi}{2}\mathbb E_r\|h_n^{(r,t)}\|^2
\ge
\frac{\underline c_\phi}{4}\|b_n^{(r)}\|^2
-
\frac{\underline c_\phi}{2}\mathbb E_r\|h_n^{(r,t)}-b_n^{(r)}\|^2.
```

By the smoothness in A1,

```math
\mathbb E_r\|b_n^{(r)}-h_n^{(r,t)}\|^2
\le
L_F^2 \mathbb E_r[\rho_n^{(r,t)}].
```

Therefore, there exists a constant $\widetilde\beta_\phi>0$, depending only on $\underline c_\phi,\overline c_\phi$,
such that

```math
\mathbb E_r\!\left[
\langle b_n^{(r)},u_{n,\phi}^{(r,t)}\rangle
\right]
\ge
\frac{\underline c_\phi}{4}\|b_n^{(r)}\|^2
-
\widetilde\beta_\phi L_F^2 \mathbb E_r[\rho_n^{(r,t)}]
-
\frac{\underline c_\phi}{2}\sigma_\phi^2.
```

Finally, weight and sum this over all clients, use $G_\phi^{(r)}=\sum_{n=1}^N \widehat\omega_n^{(r)}\|b_n^{(r)}\|^2$
and $D_t^{(r)}=\sum_{n=1}^N \widehat\omega_n^{(r)}\rho_n^{(r,t)}$,
and rename constants to obtain the conclusion. $\square$

### Lemma 4. Upper Bound on the Second Moment Along the Private Direction

There exists a constant $\kappa_\phi>0$ such that for any $r$ and any $t=0,\dots,\tau_r-1$,

```math
\sum_{n=1}^N \widehat\omega_n^{(r)}
\mathbb E_r \|u_{n,\phi}^{(r,t)}\|^2
\le
\kappa_\phi
\bigl(
G_\phi^{(r)} + L_F^2 \mathbb E_r[D_t^{(r)}] + \sigma_\phi^2
\bigr).
```

### Proof

By the unified one-step second-moment bound,

```math
\sum_{n=1}^N \widehat\omega_n^{(r)} \mathbb E_r \|u_{n,\phi}^{(r,t)}\|^2
\le
\overline c_\phi
\sum_{n=1}^N \widehat\omega_n^{(r)} \mathbb E_r\|\nabla_{\phi_n} F_n^{(r)}(x_n^{(r,t)})\|^2
+
\overline c_\phi\sigma_\phi^2.
```

Using A1 again to decompose the intra-round gradient term into the round-initial gradient term and the drift error term gives the stated upper bound. $\square$

---

## 6. Single-Round Expected Descent Analysis

### Lemma 5. Single-Round Descent Inequality

There exist constants $C_1,C_2,C_3,C_4>0$ such that when the learning rate $\eta$ is sufficiently small, for any $r$,

```math
\begin{aligned}
\mathbb E_r[\Psi^{(r)}(z^{(r+1)})]
\le\;&
\Psi^{(r)}(z^{(r)})
-
C_1 \eta G_\theta^{(r)}
-
C_2 \eta G_\phi^{(r)} \\
&+
C_3 \eta
\bigl(
\mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr) \\
&+
C_4 \eta^2
\bigl(
G_\theta^{(r)} + G_\phi^{(r)} + \mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr).
\end{aligned}
```

### Proof

By the global smoothness of $\Psi^{(r)}$ in A1,

```math
\Psi^{(r)}(z^{(r+1)})
\le
\Psi^{(r)}(z^{(r)})
+
\langle \nabla \Psi^{(r)}(z^{(r)}), \Delta z^{(r)} \rangle_w
+
\frac{L_\Psi}{2}\|\Delta z^{(r)}\|_w^2.
```

Let

```math
A_r := \langle \nabla \Psi^{(r)}(z^{(r)}), \Delta z^{(r)} \rangle_w,
\qquad
B_r := \|\Delta z^{(r)}\|_w^2.
```

Then

```math
\mathbb E_r[\Psi^{(r)}(z^{(r+1)})]
\le
\Psi^{(r)}(z^{(r)})
+
\mathbb E_r[A_r]
+
\frac{L_\Psi}{2}\mathbb E_r[B_r].
```

We first handle the first-order term. By the update formulas in Section 1,

```math
A_r
=
-\eta \sum_{t=0}^{\tau_r-1}
\left\langle
\nabla_\theta \Psi^{(r)}(z^{(r)}),
U_\theta^{(r,t)}
\right\rangle
-\eta \sum_{t=0}^{\tau_r-1}
\sum_{n=1}^N \widehat\omega_n^{(r)}
\left\langle
\nabla_{\phi_n}F_n^{(r)}(x_n^{(r,0)}),
u_{n,\phi}^{(r,t)}
\right\rangle.
```

Apply Lemma 1 and Lemma 3 to each $t$, then sum over $t$, to obtain

```math
\begin{aligned}
\mathbb E_r[A_r]
\le\;&
-\eta \tau_r \alpha_\theta G_\theta^{(r)}
-\eta \tau_r \alpha_\phi G_\phi^{(r)} \\
&+
\eta \tau_r \beta_\theta
\bigl(
\mathcal H_\theta^2 + \sigma_\theta^2
\bigr)
+
\eta \tau_r \beta_\phi \sigma_\phi^2 \\
&+
\eta (\beta_\theta+\beta_\phi)L_F^2
\sum_{t=0}^{\tau_r-1}\mathbb E_r[D_t^{(r)}].
\end{aligned}
```

Using $1\le \tau_r \le \tau_{\max}$,
there exist constants $C_{A,0},C_{A,1}>0$ such that

```math
\mathbb E_r[A_r]
\le
-\eta \alpha_\theta G_\theta^{(r)}
-\eta \alpha_\phi G_\phi^{(r)}
+
C_{A,0}\eta
\bigl(
\mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr)
+
C_{A,1}\eta
\sum_{t=0}^{\tau_r-1}\mathbb E_r[D_t^{(r)}].
```

Then by Lemma 0,

```math
\sum_{t=0}^{\tau_r-1}\mathbb E_r[D_t^{(r)}]
\le
C_D
\left(
\sum_{t=0}^{\tau_r-1} t
\right)
\eta^2
\bigl(
G_\theta^{(r)} + G_\phi^{(r)} + \mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr).
```

Combining this with $\tau_r \le \tau_{\max}$,
there exists a constant $C_{A,2}>0$ such that

```math
C_{A,1}\eta
\sum_{t=0}^{\tau_r-1}\mathbb E_r[D_t^{(r)}]
\le
C_{A,2}\eta^3
\bigl(
G_\theta^{(r)} + G_\phi^{(r)} + \mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr).
```

Hence,

```math
\mathbb E_r[A_r]
\le
-\eta \alpha_\theta G_\theta^{(r)}
-\eta \alpha_\phi G_\phi^{(r)}
+
C_{A,0}\eta
\bigl(
\mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr)
+
C_{A,2}\eta^3
\bigl(
G_\theta^{(r)} + G_\phi^{(r)} + \mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr).
```

Next, handle the quadratic term. By Jensen's inequality,

```math
\|\Delta_\theta^{(r)}\|^2
\le
\eta^2 \tau_r
\sum_{t=0}^{\tau_r-1} \|U_\theta^{(r,t)}\|^2
\le
\eta^2 \tau_{\max}
\sum_{t=0}^{\tau_r-1} \|U_\theta^{(r,t)}\|^2,
```

```math
\sum_n \widehat\omega_n^{(r)} \|\Delta_{\phi_n}^{(r)}\|^2
\le
\eta^2 \tau_r
\sum_{t=0}^{\tau_r-1}
\sum_{n=1}^N \widehat\omega_n^{(r)} \|u_{n,\phi}^{(r,t)}\|^2
\le
\eta^2 \tau_{\max}
\sum_{t=0}^{\tau_r-1}
\sum_{n=1}^N \widehat\omega_n^{(r)} \|u_{n,\phi}^{(r,t)}\|^2.
```

Therefore,

```math
\mathbb E_r[B_r]
\le
\eta^2 \tau_{\max}
\sum_{t=0}^{\tau_r-1}
\left(
\mathbb E_r\|U_\theta^{(r,t)}\|^2
+
\sum_{n=1}^N \widehat\omega_n^{(r)} \mathbb E_r \|u_{n,\phi}^{(r,t)}\|^2
\right).
```

Applying Lemma 2 and Lemma 4, there exist constants $C_{B,0},C_{B,1}>0$ such that

```math
\mathbb E_r[B_r]
\le
\eta^2 \tau_{\max}
\sum_{t=0}^{\tau_r-1}
\Bigl[
C_{B,0}
\bigl(
G_\theta^{(r)} + G_\phi^{(r)} + \mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr)
+
C_{B,1}L_F^2 \mathbb E_r[D_t^{(r)}]
\Bigr].
```

Using again $\tau_r \le \tau_{\max}$
and the bound on $\sum_{t=0}^{\tau_r-1}\mathbb E_r[D_t^{(r)}]$,
there exists a constant $\widetilde C_B>0$ such that

```math
\mathbb E_r[B_r]
\le
\widetilde C_B \eta^2
\bigl(
G_\theta^{(r)} + G_\phi^{(r)} + \mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr).
```

Substituting the bounds for the first-order and quadratic terms back into the smoothness inequality, and noting that the additional term caused by drift in the first-order part is $O(\eta^3)$ and can therefore be absorbed into the $O(\eta^2)$ correction of the quadratic term, we conclude that when $\eta$ is sufficiently small, there exist constants $C_1,C_2,C_3,C_4>0$ such that

```math
\begin{aligned}
\mathbb E_r[\Psi^{(r)}(z^{(r+1)})]
\le\;&
\Psi^{(r)}(z^{(r)})
-
C_1 \eta G_\theta^{(r)}
-
C_2 \eta G_\phi^{(r)} \\
&+
C_3 \eta
\bigl(
\mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr) \\
&+
C_4 \eta^2
\bigl(
G_\theta^{(r)} + G_\phi^{(r)} + \mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr),
\end{aligned}
```

which is exactly the desired conclusion. $\square$

---

## 7. Main Theorem: Average Stationarity Bound

Define

```math
\lambda := \frac{C_2}{C_1},
\qquad
\mathcal G^{(r)} := G_\theta^{(r)} + \lambda G_\phi^{(r)}.
```

### Theorem 1. Conditional Average Stationarity Bound for the Abstract Mathematical Update System on the Dynamic Joint Objective

Under Assumptions A1, A2, A3, A4, A5, A6, the conditional-expectation notation of Section 1, and the modeling scope "the object of analysis is the training dynamics of `FedAD.py` under `UPLOAD_MODE="baseline"`," if the learning rate $\eta$ is sufficiently small, then there exist constants $c_0,c_1,c_2,c_3>0$, depending only on the constants in the assumptions and independent of $R$ and $\eta$, such that

```math
\frac{1}{R}
\sum_{r=0}^{R-1}
\mathbb E[\mathcal G^{(r)}]
\le
\frac{c_0\bigl(\Psi^{(0)}(z^{(0)})-\underline\Psi\bigr)}{R\eta}
+
\frac{c_1 B_\Psi}{\eta}
+
c_2
\bigl(
\mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr)
+
c_3 \eta
\bigl(
\mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr).
```

### Proof

By Lemma 5, there exist constants $C_1,C_2,C_3,C_4>0$ such that for any $r$,

```math
\mathbb E_r[\Psi^{(r)}(z^{(r+1)})]
\le
\Psi^{(r)}(z^{(r)})
+
\eta(C_4\eta-C_1) G_\theta^{(r)}
+
\eta(C_4\eta-C_2) G_\phi^{(r)}
+
C_3 \eta
\bigl(
\mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr)
+
C_4 \eta^2
\bigl(
\mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr).
```

Let

```math
\eta_\star := \min\left\{\frac{C_1}{2C_4},\,\frac{C_2}{2C_4}\right\}.
```

If $0<\eta\le \eta_\star$, then

```math
\eta(C_4\eta-C_1)\le -\frac{C_1}{2}\eta,
\qquad
\eta(C_4\eta-C_2)\le -\frac{C_2}{2}\eta.
```

Thus,

```math
\mathbb E_r[\Psi^{(r)}(z^{(r+1)})]
\le
\Psi^{(r)}(z^{(r)})
-
\frac{C_1}{2}\eta G_\theta^{(r)}
-
\frac{C_2}{2}\eta G_\phi^{(r)}
+
C_3 \eta
\bigl(
\mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr)
+
C_4 \eta^2
\bigl(
\mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr).
```

Define further

```math
\widetilde c := \frac{C_1}{2},
\qquad
\widetilde C_1 := C_3,
\qquad
\widetilde C_2 := C_4.
```

Since $\lambda = C_2/C_1$
and $\mathcal G^{(r)}=G_\theta^{(r)}+\lambda G_\phi^{(r)}$,
we have $\widetilde c\,\eta\,\mathcal G^{(r)} = \frac{C_1}{2}\eta G_\theta^{(r)} + \frac{C_2}{2}\eta G_\phi^{(r)}.$
Hence the inequality above can be rewritten as

```math
\mathbb E_r[\Psi^{(r)}(z^{(r+1)})]
\le
\Psi^{(r)}(z^{(r)})
-
\widetilde c \eta \mathcal G^{(r)}
+
\widetilde C_1 \eta
\bigl(
\mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr)
+
\widetilde C_2 \eta^2
\bigl(
\mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr).
```

Define the round-dependent gap

```math
\Phi^{(r)}
:=
\Psi^{(r)}(z^{(r)})
-
\inf_{z \in \mathcal Z_{\mathrm{tr}}}\Psi^{(r)}(z).
```

By the definition of $B_\Psi$ in A5,

```math
\mathbb E_r[\Phi^{(r+1)}]
\le
\mathbb E_r[\Psi^{(r)}(z^{(r+1)})]
-
\inf_{z \in \mathcal Z_{\mathrm{tr}}}\Psi^{(r)}(z)
+
B_\Psi.
```

Therefore,

```math
\widetilde c \eta \mathcal G^{(r)}
\le
\Phi^{(r)} - \mathbb E_r[\Phi^{(r+1)}]
+
\widetilde C_1 \eta
\bigl(
\mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr)
+
\widetilde C_2 \eta^2
\bigl(
\mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr)
+
B_\Psi.
```

Take full expectation and sum over $r=0,\dots,R-1$ to obtain

```math
\widetilde c \eta
\sum_{r=0}^{R-1}\mathbb E[\mathcal G^{(r)}]
\le
\Phi^{(0)} - \mathbb E[\Phi^{(R)}]
+
R\widetilde C_1 \eta
\bigl(
\mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr)
+
R\widetilde C_2 \eta^2
\bigl(
\mathcal H_\theta^2 + \sigma_\theta^2 + \sigma_\phi^2
\bigr)
+
R B_\Psi.
```

By definition, $\Phi^{(R)} \ge 0$; by A6,

```math
\Phi^{(0)}
\le
\Psi^{(0)}(z^{(0)}) - \underline\Psi.
```

Dividing both sides by $R\widetilde c \eta$ and renaming constants to $c_0,c_1,c_2,c_3$ gives the theorem. $\square$

### Remark

The theorem above should be understood as follows: under the joint validity of A1, A2, A3, A4, A5, A6 and the conditional-expectation notation introduced in Section 1, and when the learning rate lies in the small-step regime required by Lemma 0, one can establish a conditional average stationarity bound for the more regular mathematical update system abstracted from the real training process in Section 1. It **should not be interpreted directly** as an unconditional convergence theorem for the exact batch-by-batch execution path of the original `FedAD.py`. To derive the latter, at least the following two issues would still need to be addressed:

1. the additional local-update benefit caused by the round-varying local-step counts is currently absorbed into constants through upper bounds, rather than being characterized sharply as a function of the sequence $\tau_r$;
2. although the first-step bound for cold-started Adam can be derived explicitly under the coordinatewise boundedness assumption A2, the alignment and second-moment control of later steps are still absorbed into the stability assumption A3 rather than being derived automatically from the code implementation itself.

---

## 8. Conclusion and Scope of Applicability

The theorem shows that, under the strong assumptions above, the mathematical update system abstracted from the real training process in Section 1 approaches an average stationary neighborhood on the dynamic joint objective

```math
\Psi^{(r)}(\theta,\phi_1,\dots,\phi_N)
:=
\sum_{n=1}^N \widehat\omega_n^{(r)} F_n^{(r)}(\theta,\phi_n).
```

This neighborhood is jointly determined by four parts:

1. the finite-round term $O(1/(R\eta))$ induced by the initial gap;
2. the cross-round environment-drift term $O(B_\Psi/\eta)$;
3. the constant-neighborhood term caused by shared heterogeneity and stochastic noise;
4. the higher-order learning-rate term $O(\eta)$.

Therefore, the intuition that "reducing the learning rate improves the average stationarity bound" is valid only when the cross-round drift is sufficiently small, especially in the static-objective case $B_\Psi=0$, or more strongly when $B_\Psi=O(\eta)$. If $B_\Psi$ is independent of $\eta$, then an excessively small learning rate actually amplifies this drift term.

This is a **conditional theoretical result**. The strongest assumptions are the "later-step stability assumption A3," A5, and the technical condition that the number of local steps in each round is uniformly bounded. What had previously been written as a separate drift-control assumption has now been refined in Section 3 into the provable Lemma 0:

- the first local step of cold-started Adam can be derived explicitly under the coordinatewise boundedness assumption A2, while later steps are abstracted in A3 as "effective directional alignment + second-moment control";
- Lemma 0 derives the $O(t\eta^2)$ upper bound for intra-round local trajectory drift from the update formulas, A1, A3, A4, and the small-step condition;
- Section 1 requires all clients in round $r$ to execute the same number $\tau_r$ of local steps, while allowing $\tau_r$ to vary across rounds subject to the common upper bound $\tau_{\max}$;
- Section 1 also fixes the conditional-expectation notation $\mathbb E_r[\cdot]=\mathbb E[\cdot\mid\mathcal F_{r,0}]$ and uses the tower property implicitly to handle intra-round randomness.

Therefore, a more precise statement of this document is:

> Under the explicit strong assumptions A1, A2, A3, A4, A5, A6, together with the conditional-expectation notation introduced in Section 1 and the modeling scope "the object of analysis is the training dynamics of `FedAD.py` under `UPLOAD_MODE="baseline"`," and provided that the learning rate lies in the small-step regime required by Lemma 0, one can establish an average stationarity bound for an analyzable abstraction of that training dynamics.
