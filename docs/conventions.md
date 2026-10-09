# Conventions

Definitions used throughout `tdalitz`.

## Units

* Masses in GeV, invariant masses squared in GeV².
* Times in units of the mean lifetime: $\tau = \Gamma t$, with $\Gamma = (\Gamma_L + \Gamma_H)/2$.
* Time resolution $\sigma_\tau$ is also in units of mean lifetime. 

## Dalitz plot

A spin-0 parent of mass $M$ decays to daughters 1, 2, 3.
The two coordinates are

```math
s_{12} = (p_1 + p_2)^2, \qquad s_{13} = (p_1 + p_3)^2, \qquad s_{23} = M^2 + m_1^2 + m_2^2 + m_3^2 - s_{12} - s_{13}.
```

Daughter order per channel:

| channel | (1, 2, 3) | $s_{12}$ | $s_{13}$ |
| --- | --- | --- | --- |
| `d0`, `b0`, `bs` | $(K_S^0, \pi^+, \pi^-)$ | $m^2(K_S^0\pi^+)$ | $m^2(K_S^0\pi^-)$ |
| `b0_3pi` | $(\pi^0, \pi^+, \pi^-)$ | $m^2(\pi^0\pi^+)$ | $m^2(\pi^0\pi^-)$ |

Square-Dalitz coordinates $(m', \theta') \in [0, 1]^2$ are built on the (23) pair:

```math
m' = \frac{1}{\pi}\arccos\Big(2\,\frac{m_{23} - (m_2 + m_3)}{M - m_1 - (m_2 + m_3)} - 1\Big), \qquad
\theta' = \frac{\theta_{12}}{\pi},
```

with $\theta_{12}$ the angle between daughters 1 and 2 in the (23) rest frame (`FinalState.to_square`).

## Amplitudes and CP conjugation

```math
A(s_{12}, s_{13}) = \sum_r c_r F_r(s_{12}, s_{13}), \qquad \bar A(s_{12}, s_{13}) = \sum_r \bar c_r \bar F_r(s_{12}, s_{13}).
```

* $A$ is the amplitude for $P^0 \to f$, $\bar A$ for $\bar P^0 \to f$, at the same Dalitz point.
* In every channel CP exchanges daughters 2 and 3, so $\bar F_r(s_{12}, s_{13}) = F_r(s_{13}, s_{12})$: the CP-conjugate shape is the same function at the mirrored point.
* Each $F_r$ is normalized as in Laura++: $\int |F_r|^2\, ds_{12}\, ds_{13} = 1$.
* `Coefficient(a, delta, phi)` gives $c = a\,e^{i(\delta + \phi)}$ and $\bar c = a\,e^{i(\delta - \phi)}$: $\delta$ is the strong (CP-even) phase and $\phi$ the weak (CP-odd) phase.
* `Coefficient.from_complex(c, c_bar)` sets $c$ and $\bar c$ independently, allowing $|\bar c| \neq |c|$.

## Mixing

```math
|P_L\rangle = p\,|P^0\rangle + q\,|\bar P^0\rangle, \qquad |P_H\rangle = p\,|P^0\rangle - q\,|\bar P^0\rangle,
```

```math
x = \frac{m_H - m_L}{\Gamma}, \qquad y = \frac{\Gamma_L - \Gamma_H}{2\Gamma}.
```

Relation to published conventions:

* $B^0$, $B_s^0$: this is the PDG convention; $x$, $y$ and $q/p$ enter directly.
  The Standard Model values are $q/p = e^{-2i\beta}$ for $B^0$ and $q/p = e^{-i\phi_s}$, with $\phi_s = -2\beta_s$, for $B_s^0$.
* $D^0$: HFLAV defines $D_{1,2} = p\,D^0 \pm q\,\bar D^0$ and $y = (\Gamma_2 - \Gamma_1)/2\Gamma$, so **$y = -y_\text{HFLAV}$** while $x = x_\text{HFLAV}$.
  The measured $y_\text{HFLAV} = +0.645\%$ enters as `y = -0.00645`.

## Time dependence and tags

With

```math
g_+(\tau) = \cosh\frac{(y - ix)\tau}{2}, \qquad g_-(\tau) = -\sinh\frac{(y - ix)\tau}{2},
```

the tagged rates at a Dalitz point are

```math
\mathcal R_{+}(\tau) = \Big|A\,g_+ + \frac{q}{p}\,\bar A\,g_-\Big|^2, \qquad
\mathcal R_{-}(\tau) = \Big|\bar A\,g_+ + \frac{p}{q}\,A\,g_-\Big|^2.
```

There are two time variables:

| mode | time variable | density | `tag` |
| --- | --- | --- | --- |
| single meson (default; `d0`, `b0`, `bs`) | decay time since production, $\tau \geq 0$ | $e^{-\tau}\,\mathcal R_{\text{tag}}(\tau)$ | flavour at production |
| coherent pair, `coherent=True` (`b0_3pi`) | $\Delta t = \tau_\text{signal} - \tau_\text{tag side}$, $-\infty < \Delta t < \infty$ | $e^{-\lvert\Delta t\rvert}\,\mathcal R_{\text{tag}}(\Delta t)$ | flavour of the signal meson at $\Delta t = 0$ |

In the coherent mode the pair is produced in a $C$-odd state, so at $\Delta t = 0$ the signal meson has the flavour opposite to the tag-side meson. `tag = +1` therefore means the tag-side meson decayed as a $\bar B^0$. In the notation of BaBar and Belle, where $Q_\text{tag} = +1$ for a tag-side $B^0$, **`tag` $= -Q_\text{tag}$**.

Each tag is normalized separately: a sample has a fixed number of events per tag, and total-yield differences between the tags are not modelled.

## C, S and the tag asymmetry

For $y = 0$ and $|q/p| = 1$, with $\lambda = (q/p)\bar A/A$:

```math
C = \frac{1 - |\lambda|^2}{1 + |\lambda|^2}, \qquad S = \frac{2\,\mathrm{Im}\,\lambda}{1 + |\lambda|^2}, \qquad
\mathcal R_\pm \propto 1 \pm C\cos x\tau \mp S\sin x\tau.
```

* The tag asymmetry used in the figures is $(N_+ - N_-)/(N_+ + N_-) = C\cos x\tau - S\sin x\tau$, i.e. **minus** the conventional $\mathcal A_{CP}(t)$ of $B$-factory papers.
* With $q/p = e^{-2i\beta}$, a CP eigenstate with $\bar F = \eta F$ has $S = -\eta \sin 2\beta$ (the Standard Model $S_f = -\eta_f \sin 2\beta$).

## Detector effects

* Time resolution: $\tau \to \tau + \epsilon$, $\epsilon \sim \mathcal N(0, \sigma_\tau)$; negative values are kept.
* Mistag: each `tag` is flipped with probability $\omega$; `true_tag` keeps the value before flipping.
