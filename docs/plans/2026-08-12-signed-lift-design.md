# Signed LIFT Design

## Goal

Test whether bounded signed additive features improve LIFT's compressed old-history
branch without changing its exact recent-pair branch, forget gates, relative bias,
ReZero tail gain, or trainable parameter inventory.

## Selected Design

Keep the existing normalizations `local_forgetting_rel_bias` and
`hybrid_forgetting_rel_bias`. Add a fixed `hybrid_tail_feature_map` selector with
values `identity`, `tanh`, and `abs_tanh`; its default is `identity`, preserving
existing behavior and checkpoints. Reuse the fixed Python float
`signed_feature_gamma`; neither setting is a parameter or buffer.

For head `h`, let

```text
F_ij = exp(sum_{r=j+1}^i log f_r)
O_i^W = {j : 0 <= j <= i-W}
```

and define the old branch as

```text
z_old[i] = (1 / (2N)) sum_{j in O_i^W}
             F_ij <phi_q(q_i), phi_k(k_j)> v_j.
```

The local branch is unchanged:

```text
z_local[i] = (1 / N) sum_{0 <= i-j < W}
               F_ij SiLU(q_i^T k_j + b_ij) v_j.
```

The branches are disjoint at distances `W-1` and `W`. Their combination remains
`z_local + alpha_h z_old`, with the existing
`alpha_h = 2 tanh(rho_h / 2)`. Identity is the existing LIFT tail. The signed map
uses coordinatewise `tanh(gamma x)`, and the orthant-folded control uses
`abs(tanh(gamma x))`.

## Exact Recurrence

For `t = i-W`, maintain

```text
A_t = f_t A_{t-1} + phi_k(k_t) v_t^T
R_t = F_{t+W,t}
z_old[t+W] = (1 / (2N)) A_t^T [R_t phi_q(q_{t+W})].
```

Survival multiplies the query feature after the nonlinear map. In particular,
`R phi(q)` is correct and `phi(R q)` is not. The recurrence supports the model's
causal mask with right padding; it cannot represent arbitrary pair-mask holes.
The initial training operator remains the quadratic quality oracle because the
unchanged local branch is still dense. No end-to-end linear-complexity claim is
made until the banded-plus-scan kernel is qualified.

## Capacity Contract

The feature selector and gamma add no trainables. Each LIFT layer retains
`H*dq + H + H` gate/gain parameters, giving exact named inventories of 313,432
parameters on MovieLens-1M and 38,917,472 on MovieLens-20M. All feature arms must
match the existing local and identity LIFT name, shape, dtype, `requires_grad`,
and buffer inventories.

## Experiment

Run a same-snapshot MovieLens-1M array with local W32, identity LIFT, tanh LIFT,
and abs-tanh LIFT under seeds 42-44. This reruns the controls so feature-geometry
differences cannot be attributed to source drift. The primary statistic remains
mean NDCG@10 over epochs 96-100. A tail survives only if its paired improvement
over local averages at least 0.002, is positive in at least two seeds, and is
never below -0.001. Evidence for signed geometry additionally requires tanh to
beat abs-tanh in at least two seeds.

