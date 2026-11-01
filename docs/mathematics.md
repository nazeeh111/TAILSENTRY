# Estimator assumptions and derivation

The full derivation, propositions, proofs, known-truth references and negative experiments are in [the manuscript](../paper/tailsentry.pdf). The results are an implementation of established ideas, not a verified new estimator.

Let `X=G/sqrt(S/nu)`, with independent standard Gaussian `G` and chi-square `S`; omit the scale for Gaussian factors. Student identity **scale** is not identity covariance. Supported loss is the maximum of finitely many symmetric quadratic forms plus linear terms and constants.

`build_cover` encloses the exceedance event, up to target-null sets, in `C = union H_k`, where each `H_k` is a normalized linear stress constraint. Negative eigenspaces are discarded to form an upper envelope. Caps sum to threshold minus constant. If every positive term is below its cap, the loss cannot exceed the threshold. A spherical compact region has a tighter supporting projection. Extremely remote cutoffs are relaxed downward to preserve representable proposal support. Relaxation increases the cover, never removes a mode.

Write `B=sum P(H_k)` and `M=sum 1(H_k)`. Choose a component with probability `P(H_k)/B`, then sample the exact target conditional on that event. The joint proposal is `q=f*M/B`. The unnormalized contribution `B*1(loss>threshold)/M` is unbiased and bounded by B. On a complete halfspace event this is ALOE.

For Student factors, draw axial `T` from a truncated t-nu distribution, then `S|T = chi-square(nu+1)/(1+T^2/nu)`, and an independent orthogonal Gaussian vector. Leaving S unconditional would be incorrect.

Choose a unit Gaussian direction e independently of confirmation. For each projected vector V and scale S, integrate **every** quadratic exceedance interval along e. Let U be its conditional failure probability and `D=sum P(H_k|V,S)`. Marginal proposal density is `f(V,S)*D/B`, so contribution is `Y=B*U/D`. Define it as zero on D=0, a proposal-null region. Since the event is covered, U<=D and Y<=B. It is the conditional expectation of the raw weighted contribution, so its variance cannot increase at identical fixed proposal/sample count. This theorem is not a guarantee of better runtime or of superiority over conditional Sobol.

Independent frozen confirmation supports fixed-budget empirical Bernstein bounds, with radius `sqrt(2*s^2*log(4/delta)/n) + 7*B*log(4/delta)/(3*(n-1))`. The interval is intersected with [0,1]; empty intersections never certify precision. The displayed mean must satisfy the requested relative-error bound against both interval endpoints. Prefix traces do not justify repeated stopping.

Severity integrates the maximum branch envelope, not the sum of branch losses. Short finite pieces use stabilized16-point Gauss–Legendre quadrature. No floating-point result is claimed to be interval-certified exact arithmetic. The adversarial review records residual error for extremely narrow remote islands. Probability range bounds do not bound unbounded severity. Student positive quadratic means require df>2, and usual second-moment SEs generally df>4. Unsupported mean requests are rejected and unsupported SEs omitted. VaR inversion, ES plug-ins and fixed-threshold tail means are finite-sample biased.

An unknown black-box region cannot be cheaply ruled out by high observed effective sample size. With zero hits in n independent nominal audit draws, its one-sided upper mass bound is `1-delta^(1/n)`. The structural method replaces that generic audit with a supported-model proof; it does not discover all unknown real-market mechanisms.
