# Independent mathematical and numerical review

Reviewer role: benchmark/finance specialist, separate from the author of
`conditional.py`, `engine.py`, `models.py`, `uncertainty.py` and `envelope.py`.
This is an independent AI code review within the same research session, not
external peer review, human review or independent adoption. The reviewer wrote
`references.py` and the data-preparation utility; those are not claimed as
independently reviewed here.

The review started from raw implementation files, not an author-supplied claim
of improvement. It attempted counterexamples, built independent quadrature and
density checks, and inspected the development experiment artifacts. No final
holdout result is endorsed by this document unless explicitly added below.

## Estimator derivation checked

For a frozen defensive Gaussian mixture q(z), projection onto the complement of
a unit vector e has target marginal f_V and proposal marginal q_V. Integrating
the event over the target's remaining independent standard-normal coordinate
gives U(v)=P_f(E | V=v). The implemented observation is

    Y(v) = f_V(v) U(v) / q_V(v).

Consequently E_q Y=P_f(E). This is the conditional expectation, under q, of the
ordinary likelihood-corrected indicator, so its variance cannot exceed that of
the unconditioned observation with the same frozen proposal. It does not imply
better error per CPU second, better performance after pilot overhead, or better
performance than another proposal. For q at least epsilon times f, the projected
ratio is at most 1/epsilon. The lower-dimensional ratio in `log_ratio` agrees
with direct multivariate-normal densities in an independently constructed
orthonormal basis.

For the newer envelope proposal, let H_j be its covering halfspaces, let
B=sum_j P_f(H_j), and m(z)=sum_j 1_Hj(z). Choosing j with mass P_f(H_j)/B and
sampling f conditional on H_j gives q(z)=f(z)m(z)/B. Provided E is contained in
the union up to target-null sets, the raw observation B 1_E/m is unbiased.

After projection, r_j(v)=P_f(H_j | V=v), hence

    q_V(v) = f_V(v) sum_j r_j(v) / B,
    Y(v)   = B U(v) / sum_j r_j(v).

The bound 0<=Y<=B follows from U(v)<=P_f(union_j H_j | V=v)<=sum_j r_j(v).
This explains why a small, mathematically valid cover mass can yield useful
bounded-observation intervals where a generic defensive bound is too large.
The containment proof is essential; numerical approximations must not silently
drop a cover component.

For a general symmetric quadratic, dropping negative eigenvalue terms is an
upper bound. Allocating nonnegative caps summing to threshold minus constant
across the linear term and all positive squared projection terms proves the
pigeonhole cover. This does not require independence of these projections.
The spherical shortcut is valid only for a genuinely spherical quadratic.

The Student sampler was independently derived: write X=G sqrt(nu/S), with
G standard normal and S chi-square(nu). If T is its projection onto a unit
normal, then T is Student(nu), and

    S | T=t = chi-square(nu+1) / (1+t^2/nu).

The orthogonal coordinates of G remain independent standard normals. This
matches `sample_cover`. Conditionalizing a factor direction while preserving
the auxiliary S coordinate makes the remaining event exactly quadratic in the
integrated standard-normal coordinate. Numerical distribution checks include
the truncated Student survival transform and orthogonal-factor independence.

## Integration, independence and uncertainty

The event integrator retains bounded intervals and both unbounded tails, then
unions overlaps. The severity integrator partitions at threshold crossings and
all pairwise quadratic crossings, selects the actual upper envelope, and
integrates its first three Gaussian moments. Independent tests use companion
matrix roots and adaptive quadrature of the original loss, with an algebraic
special case for exact double roots. This avoids calling the implementation's
root solver or interval builder as its own oracle.

Separate `SeedSequence` children isolate pilot, selection and confirmation
randomness. Consuming an extra 10,000 pilot random draws after training leaves
the confirmation observations byte-identical in a regression test. Conditional
on the pilot, the final proposal/direction is fixed, so ordinary independent
bounded-observation inference applies. Pilot observations are not included in
the confirmation mean. This does not authorize repeated testing with an
unchanged fixed-budget interval until a desired precision status appears.

The empirical Bernstein radius matches a two-sided application of
[Maurer and Pontil, Theorem 4](https://arxiv.org/html/0907.3740v1): scale its
bounded variable to [0,1], use the unbiased sample variance, and allocate half
the error probability to each direction. This gives

    sqrt(2 sample_variance log(4/alpha)/n)
        + 7 bound log(4/alpha)/(3(n-1)).

The guarantee concerns the ideal probability model and correct arithmetic. It
does not certify float64 rounding, the financial model, or the historical fit.
The normal interval returned separately has no comparable finite-sample
guarantee and can fail badly for nearly constant observations with rare
corrections. ESS is an observed-weight diagnostic, not an unseen-mode proof.

The fixed-threshold first-moment estimate is distinct from the tail-mean ratio.
The ratio is finite-sample biased and is ES only when its threshold is the
appropriate population VaR. Student moment-existence gates now distinguish
finite means from supported standard errors; conservative gates are acceptable.
No finite-sample severity interval or unbiased finite-sample VaR is established
by this review.

## Counterexamples found and corrections checked

1. An approximate symmetry check accepted B=[[0,10^6],[10^6+1,0]], but the
   conditional formula assumed exact symmetry. A direct loss and conditional
   polynomial differed by one at a simple point. The author changed the guard
   to zero relative tolerance and symmetrized accepted matrices.
2. Unscaled discriminants overflowed or underflowed for the polynomial
   s(t^2+t-1), s=10^200 or 10^-200, changing its probability. Normalization in
   the root solver and interval classification repaired scale invariance.
3. An empty [lower>upper] confidence intersection received a positive precision
   status because negative width passed the criterion. The status now rejects
   inconsistent intervals.
4. Spherical detection with an absolute tolerance incorrectly declared a
   nonempty scaled ellipsoid empty. B=diag(-10^-14,-4*10^-14), a=(8*10^-14,0),
   c=-15*10^-14 has positive loss at (4,0). Exact spherical detection and local
   coefficient normalization repaired this example and finite rescalings of a
   halfspace.
5. Near-equal cover constraints were deduplicated as though equal, and nonzero
   projected slopes below 10^-14 were treated as zero. These approximations
   could invalidate exact support or marginal weighting. Both now use exact
   comparisons; a boundary projection with slope 10^-15 is tested.
6. Checking only total cover mass allowed a component's normal tail mass to
   underflow to zero while other components remained. The implementation now
   explicitly rejects any zero or nonfinite component mass.
7. Positive-quadratic Student losses at nu<=2 returned finite sample severity
   as though a population tail mean existed. The author added conservative
   moment gates. For nu=3 the mean is supported but the generic quadratic
   severity variance is not, and a standard error must be omitted.
8. The envelope API initially omitted confidence-input validation. Independent
   cases require rejection of alpha outside (0,1), including NaN.

These changes were made by the source author after the reviewer supplied
counterexamples. The reviewer did not edit those implementation files.

## Benchmark skepticism

The development snapshot `results/development/pilot-v1` is tuning evidence,
with only 12 runs per condition. It is not final confirmation. Several raw
observations challenge an unqualified breakthrough claim:

- On the compact-ball development condition, raw cover sampling has lower
  observed relative RMSE than projected sampling. Projection/pilot overhead
  therefore cannot be assumed to improve practical efficiency.
- On the delta-gamma development condition, conditional Sobol has lower
  observed relative RMSE than the adaptive projected candidate at the same
  oracle-call budget. A simple established alternative can win.
- On the paired Gaussian condition, raw envelope sampling is already close to
  a zero-variance estimator. Its large improvement over crude MC primarily
  reflects established union sampling and access to analytic halfspaces.
- Several normal-interval coverage rates in that snapshot are poor even for
  envelope methods. The bounded intervals must remain the primary supported
  uncertainty statement, with widths reported alongside coverage.

The finite-sample cover bound is useful evidence only if its mathematical cover
is valid. Giving analytic coefficients and exact line integration to one method
but only point samples to a comparator is a different information budget.
Report that distinction, full elapsed cost, pilot/optimization work, the raw
cover ablation and conditional baselines. A final holdout should not be selected
or retuned using the development failures listed here.

## Scope and remaining limits

The reviewed mechanism is a coherent combination of established union
importance sampling, conditional Monte Carlo and bounded-observation inference.
This review establishes neither research novelty nor superiority over the best
existing rare-event algorithms. The strongest supported accomplishment is an
inspectable implementation and falsifiable experiment, subject to the recorded
checks. Generic black-box losses, unknown densities, unrestricted distributions
and interval-certified numerical arithmetic are outside the implementation.

No historical financial forecast result, manuscript citation collection or UI
is independently validated by this mathematical review. A separate financial
check must respect the frozen chronological split and keep model uncertainty
distinct from simulation uncertainty.

## Reproduction and current check status

From the repository root:

    python -m pytest tests/test_independent_math.py -q

The final independent suite passed **42 checks in 15.08 seconds**, including
Gaussian and Student sampler identities, independent envelope quadrature,
exact probability cases, pilot independence, cover containment, numerical
regressions, invalid confidence and Student severity-variance gates. The result
is recorded in `reviews/independent-math-checks.log`. The accompanying
`reviews/mathematical-review-source-hashes.json` record pins
the implementation bytes examined at completion of this review; later source
changes require proportionate rechecking.
