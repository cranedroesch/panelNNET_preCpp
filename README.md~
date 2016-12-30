# panelNNET
Semiparametric panel data models using neural networks

TBD:

1.  Implement methods for heterogeneous causal effects.  This is simply an interaction between the topmost derived variables and a treatment effect.  This has been started.  The following things need attention:
  --man pages

2.  Need a method for including groups that aren't necessarily represented by fixed effects in estimating cluster vcv

3.  Remove duplicate Jacobians inside $vc.  They waste space.

4.  Optimization/rewriting in low-level languages

5.  In cross-validation function when dealing with fixed effects, FE's are estimated off of the unpenalized model.  This will give the wrong fixed effects.  Will need to estimate them longhand rather than relying on felm and getfe from lfe

6.  Try to make CV work with Jacobian rather than with top layer.

7.  Need approaches for dealing with local minima, especially in dealing with HTEs.  This could be attempting random jitter when a local minimum is reached, and then reverting to the previous minimum if that doesn't offer any improvement.
