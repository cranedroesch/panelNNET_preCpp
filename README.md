# panelNNET
Semiparametric panel data models using neural networks

TBD:

1.  Implement methods for heterogeneous causal effects.  This is simply an interaction between the topmost derived variables and a treatment effect.  This has been started.  The following things need attention:
  --prediction function
  --vcov and se.fit on predict
  --summary function
  --man pages

2.  Need a method for including groups that aren't necessarily represented by fixed effects in estimating cluster vcv

3.  Remove duplicate Jacobians inside $vc.  They waste space.

3.  Optimization/rewriting in low-level languages


