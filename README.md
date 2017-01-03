# panelNNET
Semiparametric panel data models using neural networks

TBD:

1.  Update manpages

2.  Need a method for including groups that aren't necessarily represented by fixed effects in estimating cluster vcv

3.  Remove duplicate Jacobians inside $vc.  They waste space.

4.  Optimization/rewriting in low-level languages

6.  Need approaches for dealing with local minima, especially in dealing with HTEs.  This could be attempting random jitter when a local minimum is reached, and then reverting to the previous minimum if that doesn't offer any improvement.

7.  Implement minibatch gradient descent as a training option

8.  Write a function that steps through values of lambda
