# panelNNET
Semiparametric panel data models using neural networks

TBD:

1.  Update manpages

2.  Need a method for including groups that aren't necessarily represented by fixed effects in estimating cluster vcv

3.  Remove duplicate Jacobians inside $vc.  They waste space.

4.  Optimization/rewriting in low-level languages

5.  There is a bug somewhere in the MSE calculation.  When you fit a network, and then fit a new network starting from the same weights, the MSE of the new network with the same weights is not identical.  This complicates one network starting where another left off.  Get to the bottom of this.
