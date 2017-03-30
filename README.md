# panelNNET
Semiparametric panel data models using neural networks

TBD:

1.  Update manpages

2.  Need a method for including groups that aren't necessarily represented by fixed effects in estimating cluster vcv

3.  Remove duplicate Jacobians inside $vc.  They waste space.

4.  Optimization/rewriting in low-level languages

5.  Implement minibatch with the optim option

6.  fix plot method for when optim is selected

6.  Look into GPU libraries

7.  Implement convolutional neural nets with weight sharing

8.  Verbose output should give information about mean and sd of activations, to monitor saturation

9.  Add option to include validation set, tracked in terms of MSE when verbose set to TRUE

10.  VCV should account for effective degrees of freedom, not raw DF

11.  Add sanity checks:  that FE input is defined and is a factor, that the parametric penalty multiplier is the same length as the parametric term vector, etc.
