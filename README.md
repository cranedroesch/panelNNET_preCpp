# panelNNET
Semiparametric panel data models using neural networks

TBD:

1.  Update manpages

2.  Need a method for including groups that aren't necessarily represented by fixed effects in estimating cluster vcv

3.  Optimization/rewriting in low-level languages

4.  Look into GPU libraries

5.  Add sanity checks:  that FE input is defined and is a factor, that the parametric penalty multiplier is the same length as the parametric term vector, etc.

6.  Add effective degrees of freedom to summary output

7.  Build interactive mode, using the keypress package

8.  Write a function to predict from a new dataset that has missing observations.  As arguments it should take distributions from which to draw the missing values.

9.  Code up permutation importance function

10.  Write a vignette

11.  Save activations as functions, rather than strings/pointers, then remove all of the redundant headers in the various files

12.  Remove storage of hidden layers to degree possible, to reduce memory footprint.

13.  Reduce number of things in the output, perhaps subject to an argument.  Goal is to reduce storage footprint and loading time.  This will involve not storing the input data, but storing the scaling factors from the input data.

14.  Create infrastructure to bag (bootstrap aggregate) neural nets, including creating objects that can predict from bagged NNs ensembles.  



