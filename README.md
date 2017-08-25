# panelNNET
Semiparametric panel data models using neural networks

TBD:

1.  Update manpages

2.  Need a method for including groups that aren't necessarily represented by fixed effects in estimating cluster vcv

3.  Optimization/rewriting in low-level languages

4.  Look into GPU libraries

5.  Verbose output should give information about mean and sd of activations, to monitor saturation

6.  Add sanity checks:  that FE input is defined and is a factor, that the parametric penalty multiplier is the same length as the parametric term vector, etc.

7.  Add effective degrees of freedom to summary output

8.  Write multistart function

9.  Build interactive mode, using the keypress package

10.  Write a function to predict from a new dataset that has missing observations.  As arguments it should take distributions from which to draw the missing values.

11.  Code up permutaiton importance function

12.  Write a vignette

13.  Remove the doscale and bias_hlayers arguments -- they should always be on

14.  Save activations as functions, rather than strings/pointers, then remove all of the redundant headers in the various files
