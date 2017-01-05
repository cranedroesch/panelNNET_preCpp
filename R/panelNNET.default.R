panelNNET.default <-
function(y, X, hidden_units, fe_var
  , maxit = 100, lam = 0, time_var = NULL, param = NULL
  , parapen = rep(0, ncol(param)), parlist = NULL, verbose = FALSE
  , save_each_iter = FALSE, path = NULL, tag = "", gravity = 1.01
  , convtol = 1e-8, bias_hlayers = TRUE, RMSprop = TRUE, start_LR = .01
  , activation = 'tanh', inference = TRUE, doscale = TRUE
  , treatment = NULL, interact_treatment = TRUE, batchsize = nrow(X), ...)
{
  out <- panelNNET.est(y, X, hidden_units, fe_var, maxit, lam
    , time_var, param, parapen, parlist, verbose, save_each_iter
    , path, tag, gravity, convtol, bias_hlayers, RMSprop
    , start_LR, activation, inference, doscale 
    , treatment, interact_treatment, batchsize
  )
  out$call = match.call()
  class(out) <- 'panelNNET'
  out
}
