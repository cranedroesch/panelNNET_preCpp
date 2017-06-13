panelNNET.default <-
function(y, X, hidden_units, fe_var
  , maxit = 100, lam = 0, time_var = NULL, param = NULL
  , parapen = rep(0, ncol(param)), parlist = NULL, verbose = FALSE
  , para_plot = FALSE, report_interval = 100
  , gravity = 1.01, convtol = 1e-8, bias_hlayers = TRUE, RMSprop = TRUE, start_LR = .01
  , activation = 'tanh', doscale = TRUE
  , treatment = NULL, interact_treatment = TRUE, batchsize = nrow(X)
  , maxstopcounter = 10, OLStrick = FALSE, initialization = 'enforce_normalization'
  , dropout_hidden = 1, dropout_input = 1, test_set = NULL, ...)
{
  out <- panelNNET.est(y, X, hidden_units, fe_var, maxit, lam
    , time_var, param, parapen, parlist, verbose, para_plot
    , report_interval, gravity, convtol, bias_hlayers, RMSprop
    , start_LR, activation, doscale 
    , treatment, interact_treatment, batchsize, maxstopcounter
    , OLStrick, initialization, dropout_hidden, dropout_input
    , test_set
  )
  out$call = match.call()
  class(out) <- 'panelNNET'
  out
}
