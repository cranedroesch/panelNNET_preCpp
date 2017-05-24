


do_inference <- function(obj, numerical = FALSE, parallel = TRUE
  , step = 1e-9, J = NULL, verbose = FALSE, OLS_only = FALSE){
  #Compute Jacobian if not supplied and asked-for
  if (is.null(J) & OLS_only == FALSE){
    J <- Jacobian.panelNNET(obj, parallel, step, numerical)
  }
  #top layer for OLS approximation
  X <- obj$hidden_layers[[length(obj$hidden_layers)]]
  #empty list of vcovs
  vcs <- list()
  #calculate Jacobian-based vcovs
  if (OLS_only == FALSE){
    vcs[["vc.JacHomo"]] <- tryCatch(vcov.panelNNET(obj, 'Jacobian_homoskedastic', J = J), error = function(e)e, finally = NULL)
    vcs[["vc.JacSand"]] <- tryCatch(vcov.panelNNET(obj, 'Jacobian_sandwich', J = J), error = function(e)e, finally = NULL)    
    if (!is.null(obj$fe_var)){
      vcs[["vc.JacClus"]] <- tryCatch(vcov.panelNNET(obj, 'Jacobian_cluster', J = J), error = function(e)e, finally = NULL)
    }
  } 
  #calculate OLS aproximations
  vcs[["vc.OLSHomo"]] <- tryCatch(vcov.panelNNET(obj, 'OLS', J = X), error = function(e)e, finally = NULL)
  vcs[["vc.OLSSand"]] <- tryCatch(vcov.panelNNET(obj, 'sandwich', J = X), error = function(e)e, finally = NULL)
  if (!is.null(obj$fe_var)){
    vcs[["vc.OLSClus"]] <- tryCatch(vcov.panelNNET(obj, 'cluster', J = X), error = function(e)e, finally = NULL)
  }
  obj$vcs <- vcs
  #calculate EDF and add to output
  if (OLS_only == FALSE){
    obj$J <- J
    jtj <- crossprod(J)
    ev <- eigen(jtj)$values
    D <- rep(obj$lam, ncol(J))
    if (is.null(obj$fe_var)){
      pp <- c(0, obj$parapen) #never penalize the intercept
    } else {
      pp <- obj$parapen #parapen
    }
    if (!is.null(obj$treatment)){pp <- append(pp, 0)}#treatment always follows parametric terms and will not be penalized
    D[1:length(pp)] <- D[1:length(pp)]*pp #incorporate parapen into diagonal of covmat
    obj$edf <- sum(ev/(ev+D))
  }
  return(obj)
}



