Jacobian.panelNNET <-
function(obj){
  if (obj$activation == 'tanh'){
    sigma <- tanh
    sigma_prime <- tanh_prime
    sigma_prime2 <- tanh_prime2
  }
  if (obj$activation == 'logistic'){
    sigma <- logistic
    sigma_prime <- logistic_prime
    sigma_prime2 <- logistic_prime2
  }
  plist <- as.relistable(obj$parlist)
  pvec <- unlist(plist)
  Jfun <- function(pvec, obj){
    parlist <- relist(pvec)
    nlayers <- length(obj$hidden_units)
    hlayers <- vector('list', nlayers)
    for (i in 1:nlayers){
      if (i == 1){D <- obj$X} else {D <- hlayers[[i-1]]}
      if (obj$used_bias == TRUE){D <- cbind(1, D)}
      hlayers[[i]] <- sigma(D %*% parlist[[i]])
    }
    if (!is.null(obj$param)){hlayers[[i]] <- cbind(obj$param, hlayers[[i]])}
    if (is.null(obj$fe_var)){hlayers[[i]] <- cbind(1, hlayers[[i]])}#add intercept if no FEs
    if (!is.null(obj$fe_var)){
      Zdm <- demeanlist(hlayers[[i]], list(obj$fe_var))
      ydm <<- demeanlist(obj$y, list(obj$fe_var))
      fe <- (obj$y-ydm) - as.matrix(hlayers[[i]]-Zdm) %*% as.matrix(c(parlist$beta_param, parlist$beta))
      yhat <- hlayers[[i]] %*% c(parlist$beta_param, parlist$beta) + fe    
    } else {
      yhat <- hlayers[[i]] %*% c(parlist$beta_param, parlist$beta)    
    }
    return(yhat)
  }
  J <- jacobian(Jfun, pvec, obj = obj)
  J <- J[,c(#re-order jacobian so that parametric terms are on the front, followed by top layer.
      which(grepl('param', names(pvec)))
    , which(grepl('beta', names(pvec)) & !grepl('param', names(pvec)))
    , which(!grepl('beta', names(pvec)))#no particular order to lower-level parameters
   )]
  return(J)
}
