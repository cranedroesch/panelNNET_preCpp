Jacobian.panelNNET <-
function(obj, ...){
  if (obj$activation == 'tanh'){
    sigma <- tanh
    sigma_prime <- tanh_prime
  }
  if (obj$activation == 'logistic'){
    sigma <- logistic
    sigma_prime <- logistic_prime
  }
  if (obj$activation == 'relu'){
    sigma <- relu
    sigma_prime <- relu_prime
  }
  if (obj$activation == 'lrelu'){
    sigma <- lrelu
    sigma_prime <- lrelu_prime
  }
  plist <- as.relistable(obj$parlist)
  pvec <- unlist(plist)
  #define function to pass to `jacobian` from `numDeriv`
  Jfun <- function(pvec, obj){
    parlist <- relist(pvec)
    D <- obj$X
    for (i in 1:length(obj$hidden_units)){
      if (obj$used_bias == TRUE){D <- cbind(1,D)}
      D <- sigma(D %*% parlist[[i]])
    } 
    colnames(D) <- paste0('nodes',1:ncol(D))
    if (!is.null(obj$treatment)){
      #Add treatment interactions
      if (obj$interact_treatment == TRUE){
        ints <- sweep(D, 1, obj$treatment, '*')
        colnames(ints) <- paste0('TrInts',1:ncol(ints))
        D <- cbind(ints, D)
      }
      #Add treatment dummy
      D <- cbind(obj$treatment, D)
      colnames(D)[1] <- 'treatment'
    }
    if (!is.null(obj$param)){
      D <- cbind(obj$param, D)
      colnames(D)[1:ncol(obj$param)] <- paste0('param',1:ncol(obj$param))
    }
    if (is.null(obj$fe_var)){D <- cbind(1, D)}#add intercept if no FEs
    if (!is.null(obj$fe_var)){
      Zdm <- demeanlist(D, list(obj$fe_var))
      ydm <<- demeanlist(obj$y, list(obj$fe_var))
      fe <- (obj$y-ydm) - as.matrix(D-Zdm) %*% c(parlist$beta_param, parlist$beta_treatment, parlist$beta_treatmentinteractions, parlist$beta)
      yhat <- D %*% c(parlist$beta_param, parlist$beta_treatment, parlist$beta_treatmentinteractions, parlist$beta) + fe    
    } else {
      yhat <- D %*% c(parlist$beta_param, parlist$beta_treatment, parlist$beta_treatmentinteractions, parlist$beta)
    }
    return(yhat)
  }
  #pass `Jfun` to `jacobian` from `numDeriv`
  J <- jacobian(Jfun, pvec, obj = obj)
  #drop any zero columns that represent lower-level parameters
  dJ <- ncol(J)
  tokeep <- which(!(apply(J, 2, function(x){all(x==0)}) & !grepl('treatment|param', names(pvec))))
  J <- J[,tokeep]
  if (ncol(J) < dJ){
    warning(paste0(dJ - ncol(J), ' columns dropped from Jacobian because dY/dParm =~ 0'))
  }
  pvec <- pvec[tokeep]
  #re-order jacobian so that parametric terms are on the front, followed by top layer.
  J <- J[,c(
      which(grepl('param', names(pvec)))
    , which(names(pvec) == 'beta_treatment')
    , which(grepl('beta_treatmentinteractions', names(pvec)))
    , which(grepl('beta', names(pvec)) & !grepl('param', names(pvec)) & !grepl('treatment', names(pvec)))
    , which(!grepl('beta', names(pvec)))#no particular order to lower-level parameters
   )]
  return(J)
}
