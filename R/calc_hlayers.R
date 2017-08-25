
calc_hlayers <- function(parlist, X = X, param = param, fe_var = fe_var, nlayers = nlayers, convolutional = convolutional, activ = activ){
  hlayers <- vector('list', nlayers)
  for (i in 1:(nlayers + !is.null(convolutional))){
    if (i == 1){D <- X} else {D <- hlayers[[i-1]]}
    if (bias_hlayers == TRUE){D <- cbind(1, D)}
    # make sure that the time-invariant variables pass through the convolutional layer without being activated
    if (is.null(convolutional) | i > 1){
      hlayers[[i]] <- activ(D %*% parlist[[i]])        
    } else {
      HL <- D %*% parlist[[i]]
      HL[,1:(N_TV_layers * convolutional$Nconv)] <- activ(HL[,1:(N_TV_layers * convolutional$Nconv)])
      hlayers[[i]] <- HL
    }
  }
  colnames(hlayers[[i]]) <- paste0('nodes',1:ncol(hlayers[[i]]))
  if (!is.null(param)){#Add parametric terms to top layer
    hlayers[[i]] <- cbind(param, hlayers[[i]])
    colnames(hlayers[[i]])[1:ncol(param)] <- paste0('param',1:ncol(param))
  }
  if (is.null(fe_var)){#add intercept if no FEs
    hlayers[[i]] <- cbind(1, hlayers[[i]])
  }
  return(hlayers)
}