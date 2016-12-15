predict.panelNNET <-
function(obj, newX = NULL, fe.newX = NULL, new.param = NULL){
  if (obj$activation == 'tanh'){
    sigma <- tanh
  }
  if (obj$activation == 'logistic'){
    sigma <- logistic
  }
  if (is.null(newX)){
    return(obj$yhat)
  } else {
    if (!all(unique(fe.newX) %in% unique(obj$fe$fe_var))){
      stop('New data has cross-sectional units not observed in training data')
    }
    #Scale the new data by the scaling rules in the training data
    if (obj$doscale == TRUE){
      D <- sweep(sweep(newX, 2, STATS = attr(obj$X, "scaled:center"), FUN = '-'), 2, STATS = attr(obj$X, "scaled:scale"), FUN = '/')
      P <- sweep(sweep(new.param, 2, STATS = attr(obj$param, "scaled:center"), FUN = '-'), 2, STATS = attr(obj$param, "scaled:scale"), FUN = '/')
    }
    for (i in 1:length(obj$hidden_units)){
      if (obj$used_bias == TRUE){D <- cbind(1,D)}
      D <- sigma(D %*% obj$parlist[[i]])
    } 
    D <- cbind(P, D)
    if (is.null(obj$fe)){D <- cbind(1, D)
      yhat <- D %*% c(obj$parlist$beta_param, obj$parlist$beta)
    } else {
      xpart <- D %*% c(obj$parlist$beta_param, obj$parlist$beta)
      tm <- foreach(i = 1:length(unique(obj$fe$fe_var)), .combine = rbind) %do% {
        #Because of numerical error, fixed effects within units can sometimes be slightly different.  This averages them.
        c(unique(obj$fe$fe_var)[i], mean(obj$fe$fe[obj$fe$fe_var == unique(obj$fe$fe_var)[i]]))
      }
      colnames(tm) <- c('fe_var','fe')
      nd <- data.frame(fe.newX, xpart, id = 1:length(fe.newX))       
      nd <- merge(nd, tm, by.x = 'fe.newX', by.y = 'fe_var', all.x = TRUE, sort = FALSE)
      nd <- nd[order(nd$id),]
      yhat <- nd$fe + nd$xpart
    }
    return(yhat)
  }
}





