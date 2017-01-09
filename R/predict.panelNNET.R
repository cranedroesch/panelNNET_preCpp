predict.panelNNET <-
function(obj, newX = NULL, fe.newX = NULL, new.param = NULL, new.treatment = NULL, se.fit = FALSE, tauhat = FALSE){
#obj <- m
#newX = Z[e,]
#fe.newX = id[e]
#new.param = matrix(time[e])
#se.fit = TRUE
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
    if (tauhat == TRUE){
      if (is.null(obj$treatment)){
        stop('no treatment in object')
      }
      new.treatment <- rep(1, nrow(newX))
    }
    #Scale the new data by the scaling rules in the training data
    plist <- as.relistable(obj$parlist)
    pvec <- unlist(plist)
    #prepare fe's in advance...
    if (!is.null(obj$fe)){
      FEs_to_merge <- foreach(i = 1:length(unique(obj$fe$fe_var)), .combine = rbind) %do% {
        #Because of numerical error, fixed effects within units can sometimes be slightly different.  This averages them.
        data.frame(unique(obj$fe$fe_var)[i], mean(obj$fe$fe[obj$fe$fe_var == unique(obj$fe$fe_var)[i]]))
      }
      colnames(FEs_to_merge) <- c('fe_var','fe')
    } else {FEs_to_merge <- NULL}
    #(predfun is defined below)
    if (tauhat == TRUE){
      taumat <- predfun(pvec = pvec, obj = obj, newX = newX
        , fe.newX = fe.newX, new.param = new.param, new.treatment, tauhat = TRUE, FEs_to_merge = FEs_to_merge)
      tauhat <- taumat %*% c(obj$parlist$beta_treatment, obj$parlist$beta_treatmentinteractions)
      if (se.fit == TRUE){
        J <- jacobian(predfun, pvec, obj = obj, newX = newX, fe.newX = fe.newX
          , new.param = new.param, new.treatment = new.treatment, FEs_to_merge = FEs_to_merge)
        Tidx <- c(which(names(pvec) == 'beta_treatment'), which(grepl('beta_treatmentinteractions', names(pvec))))
        J <- J[,Tidx]
        ni <- c()
        semat <- foreach(i = 1:length(obj$vcs), .combine = cbind, .errorhandling = 'remove') %do% {
          if (grepl('OLS', names(obj$vcs)[i])){
            Txidx = which(grepl('tr', colnames(obj$vcs[[i]]$vc), ignore.case = TRUE))
            vc <- obj$vcs[[i]]$vc[Txidx, Txidx]
            se <- sqrt(diag(J %*% vc %*% t(J)))
          } else {
            vc <- obj$vcs[[i]]$vc[Tidx, Tidx]
            se <- sqrt(diag(J %*% vc %*% t(J)))
          }
          ni[i] <- names(obj$vcs)[i]
          return(se)
        }
        colnames(semat) <- ni[!is.na(ni)]
        return(cbind(tauhat, semat))
      } else {
        return(tauhat)
      }
    } else { #if tauhat !=TRUE
      yhat <- predfun(pvec = pvec, obj = obj, newX = newX, fe.newX = fe.newX
        , new.param = new.param, new.treatment = NULL, FEs_to_merge = FEs_to_merge)
      if (se.fit == FALSE){
        return(yhat)
      } else {
        if (is.null(obj$vcs)){
          stop("No vcov matrices in object.  Can't calculate se's")
        }
        J <- jacobian(predfun, pvec, obj = obj, newX = newX, fe.newX = fe.newX
          , new.param = new.param, new.treatment = new.treatment, FEs_to_merge = FEs_to_merge)
        J <- J[,c(#re-order jacobian so that parametric terms are on the front, followed by top layer.
            which(grepl('param', names(pvec)))
          , which(names(pvec) == 'beta_treatment')
          , which(grepl('beta_treatmentinteractions', names(pvec)))
          , which(grepl('beta', names(pvec)) & !grepl('param', names(pvec)) & !grepl('treatment', names(pvec)))
          , which(!grepl('beta', names(pvec)))#no particular order to lower-level parameters
         )]
        vcnames <- c()
        semat <- foreach(i = 1:length(obj$vcs), .combine = cbind, .errorhandling = 'remove') %do% {
          if (grepl('OLS', names(obj$vcs)[i])){
            X <- J[, 1:sum(grepl('beta', names(pvec)))]#the Jacobian is ordered so that the top layer is first...
            se <- sqrt(diag(X %*% obj$vcs[[i]]$vc %*% t(X)))
          } else {
            se <- sqrt(diag(J %*% obj$vcs[[i]]$vc %*% t(J)))
          }
          vcnames[i] <- names(obj$vcs)[i]
          return(matrix(se))
        }
        if (any(is.na(vcnames))){warning("One or more VCV has negative diagonals")}
        colnames(semat) <- vcnames[!is.na(vcnames)]
      }
      return(cbind(yhat, semat))
    }
  }
}



#prediction function, potentially for the Jacobian
predfun <- function(pvec, obj, newX = NULL, fe.newX = NULL, new.param = NULL, new.treatment = NULL, tauhat = FALSE, FEs_to_merge = NULL){
  if (obj$activation == 'tanh'){
    sigma <- tanh
  }
  if (obj$activation == 'logistic'){
    sigma <- logistic
  }
  parlist <- relist(pvec)
  if (obj$doscale == TRUE){
    D <- sweep(sweep(newX, 2, STATS = attr(obj$X, "scaled:center"), FUN = '-'), 2, STATS = attr(obj$X, "scaled:scale"), FUN = '/')
    if (!is.null(obj$param)){
      P <- sweep(sweep(new.param, 2, STATS = attr(obj$param, "scaled:center"), FUN = '-'), 2, STATS = attr(obj$param, "scaled:scale"), FUN = '/')
    }
  }
  for (i in 1:length(obj$hidden_units)){
    if (obj$used_bias == TRUE){D <- cbind(1,D)}
    D <- sigma(D %*% parlist[[i]])
  } 
  colnames(D) <- paste0('nodes',1:ncol(D))
  if (!is.null(obj$treatment)){
    #Add treatment interactions
    if (obj$interact_treatment == TRUE){
      ints <- sweep(D, 1, new.treatment, '*')
      colnames(ints) <- paste0('TrInts',1:ncol(ints))
      D <- cbind(ints, D)
    }
    #Add treatment dummy
    D <- cbind(new.treatment, D)
    colnames(D)[1] <- 'treatment'
  }
  if (!is.null(obj$param)){
    D <- cbind(P, D)
    colnames(D)[1:ncol(new.param)] <- paste0('param',1:ncol(new.param))
  }
  if (is.null(obj$fe_var)){D <- cbind(1, D)}#add intercept if no FEs
  if (is.null(obj$fe)){
    yhat <- D %*% c(parlist$beta_param, parlist$beta_treatment, parlist$beta_treatmentinteractions, parlist$beta)
  } else {
    xpart <- D %*% c(parlist$beta_param, parlist$beta_treatment, parlist$beta_treatmentinteractions, parlist$beta)
    nd <- data.frame(fe.newX, xpart, id = 1:length(fe.newX))       
    nd <- merge(nd, FEs_to_merge, by.x = 'fe.newX', by.y = 'fe_var', all.x = TRUE, sort = FALSE)
    nd <- nd[order(nd$id),]
    yhat <- nd$fe + nd$xpart
  }
  if (tauhat == TRUE) {
    taumat <- D[,grepl('tr', colnames(D), ignore.case = TRUE)]
    return(taumat)
  } else {
    return(yhat)
  }
}

