predict.panelNNET <-
function(obj, newX = NULL, fe.newX = NULL, new.param = NULL, new.treatment = NULL, se.fit = FALSE){
#obj <- pnn
#fe.newX = NULL
#newX = matrix(x)
#new.param = matrix(time)
#new.treatment = rep(1, N)
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
    #Scale the new data by the scaling rules in the training data
    plist <- as.relistable(obj$parlist)
    pvec <- unlist(plist)
    #prepare fe's in advance...
    if (!is.null(obj$fe)){
      tm <- foreach(i = 1:length(unique(obj$fe$fe_var)), .combine = rbind) %do% {
        #Because of numerical error, fixed effects within units can sometimes be slightly different.  This averages them.
        data.frame(unique(obj$fe$fe_var)[i], mean(obj$fe$fe[obj$fe$fe_var == unique(obj$fe$fe_var)[i]]))
      }
      colnames(tm) <- c('fe_var','fe')
    }
    #prediction function, potentially for the Jacobian
    predfun <- function(pvec, obj, newX = NULL, fe.newX = NULL, new.param = NULL, new.treatment = NULL){
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
        yhat <- D %*% with(parlist, c(beta_param, beta_treatment, beta_treatmentinteractions, beta))
      } else {
        xpart <- D %*% with(parlist, c(beta_param, beta_treatment, beta_treatmentinteractions, beta))
        nd <- data.frame(fe.newX, xpart, id = 1:length(fe.newX))       
        nd <- merge(nd, tm, by.x = 'fe.newX', by.y = 'fe_var', all.x = TRUE, sort = FALSE)
        nd <- nd[order(nd$id),]
        yhat <- nd$fe + nd$xpart
      }
      return(yhat)
    }
    yhat <- predfun(pvec = pvec, obj = obj, newX = newX, fe.newX = fe.newX, new.param = new.param, new.treatment)
    if (se.fit == FALSE){
      return(yhat)
    } else {
      if (is.null(obj$vcs)){
        stop("No vcov matrices in object.  Can't calculate se's")
      }
      J <- jacobian(predfun, pvec, obj = obj, newX = newX, fe.newX = fe.newX, new.param = new.param, new.treatment = new.treatment)
      J <- J[,c(#re-order jacobian so that parametric terms are on the front, followed by top layer.
          which(grepl('param', names(pvec)))
        , which(names(pvec) == 'beta_treatment')
        , which(grepl('beta_treatmentinteractions', names(pvec)))
        , which(grepl('beta', names(pvec)) & !grepl('param', names(pvec)) & !grepl('treatment', names(pvec)))
        , which(!grepl('beta', names(pvec)))#no particular order to lower-level parameters
       )]
      ni <- c()
      semat <- foreach(i = 1:length(obj$vcs), .combine = cbind, .errorhandling = 'remove') %do% {
        if (grepl('OLS', names(obj$vcs)[i])){
          X <- J[, 1:sum(grepl('beta', names(pvec)))]#the Jacobian is ordered so that the top layer is first...
          se <- sqrt(diag(X %*% obj$vcs[[i]]$vc %*% t(X)))
        } else {
          se <- sqrt(diag(J %*% obj$vcs[[i]]$vc %*% t(J)))
        }
        ni[i] <- names(obj$vcs)[i]
        return(se)
      }
      colnames(semat) <- ni[!is.na(ni)]
    }
    return(cbind(yhat, semat))
  }
}



