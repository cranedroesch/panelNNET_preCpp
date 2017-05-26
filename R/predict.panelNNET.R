predict.panelNNET <-
function(obj, newX = NULL, fe.newX = NULL, new.param = NULL, new.treatment = NULL, se.fit = FALSE, tauhat = FALSE, numerical_jacobian = FALSE, parallel_jacobian = FALSE){
#obj <- pnn
#newX = Z[e,]
#fe.newX = id[e]
#new.param = P[e,, drop = FALSE]
#se.fit = TRUE
#parallel_jacobian = TRUE
#numerical_jacobian = FALSE
  if (obj$activation == 'tanh'){
    activ <- tanh
  }
  if (obj$activation == 'logistic'){
    activ <- logistic
  }
  if (obj$activation == 'relu'){
    activ <- relu
  }
  if (obj$activation == 'lrelu'){
    activ <- lrelu
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
      FEs_to_merge <- foreach(i = 1:length(unique(obj$fe$fe_var)), .combine = rbind) %do% {
        #Because of numerical error, fixed effects within units can sometimes be slightly different.  This averages them.
        data.frame(unique(obj$fe$fe_var)[i], mean(obj$fe$fe[obj$fe$fe_var == unique(obj$fe$fe_var)[i]]))
      }
      colnames(FEs_to_merge) <- c('fe_var','fe')
    } else {FEs_to_merge <- NULL}
    #(predfun is defined below)
    if (tauhat == TRUE){
      stop('HTEs are depricated, and would need major attention to be rebuilt')
    } else { #if tauhat !=TRUE
      yhat <- predfun(pvec = pvec, obj = obj, newX = newX, fe.newX = fe.newX
        , new.param = new.param, new.treatment = NULL, FEs_to_merge = FEs_to_merge)
      if (se.fit == FALSE){
        return(yhat)
      } else {
        if (is.null(obj$vcs)){
          stop("No vcov matrices in object.  Can't calculate se's")
        }
        #predicted pseudovariables
        if (any(grepl('Jac', names(obj$vcs)))){#only calculate the jacobian of the new obs if you have to
          if (numerical_jacobian == FALSE){
            J <- Jacobian.panelNNET(obj, numerical = FALSE, parallel = parallel_jacobian
              , step = 1e-9, newX = newX, new.param = new.param, fe.newX = fe.newX)
          } else {
            J <- jacobian(predfun, pvec, obj = obj, newX = newX, fe.newX = fe.newX
              , new.param = new.param, new.treatment = new.treatment, FEs_to_merge = FEs_to_merge)
            J <- J[,c(#re-order jacobian so that parametric terms are on the front, followed by top layer.
                which(grepl('param', names(pvec)))
              , which(names(pvec) == 'beta_treatment')
              , which(grepl('beta_treatmentinteractions', names(pvec)))
              , which(grepl('beta', names(pvec)) & !grepl('param', names(pvec)) & !grepl('treatment', names(pvec)))
              , which(!grepl('beta', names(pvec)))#no particular order to lower-level parameters
            )]
          }
        }
        #predicted top-level variables
        X <- predfun(pvec, obj = obj, newX = newX, fe.newX = fe.newX
            , new.param = new.param, new.treatment = new.treatment, FEs_to_merge = FEs_to_merge, return_toplayer = TRUE)
        vcnames <- c()
        semat <- foreach(i = 1:length(obj$vcs), .combine = cbind, .errorhandling = 'remove') %do% {
          if (grepl('OLS', names(obj$vcs)[i])){
            se <- foreach(j = 1:N, .combine = c)%do% {
              sqrt(X[j,, drop = FALSE] %*% obj$vcs[[i]] %*% t(X[j,, drop = FALSE]))
            }
          } else {
            se <- foreach(j = 1:N, .combine = c)%do% {
              sqrt(J[j,, drop = FALSE] %*% obj$vcs[[i]] %*% t(J[j,, drop = FALSE]))
            }
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


N <- 10000
P <- 4000
A <- matrix(rnorm(P^2), P)
vc <- crossprod(A)
X <- matrix(rnorm(N*P), ncol = P)

PT <- proc.time()
semat <- sqrt(diag(X %*% vc %*% t(X)))
proc.time() - PT

PT <- proc.time()
seloop <- foreach(i = 1:N, .combine = c)%do% {
  sqrt(X[i,, drop = FALSE] %*% vc %*% t(X[i,, drop = FALSE]))
}
proc.time() - PT

all.equal(semat, seloop)


#prediction function, potentially for the Jacobian
predfun <- function(pvec, obj, newX = NULL, fe.newX = NULL, new.param = NULL, new.treatment = NULL, tauhat = FALSE, FEs_to_merge = NULL, return_toplayer = FALSE){
  if (obj$activation == 'tanh'){
    activ <- tanh
  }
  if (obj$activation == 'logistic'){
    activ <- logistic
  }
  if (obj$activation == 'relu'){
    activ <- relu
  }
  if (obj$activation == 'lrelu'){
    activ <- lrelu
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
    D <- activ(as.matrix(D) %*% parlist[[i]])
  } 

  colnames(D) <- paste0('nodes',1:ncol(D))
  if (!is.null(obj$param)){
    D <- cbind(P, D)
    colnames(D)[1:ncol(new.param)] <- paste0('param',1:ncol(new.param))
  }
  if (is.null(obj$fe_var)){D <- cbind(1, D)}#add intercept if no FEs
  if (return_toplayer == TRUE){
    return(D)
  }
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
  }
  #otherwise...
  return(yhat)
}

