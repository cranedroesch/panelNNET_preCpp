vcov.panelNNET <-
function(obj, option, J = NULL){
#obj <- pnn
#option = 'Jacobian_cluster'
  e <- obj$y - obj$yhat
  if (is.null(J)){
    if (grepl('Jacobian', option)){
      J <- Jacobian.panelNNET(obj)
    } else {#if using OLS approximation...
      J <- obj$hidden_layers[[length(obj$hidden_layers)]]
      if (!is.null(obj$fe_var)){
        J <- demeanlist(J, list(obj$fe_var))
        targ <- demeanlist(obj$y, list(obj$fe_var))
      } else {targ = obj$y}
      #OLS trick
      constraint <- sum(c(obj$parlist$beta_param*obj$parapen, obj$parlist$beta)^2)
      #getting implicit regressors depending on whether regression is panel
      if (!is.null(obj$fe_var)){
        Zdm <- demeanlist(obj$hidden_layers[[length(obj$hidden_layers)]], list(obj$fe_var))
      } else {
        Zdm <- obj$hidden_layers[[length(obj$hidden_layers)]]
        targ <- y
      }
      #function to find implicit lambda
      f <- function(lam){
        bi <- solve(t(J) %*% J + diag(rep(lam, ncol(J)))) %*% t(J) %*% targ
        (t(bi) %*% bi - constraint)^2
      }
      #optimize it
      o <- optim(par = obj$lam, f = f, method = 'Brent', lower = obj$lam/2, upper = 1e9)
      #new lambda
      obj$lam <- o$par
    }
  }
  #set uo penality vector
  D <- rep(obj$lam, ncol(J))
  if (is.null(obj$fe_var)){
    pp <- c(0, obj$parapen) #never penalize the intercept
  } else {
    pp <- obj$parapen #parapen
  }
  if (!is.null(obj$treatment)){pp <- append(pp, 0)}#treatment always follows parametric terms and will not be penalized
  D[1:length(pp)] <- D[1:length(pp)]*pp #incorporate parapen into diagonal of covmat
  #check effective degrees of freedom against parameters
  jtj <- crossprod(J)
  ev <- eigen(jtj)$values
  edf <- sum(ev/(ev+D))
  if (edf>= nrow(J)){
    warning('more effective degrees of freedom than observations.  change the architecture or increase the penalty if you want to calculate a parameter covariance matrix')
    return(NULL)
  }
  bread <- solve(t(J) %*% J + diag(D))
  if (option == 'Jacobian_homoskedastic'){
    vcov <- sum(e^2)/(length(e) - edf) * bread
  }
  if (option == 'Jacobian_sandwich'){
    meat <- foreach(i = 1:length(e), .combine = '+') %do% {
      e[i]^2*J[i,] %*% t(J[i,])
    }
    vcov <- (length(e)-1)/(length(e) - edf) * bread %*% meat %*% bread
  }
  if (option == 'Jacobian_cluster'){
    G <- length(unique(obj$fe_var))
    meat <- foreach(i = 1:G, .combine = '+')%do%{
      ei <- e[obj$fe_var == unique(obj$fe_var)[i]]
      Ji <- J[obj$fe_var == unique(obj$fe_var)[i],]
      t(Ji) %*% ei %*% t(ei) %*% Ji
    }
    vcov <- G/(G-1)*(length(e) - 1)/(length(e) - edf) * bread %*% meat %*% bread
  }
  if (option == 'OLS'){
    vcov <- sum(e^2)/(length(e) - edf) * bread
  }
  if (option == 'sandwich'){
    meat <- foreach(i = 1:length(e), .combine = '+') %do% {
      e[i]^2*J[i,] %*% t(J[i,])
    }
    vcov <- (length(e)-1)/(length(e) - edf) * bread %*% meat %*% bread
  }
  if (option == 'cluster'){
    G <- length(unique(obj$fe_var))
    meat <- foreach(i = 1:G, .combine = '+')%do%{
      ei <- e[obj$fe_var == unique(obj$fe_var)[i]]
      Ji <- J[obj$fe_var == unique(obj$fe_var)[i],]
      t(Ji) %*% ei %*% t(ei) %*% Ji
    }
    vcov <- G/(G-1)*(length(e) - 1)/(length(e) - edf) * bread %*% meat %*% bread
  }
  return(list(vc = vcov, J = J))
}


