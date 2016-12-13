vcov.panelNNET <-
function(obj, option, J = NULL){
  e <- obj$y - obj$yhat
  if (is.null(J) & grepl('Jacobian', option)){  J <- Jacobian.panelNNET(obj)
    } else {J <- obj$hidden_layers[[length(obj$hidden_layers)]]}
  D <- rep(obj$lam, ncol(J))
  if (is.null(obj$fe_var)){
    pp <- c(1, obj$parapen) #parapen
  } else {
    pp <- obj$parapen #parapen
  }
  D[1:length(pp)] <- D[1:length(pp)]*pp #incorporate parapen into diagonal of covmat
  bread <- solve(t(J) %*% J + diag(D))
  if (option == 'Jacobian_homoskedastic'){
    vcov <- sum(e^2)/(length(e) - ncol(J)) * bread
  }
  if (option == 'Jacobian_sandwich'){
    meat <- foreach(i = 1:length(e), .combine = '+') %do% {
      e[i]^2*J[i,] %*% t(J[i,])
    }
    vcov <- (length(e)-1)/(length(e) - ncol(J)) * bread %*% meat %*% bread
  }
  if (option == 'Jacobian_cluster'){
    G <- length(unique(obj$fe_var))
    meat <- foreach(i = 1:G, .combine = '+')%do%{
      ei <- e[obj$fe_var == unique(obj$fe_var)[i]]
      Ji <- J[obj$fe_var == unique(obj$fe_var)[i],]
      t(Ji) %*% ei %*% t(ei) %*% Ji
    }
    vcov <- G/(G-1)*(length(e) - 1)/(length(e) - ncol(J)) * bread %*% meat %*% bread
  }
  if (option == 'OLS'){
    vcov <- sum(e^2)/(length(e) - ncol(J)) * bread
  }
  if (option == 'sandwich'){
    meat <- foreach(i = 1:length(e), .combine = '+') %do% {
      e[i]^2*J[i,] %*% t(J[i,])
    }
    vcov <- (length(e)-1)/(length(e) - ncol(J)) * bread %*% meat %*% bread
  }
  if (option == 'cluster'){
    G <- length(unique(obj$fe_var))
    meat <- foreach(i = 1:G, .combine = '+')%do%{
      ei <- e[obj$fe_var == unique(obj$fe_var)[i]]
      Ji <- J[obj$fe_var == unique(obj$fe_var)[i],]
      t(Ji) %*% ei %*% t(ei) %*% Ji
    }
    vcov <- G/(G-1)*(length(e) - 1)/(length(e) - ncol(X)) * bread %*% meat %*% bread
  }
  return(list(vc = vcov, J = J))
}
