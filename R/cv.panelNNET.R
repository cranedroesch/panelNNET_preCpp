cv.panelNNET <-
function(obj, folds = NULL, nfolds = 10, parallel = TRUE, approx = 'OLS', J = NULL, wise = 'fewise'...){
obj <- mlist[[10]]
folds = NULL
nfolds = 10
parallel = TRUE
approx = 'OLS'
wise = 'obswise'
#J = J
  #Assign folds if unassigned
  if (is.null(folds)){
    if (is.null(obj$fe_var) | wise == 'obswise'){#If no time variable, assume that the data is not panel and do obs-wise cross-validation
      foldid <- sample(1:nrow(obj$X) %% nfolds)+1      
      folds <- data.frame(id = 1:length(foldid), foldid)
    } else {#If time variable assume panel and do time-period-wise cross-validation
      utv <- sort(unique(obj$time_var))
      foldid <- sample(1:length(utv) %% nfolds)+1
      folds <- data.frame(id = utv, foldid)
      if (nrow(folds)<nfolds){
        nfolds <- nrow(folds)
        warning('More folds than time periods -- CV is now leave-one-time-period-out-CV')
      } #If there are fewer time periods than folds, reset the number of folds
    }
  }
  #parallelization...
  `%fun%` <- ifelse(parallel == TRUE, `%dopar%`, `%do%`)
  #"X" matrix -- based on OLS approximation
  if (approx == 'OLS'){
    X <- obj$hidden_layers[[length(obj$hidden_layers)]]
  } else {
    X <- J
  }
  #de-mean the y's
  if (!is.null(obj$fe_var)){
    ydm <- demeanlist(obj$y, list(obj$fe_var))
  }
  #ridge penalty
  D <- rep(obj$lam, ncol(X))
  if (is.null(obj$fe_var)){
    pp <- c(0, obj$parapen) #never penalize the intercept
  } else {
    pp <- obj$parapen #parapen
  }
  D[1:length(pp)] <- D[1:length(pp)]*pp #incorporate parapen into diagonal of covmat
  #Loop through folds:
  cv.err <- foreach(i = 1:nfolds, .combine = c) %fun% {
    tr <- folds$foldid != i
    te <- tr == FALSE
    if (all(te == FALSE)){
      warning("One of the folds had no test set and got dropped")
      return(NULL)
    }
    #get the fe's
    if (!is.null(obj$fe_var)){
      #get the coefs
      if (wise == 'obswise'){
        Xdm <- demeanlist(X, list(obj$fe_var))
        B <- solve(crossprod(Xdm[tr,]) + diag(D)) %*% t(Xdm[tr,]) %*% ydm[tr]
        yhatdmi <- Xdm[te,] %*% B
        mse <- mean((ydm[te] - yhatdmi)^2)
      } else {
        Xdm <- demeanlist(X[tr,], list(obj$fe_var[tr]))
        B <- solve(crossprod(Xdm) + diag(D)) %*% t(Xdm) %*% ydm[tr]
        #get the FE's
        alpha <- (obj$y - ydm)[tr] - (X[tr,] - Xdm) %*% B
        fe <- tapply(alpha, factor(obj$fe_var[tr]), mean)
        fe <- data.frame(fe_var = names(fe), fe = fe)
        fe <- merge(data.frame(fe_var = obj$fe_var[te]), fe)$fe
        #Calc the MSE
        yhati <- fe + as.numeric(X[te,] %*% B)
        mse <- mean((obj$y[te] - yhati)^2)
      }
    } else {
      Xr <- X[tr,]
      B <- solve(crossprod(Xr) + diag(D)) %*% t(Xr) %*% obj$y[tr]
      p <- X[te,]%*% B
      mse <- mean((obj$y[te] - p)^2)
    }
    return(mse)
  }
  return(list(err.mean = mean(cv.err), err.sd = sd(cv.err), err.vec = cv.err))
}
