cv.panelNNET <-
function(obj, folds = NULL, nfolds = 10, parallel = TRUE, type = 'OLS', J = NULL, wise = 'fewise',...){
##test arguments...
#obj <- pnn
#folds = NULL
#nfolds = 10
#parallel = TRUE
#type = 'full'
#wise = 'fewise'
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
  if (type == 'OLS'){
    X <- obj$hidden_layers[[length(obj$hidden_layers)]]
    if (obj$lam>0){
      warning('OLS approximation will perform poorly when neural net is penalized')
    }
  }
  if (type == 'Jacobian'){
    X <- J
#    if (obj$lam>0){
#      warning('Jacobian approximation will perform poorly when neural net is penalized')
#    }
  }
  if (type == 'full'){
    X <- obj$X
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
    tr <- folds$id[which(folds$foldid != i)]
    te <- folds$id[which(folds$foldid == i)]
    if (all(te == FALSE)){
      warning("One of the folds had no test set and got dropped")
      return(NULL)
    }
    if (type == 'full'){
      #jitter the parameters
      pl <- unlist(as.relistable(obj$parlist))
      pl <- relist(pl+rnorm(length(pl), sd = abs(pl+.01)/10))
      conv <- FALSE
      subcounter <- 0
      while(conv == FALSE){
        optpass <- panelNNET(obj$y[obj$time_var %in% tr], obj$X[obj$time_var %in% tr,], hidden_units = obj$hidden_units
          , fe_var = obj$fe_var[obj$time_var %in% tr], maxit = obj$maxit, lam = obj$lam
          , time_var = obj$time[obj$time_var %in% tr], param = obj$param[obj$time_var %in% tr,, drop = FALSE],  verbose = FALSE
          , convtol = obj$convtol, activation = obj$activation, inference = FALSE
          , parlist = pl, 
          , useOptim = obj$usedOptim, optimMethod = obj$optimMethod
        )
        MBpass <- panelNNET(obj$y[obj$time_var %in% tr], obj$X[obj$time_var %in% tr,], hidden_units = obj$hidden_units
          , fe_var = obj$fe_var[obj$time_var %in% tr], maxit = 200, lam = obj$lam
          , time_var = obj$time[obj$time_var %in% tr], param = obj$param[obj$time_var %in% tr,, drop = FALSE],  verbose = FALSE
          , convtol = obj$convtol, activation = obj$activation, inference = FALSE
          , parlist = optpass$parlist, 
          , batchsize = round(sum(obj$time_var %in% tr)/10)
        )
        if (optpass$loss <= MBpass$loss | subcounter > 10){
          conv <- TRUE
        } else {
          pl <- MBpass$parlist
          subcounter <- subcounter + 1
        }
      }
      p <- predict(optpass, newX = obj$X[obj$time_var %in% te,]
        , fe.newX = obj$fe_var[obj$time_var %in% te]
        , new.param =obj$param[obj$time_var %in% te,, drop = FALSE]
      )
      mse <- mean((obj$y[obj$time_var %in% te] - p)^2)
    } else {
      #get the fe's
      if (!is.null(obj$fe_var)){
        #get the coefs
        if (wise == 'obswise'){
          Xdm <- demeanlist(X, list(obj$fe_var))
          B <- solve(crossprod(Xdm[tr,]) + diag(D)) %*% t(Xdm[tr,]) %*% ydm[tr]
          yhatdmi <- Xdm[te,] %*% B
          mse <- mean((ydm[te] - yhatdmi)^2)
        } else {
          Xdm <- demeanlist(X[obj$time_var %in% tr,], list(obj$fe_var[obj$time_var %in% tr]))
          B <- solve(crossprod(Xdm) + diag(D)) %*% t(Xdm) %*% ydm[obj$time_var %in% tr]
          #get the FE's
          alpha <- (obj$y - ydm)[obj$time_var %in% tr] - (X[obj$time_var %in% tr,] - Xdm) %*% B
          fe <- tapply(alpha, factor(obj$fe_var[obj$time_var %in% tr]), mean)
          fe <- data.frame(fe_var = names(fe), fe = fe)
          fe <- merge(data.frame(fe_var = obj$fe_var[obj$time_var %in% te]), fe)$fe
          #Calc the MSE
          yhati <- fe + as.numeric(X[obj$time_var %in% te,] %*% B)
          mse <- mean((obj$y[obj$time_var %in% te] - yhati)^2)
        }
      } else {
        Xr <- X[tr,]
        B <- solve(crossprod(Xr) + diag(D)) %*% t(Xr) %*% obj$y[tr]
        p <- X[te,]%*% B
        mse <- mean((obj$y[te] - p)^2)
      }
    }
    return(mse)
  }
  return(list(err.mean = mean(cv.err), err.sd = sd(cv.err), err.vec = cv.err))
}
