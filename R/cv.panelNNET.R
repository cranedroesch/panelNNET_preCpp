cv.panelNNET <-
function(obj, folds = NULL, nfolds = 10, parallel = TRUE){
#obj <- pnn
#folds = NULL
#nfolds = 10
#parallel = TRUE
  if (is.null(folds)){
    utv <- sort(unique(obj$time_var))
    foldid <- sample(1:length(utv) %% nfolds)+1
    folds <- data.frame(year = utv, foldid)
  }
  `%fun%` <- ifelse(parallel == TRUE, `%dopar%`, `%do%`)
  #"X" matrix -- based on OLS approximation
  X <- obj$hidden_layers[[length(obj$hidden_layers)]]
  #de-mean the y's
  ydm <- demeanlist(obj$y, list(obj$fe_var))
  #ridge penalty
  D <- rep(obj$lam, ncol(X))
  if (is.null(obj$fe_var)){
    pp <- c(1, obj$parapen) #parapen
  } else {
    pp <- obj$parapen #parapen
  }
  D[1:length(pp)] <- D[1:length(pp)]*pp #incorporate parapen into diagonal of covmat
  cv.err <- foreach(i = 1:nfolds, .combine = c) %fun% {
    tr <- obj$time_var %ni% folds$year[folds$foldid == i]
    te <- tr == FALSE
    #get the fe's
    m <- felm(obj$y[tr]~X[tr,]|obj$fe_var[tr])
    fe <- merge(obj$fe, getfe(m), by.x = 'fe_var', by.y = 'idx')[te,'effect']
    #get the coefs
    Xdm <- demeanlist(X[tr,], list(obj$fe_var[tr]))
    B <- solve(crossprod(Xdm) + diag(D)) %*% t(Xdm) %*% ydm[tr]
    yhati <- fe + X[te,] %*% B
    mse <- mean((obj$y[te] - yhati)^2)
    return(mse)
  }
  return(list(err.mean = mean(cv.err), err.sd = sd(cv.err), err.vec = cv.err))
}
