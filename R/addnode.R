


addnode <- function(obj, inv_activ, layer = 1){
#layer <- 1
#obj <- pnn
#inv_activ <- function(v){v[v < 0] <- v[v < 0] / 0.01; v}
  warning('addnode is still in early development.  results can be pretty unstable')
  if (length(pnn$hidden_layers) == 1){
    x <- obj$X
    if (obj$doscale == TRUE) {
      x <- scale(x)
    }
  } else {
    stop('this has not yet been implemented for multi-layer nets')
  }
  #get residual
  u <- pnn$y - pnn$yhat
  #get size of other param vecs
  parmsize <- mean(apply(obj$parlist[[layer]],2,function(x){sum(x^2)}))
  #fit linear model  
  M <- cbind(1,x)
  f <- function(L){
    m <- glmnet(y = inv_activ(u), x = M, intercept = FALSE, lambda = L)
    (sum(coef(m)^2) - parmsize)^2
  }
  o <- optim(f, par = 30, method = 'Brent', lower = 0, upper = 100)
  m <- glmnet(y = inv_activ(u), x = M, intercept = FALSE, lambda = o$par)
  pl <- pnn$parlist
  pl[[1]] <- cbind(pl[[1]], as.numeric(coef(m))[-1])
  pl$beta <- c(pl$beta, 1)
  return(pl)
}





#rm(list=ls())
#gc()
#gc()
#"%ni%" <- Negate("%in%")

###for AWS...
##system('sudo apt-get install libcurl4-openssl-dev libssl-dev htop')
###tmux attach
##install.packages('doParallel')
##install.packages('mvtnorm')
##install.packages('foreach')
##install.packages('numDeriv')
##install.packages('devtools')
##install.packages('lfe')
##setwd('/home/ubuntu/pnn')

#library(devtools)
#install_github('cranedroesch/panelNNET')

#library(panelNNET)
#library(doParallel)
#library(parallel)
#library(mvtnorm)
#registerDoParallel(detectCores())


##function to work through 1000 draws of the DGP for a given sample size and number of time periods
##simfunc <- function(N, t){
#N <- 2000
#t = 40
#  pz <- 5
#  pid <- N/t
#  id <- (1:N-1) %/% (N/pid) +1
#  time <- 0:(N-1) %% (N/pid) +1
#  #start looping through draws of the DGP
##  sims <- foreach(k = 1:1000, .errorhandling = 'stop') %dopar%{
##k=1
##    set.seed(k)
##    print(k)
#    #Each group has its own covariance matrix
#    groupcov <- foreach(i = 1:pid) %do% {
#      A <- matrix(rnorm(pz^2), pz)
#      t(A) %*% A
#    }
#    #and its own mean
#    groupmean <- foreach(i = 1:pid) %do% {
#      rnorm(pz, sd  =5)
#    }
#    #and it's own effect that is distinct from its covariate distribution
#    id.eff <- as.numeric(id)

#    #this is the data generated from those distributions
#    Z <- foreach(i = 1:N, .combine = rbind)%do%{
#      mvrnorm(1, groupmean[[id[i]]], groupcov[[id[i]]])
#    }
#    #outcome minus noise
#    y <- time +log(dmvnorm(Z, rep(0, pz), diag(rep(1, pz)))) + id.eff
#    #y <- time +Z %*% rnorm(1:pz)
#    u <- rnorm(N, sd = 20)
#    y <- y+u
#    id <- as.factor(id)
#    #training and test and validation
#    v <- time>max(time)*.9
#    r <- time %in% time[which(v==FALSE & time %%2)]
#    e <- time %in% time[which(v==FALSE & (time+1) %%2)]
#    P <- matrix(time)
#    #put in data frame and estimate fe model
#    dat <- data.frame(y, Z, time, id)
#    mfe <- lm(y~.-1, data = dat[r|e,])
#    pfe <- predict(mfe, newdata = dat[v,])

#    cvlist <- mlist <- list()
#    i <- counter <- 0
#    oldmse <- newmse <- Inf
#    msevalvec <- msetestvec <- msecv <- c()
#    lam <- 0
#    g = c(2)
#    pl <- NULL
##    while(i <20 & counter < 4){
#      i <- 1+i
#      oldmse <- newmse
#      #Batch gradeint descent
#      pnn <- panelNNET(y[r], Z[r,], hidden_units = g
#        , fe_var = id[r], maxit = 100, lam = lam
#        , time_var = time[r], param = P[r,, drop = FALSE],  verbose = TRUE
#        , gravity = 1.01, convtol = 1e-6, activation = 'lrelu', inference = FALSE
#        , start_LR = .01, parlist = pl, OLStrick = TRUE
#        , initialization = 'enforce_normalization'
#      )


#pl <- addnode(pnn, function(v){v[v < 0] <- v[v < 0] / 0.01; v}, 1)
#pnn$loss
#pnn2 <- panelNNET(y[r], Z[r,], hidden_units = g
#        , fe_var = id[r], maxit = 100, lam = lam
#        , time_var = time[r], param = P[r,, drop = FALSE],  verbose = TRUE
#        , gravity = 1.01, convtol = 1e-6, activation = 'lrelu', inference = FALSE
#        , start_LR = .01, parlist = pl, OLStrick = TRUE
#        , initialization = 'enforce_normalization'
#      )
#pnn2$loss


#lsvec <- lvec <- c()
#lam <- .01
#pl <- NULL
#while(TRUE){
#  pnn <- panelNNET(y[r], Z[r,], hidden_units = g
#    , fe_var = id[r], maxit = 1, lam = lam
#    , time_var = time[r], param = P[r,, drop = FALSE],  verbose = FALSE
#    , gravity = 1.01, convtol = 1e-6, activation = 'lrelu', inference = FALSE
#    , start_LR = .01, parlist = pl, OLStrick = FALSE
#    , initialization = 'enforce_normalization'
##    , batchsize  = ceiling(sum(r)/10)
#  )
###
#  lsvec <- append(lsvec, pnn$loss)
#  pnn <- panelNNET(y[r], Z[r,], hidden_units = g
#    , fe_var = id[r], maxit = 200, lam = lam
#    , time_var = time[r], param = P[r,, drop = FALSE],  verbose = FALSE
#    , gravity = 1.01, convtol = 1e-6, activation = 'lrelu', inference = FALSE
#    , start_LR = .01, parlist = pnn$parlist, OLStrick = FALSE
#    , initialization = 'enforce_normalization'
#    , batchsize  = ceiling(sum(r)/10)
#  )
#  pnn$parlist <- OLStrick_function(pnn$parlist, pnn$hidden_layers, pnn$y, pnn$fe_var, pnn$lam, pnn$parapen, pnn$treatment)
#  pnn <- panelNNET(y[r], Z[r,], hidden_units = g
#    , fe_var = id[r], maxit = 200, lam = lam
#    , time_var = time[r], param = P[r,, drop = FALSE],  verbose = FALSE
#    , gravity = 1.01, convtol = 1e-6, activation = 'lrelu', inference = FALSE
#    , start_LR = .01, parlist = pnn$parlist, OLStrick = TRUE
#    , initialization = 'enforce_normalization'
#  )

#  pl <- addnode(pnn, function(v){v[v < 0] <- v[v < 0] / 0.01; v}, 1)
#print(pl)
#  pnn <- panelNNET(y[r], Z[r,], hidden_units = g
#    , fe_var = id[r], maxit = 200, lam = lam
#    , time_var = time[r], param = P[r,, drop = FALSE],  verbose = FALSE
#    , gravity = 1.01, convtol = 1e-6, activation = 'lrelu', inference = FALSE
#    , start_LR = .01, parlist = pl, OLStrick = FALSE
#    , initialization = 'enforce_normalization'
#    , batchsize  = ceiling(sum(r)/10)
#  )
#  lvec <- append(lvec, pnn$loss)
#  par(mfrow = c(1,2))
#  plot(lvec)
#  plot(lsvec)
#  print(lvec)
#  print(lsvec)

#print(pnn$parlist)
#}


#while(TRUE){
#  pnn <- panelNNET(y[r], Z[r,], hidden_units = g
#    , fe_var = id[r], maxit = 1000, lam = lam
#    , time_var = time[r], param = P[r,, drop = FALSE],  verbose = TRUE
#    , gravity = 1.01, convtol = 1e-6, activation = 'lrelu', inference = FALSE
#    , start_LR = .01, parlist = pnn$parlist, OLStrick = FALSE
#    , initialization = 'enforce_normalization'
#    , batchsize  = ceiling(sum(r)/10)
#  )
#  pnn <- panelNNET(y[r], Z[r,], hidden_units = g
#    , fe_var = id[r], maxit = 1000, lam = lam
#    , time_var = time[r], param = P[r,, drop = FALSE],  verbose = TRUE
#    , gravity = 1.01, convtol = 1e-6, activation = 'lrelu', inference = FALSE
#    , start_LR = .01, parlist = pnn$parlist, OLStrick = TRUE
#    , initialization = 'enforce_normalization'
##    , batchsize  = ceiling(sum(r)/10)
#  )
#}

#plot(pnn)
#    p <- predict(pnn, newX = Z[v,], fe.newX = id[v], new.param = matrix(time[v]), se.fit = FALSE)
#mean((y[v] - p)^2)






