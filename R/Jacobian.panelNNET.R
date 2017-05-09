

Jacobian.panelNNET <- function(obj, numerical = FALSE, ...){
  if (numerical == FALSE){
    Jacobian.analytical(obj)
  } else {
    Jacobian.numerical(obj)
  }
}

Jacobian.numerical <- function(obj){
  if (obj$activation == 'tanh'){
    activ <- tanh
    activ_prime <- tanh_prime
  }
  if (obj$activation == 'logistic'){
    activ <- logistic
    activ_prime <- logistic_prime
  }
  if (obj$activation == 'relu'){
    activ <- relu
    activ_prime <- relu_prime
  }
  if (obj$activation == 'lrelu'){
    activ <- lrelu
    activ_prime <- lrelu_prime
  }
  plist <- as.relistable(obj$parlist)
  pvec <- unlist(plist)
  #define function to pass to `jacobian` from `numDeriv`
  Jfun <- function(pvec, obj){
    parlist <- relist(pvec)
    D <- obj$X
    for (i in 1:length(obj$hidden_units)){
      if (obj$used_bias == TRUE){D <- cbind(1,D)}
      D <- activ(D %*% parlist[[i]])
    } 
    colnames(D) <- paste0('nodes',1:ncol(D))
    if (!is.null(obj$treatment)){
      #Add treatment interactions
      if (obj$interact_treatment == TRUE){
        ints <- sweep(D, 1, obj$treatment, '*')
        colnames(ints) <- paste0('TrInts',1:ncol(ints))
        D <- cbind(ints, D)
      }
      #Add treatment dummy
      D <- cbind(obj$treatment, D)
      colnames(D)[1] <- 'treatment'
    }
    if (!is.null(obj$param)){
      D <- cbind(obj$param, D)
      colnames(D)[1:ncol(obj$param)] <- paste0('param',1:ncol(obj$param))
    }
    if (is.null(obj$fe_var)){D <- cbind(1, D)}#add intercept if no FEs
    if (!is.null(obj$fe_var)){
      Zdm <- demeanlist(D, list(obj$fe_var))
      ydm <<- demeanlist(obj$y, list(obj$fe_var))
      fe <- (obj$y-ydm) - as.matrix(D-Zdm) %*% c(parlist$beta_param, parlist$beta_treatment, parlist$beta_treatmentinteractions, parlist$beta)
      yhat <- D %*% c(parlist$beta_param, parlist$beta_treatment, parlist$beta_treatmentinteractions, parlist$beta) + fe    
    } else {
      yhat <- D %*% c(parlist$beta_param, parlist$beta_treatment, parlist$beta_treatmentinteractions, parlist$beta)
    }
    return(yhat)
  }
  #pass `Jfun` to `jacobian` from `numDeriv`
  J <- jacobian(Jfun, pvec, obj = obj)
  #drop any zero columns that represent lower-level parameters
  dJ <- ncol(J)
  tokeep <- which(!(apply(J, 2, function(x){all(x==0)}) & !grepl('treatment|param', names(pvec))))
  J <- J[,tokeep]
  if (ncol(J) < dJ){
    warning(paste0(dJ - ncol(J), ' columns dropped from Jacobian because dY/dParm =~ 0'))
  }
  pvec <- pvec[tokeep]
  #re-order jacobian so that parametric terms are on the front, followed by top layer.
  J <- J[,c(
      which(grepl('param', names(pvec)))
    , which(names(pvec) == 'beta_treatment')
    , which(grepl('beta_treatmentinteractions', names(pvec)))
    , which(grepl('beta', names(pvec)) & !grepl('param', names(pvec)) & !grepl('treatment', names(pvec)))
    , which(!grepl('beta', names(pvec)))#no particular order to lower-level parameters
   )]
  return(J)
}

Jacobian.analytical <- function(obj){
  if (obj$activation == 'tanh'){
    activ <- tanh
    activ_prime <- tanh_prime
  }
  if (obj$activation == 'logistic'){
    activ <- logistic
    activ_prime <- logistic_prime
  }
  if (obj$activation == 'relu'){
    activ <- relu
    activ_prime <- relu_prime
  }
  if (obj$activation == 'lrelu'){
    activ <- lrelu
    activ_prime <- lrelu_prime
  }
  #Start parameter data frame
  par.df <- data.frame(par = unlist(obj$parlist))
  #make list of only lower parameters
  lowerpars <- obj$parlist
  lowerpars$beta <- lowerpars$beta_param <- NULL
  #add variable signifying their level
  lev <- foreach(i = 1:length(lowerpars), .combine = c) %do% {
    rep(i, length(lowerpars[[i]]))
  }
  #add top-level parameters
  par.df$lev <- c(lev, rep('beta_param', length(obj$parlist$beta_param)), rep('beta', length(obj$parlist$beta)))
  #Which column of their layer they multiply
  which.layer <- foreach(i = 1:length(lowerpars), .combine = rbind) %do% {
    foreach(j = 1:ncol(lowerpars[[i]]), .combine = rbind) %do% {
      lev.pointer <- foreach(k = (1+obj$used_bias):nrow(lowerpars[[i]]), .combine = c) %do% {
        k
      }
      upper.lev.pointer <- foreach(k = (obj$used_bias):nrow(lowerpars[[i]]), .combine = c) %do% {
        j+ifelse(i == length(lowerpars) & !is.null(obj$fe_var), 0, obj$used_bias) 
        #^No bias at top later if we're dealing with a fixed-effects model
      }
      if(obj$used_bias){
        lev.pointer <- lev.pointer - 1
        lev.pointer <- c('bias', lev.pointer)
      }
      cbind(lev.pointer, upper.lev.pointer)
    }
  }
  par.df$lev.pointer <- c(which.layer[,1], rep(NA, length(c(obj$parlist$beta, obj$parlist$beta_param))))
  par.df$upper.lev.pointer <- c(which.layer[,2], rep(NA, length(c(obj$parlist$beta, obj$parlist$beta_param))))
  #^the betas in fact emanate from a level, but their jacobins are invariant to this

  #use chain rule to get a list of matrices of the form 
  #a'(V_L B_L)B_L-1.  
  chains <- foreach(L = 0:(length(obj$hidden_layers)-1)) %do% {
    bias <- ifelse(obj$used_bias, 1, NULL)
    if (L == 0){
      D <- cbind(bias, obj$X)
    } else {
      if (L != length(obj$hidden_layers)){
        D <- cbind(bias, obj$hidden_layers[[L]])
      } else { #top layer shouldn't include parametric terms in ere
        D <- obj$hidden_layers[[L]]
        D <- D[,grepl('nodes', colnames(D))]
        D <- cbind(bias, obj$hidden_layers[[L]])
      }
    }
    lower_par <- obj$parlist[[L+1]]
    if ((L+2) > length(obj$hidden_layers)){
      upper_par <- obj$parlist$beta #upper level param will be beta at the top level
    } else {
      upper_par <- obj$parlist[[L+2]]
    }
    #a'(layer)
    if ((L+2) > length(obj$hidden_layers)){
      chn <- activ_prime(D %*% lower_par) %*% upper_par
    } else {
      chn <- cbind(bias, activ_prime(D %*% lower_par)) %*% upper_par
    }
    return(chn)
  }
  chains <- lapply(chains, rowSums)#they will only get used summed, so sum them now
  #Lev is the level that the parameter emanates from
  #lev.pointer is the column of the level that the parameter emanates from
  #upper.lev.pointer is the column of the upper level that the parameter goes to
  #Note that the bias terms are not included in the saved hidden layers, nor in the X.  So sometimes the number one needs to be added or subtracted from the index
  Jacobian_ab <- foreach(i = 1:nrow(par.df[!grepl('beta', par.df$lev),]), .combine = cbind) %do% {
    #Get vector of (derived) variable at current level
    if (par.df$lev[i] == "1"){
      if (par.df$lev.pointer[i] == 'bias') {
        lay <- rep(1, nrow(obj$X))
      } else {
        lay <- obj$X[,as.numeric(par.df$lev.pointer[i])]
      }
    } else {
      if (par.df$lev.pointer[i] == 'bias') {
        lay <- rep(1, nrow(obj$X))
      } else {
        lay <- obj$hidden_layers[[as.numeric(par.df$lev[i])-1]][,as.numeric(par.df$lev.pointer[i])]
      }  
    }
    #Start calculating Jacobia column.  At the top level, we're just dealing with the layer
    Jcol <- lay
    if (!is.na(par.df$upper.lev.pointer[i])){
      #The layer directly above is the row-wise sum of the single linked layer and the paramters emanating from it
      if (class(obj$parlist[[as.numeric(par.df$lev[i])+1]]) == 'numeric'){#matrix or vector determines which layer we're on
        above.pars <- obj$parlist$beta[as.numeric(par.df$upper.lev.pointer[i])]
        above.layer <- obj$hidden_layers[[as.numeric(par.df$lev[i])]][,as.numeric(par.df$upper.lev.pointer[i])+ncol(obj$param)]
      } else {
        above.pars <- obj$parlist[[as.numeric(par.df$lev[i])+1]][as.numeric(par.df$upper.lev.pointer[i]),]
        above.layer <- obj$hidden_layers[[as.numeric(par.df$lev[i])]][,as.numeric(par.df$upper.lev.pointer[i])-1]
      }
      above <- rowSums(as.matrix(activ_prime(above.layer)) %*% t(as.matrix(above.pars)))
      Jcol <- Jcol * above
      #After the top layer, matrix expressions work again
      #Element-wise multiply the row-sums of the chains for the corresponding upper layers
      if (as.numeric(par.df$lev[i]) < length(lowerpars) & !is.na(as.numeric(par.df$lev[i]))){
        chainstart <- as.numeric(par.df$lev[i]) + 1 
        top <- foreach(ch = chainstart:length(chains), combine = '*') %do% {
          chains[[ch]]
        }
        top <- unlist(top)

        #Calculate jacobin column
        Jcol <- top * above * lay
      }
    }
    return(Jcol)
  }
  #Bind on the top layer such that the parametric terms are at the front
  toplayer <- obj$hidden_layers[[length(obj$hidden_layers)]]
  scaled_parametric <- toplayer[,grepl('param', colnames(toplayer))]
  Jacobian <- cbind(scaled_parametric, Jacobian_ab)
  return(Jacobian)
}






