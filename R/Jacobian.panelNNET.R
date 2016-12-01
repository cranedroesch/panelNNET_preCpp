Jacobian.panelNNET <-
function(obj){
  if (obj$activation == 'tanh'){
    sigma <- tanh
    sigma_prime <- tanh_prime
    sigma_prime2 <- tanh_prime2
  }
  if (obj$activation == 'logistic'){
    sigma <- logistic
    sigma_prime <- logistic_prime
    sigma_prime2 <- logistic_prime2
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
  par.df$lev <- c(lev, rep('beta', length(obj$parlist$beta)), rep('beta_param', length(obj$parlist$beta_param)))
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

  #Lev is the level that the parameter emanates from
  #lev.pointer is the column of the level that the parameter emanates from
  #upper.lev.pointer is the column of the upper level that the parameter goes to
  #Note that the bias terms are not included in the saved hidden layers, nor in the X.  So sometimes the number one needs to be added or subtracted from the index

  Jacobian_ab <- foreach(i = 1:nrow(par.df[!grepl('beta', par.df$lev),]), .combine = cbind) %do% {
#  i=1+i
#  par.df[i,]
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
      if (class(obj$parlist[[as.numeric(par.df$lev[i])+1]]) == 'numeric'){
        above.pars <- obj$parlist$beta[as.numeric(par.df$upper.lev.pointer[i])]
        above.layer <- obj$hidden_layers[[as.numeric(par.df$lev[i])]][,as.numeric(par.df$upper.lev.pointer[i])+ncol(obj$param)]
      } else {
        above.pars <- obj$parlist[[as.numeric(par.df$lev[i])+1]][as.numeric(par.df$upper.lev.pointer[i]),]
        above.layer <- obj$hidden_layers[[as.numeric(par.df$lev[i])]][,as.numeric(par.df$upper.lev.pointer[i])-1]
      }
      above <- rowSums(as.matrix(sigma_prime2(above.layer)) %*% t(as.matrix(above.pars)))
      Jcol <- Jcol * above
      if (as.numeric(par.df$lev[i]) < length(lowerpars) & !is.na(as.numeric(par.df$lev[i]))){
        #After the top layer, matrix expressions work again
        #If at top layer, make sure not to include parametric terms
        top <- sigma_prime2(obj$hidden_layers[[as.numeric(par.df$lev[i]) + 1 ]][, (length(obj$parlist$beta_param)+1):(length(obj$parlist$beta)+length(obj$parlist$beta_param))]) %*% obj$parlist$beta
        #Calculate jacobin column
        Jcol <- top * above * lay
        if (as.numeric(par.df$lev[i]) > length(lowerpars)){stop("not implemented for 3+ layer networks")}
      }
    }
    return(Jcol)
  }
  #Bind on the top layer such that the parametric terms are at the front
  Jacobian <- cbind(obj$hidden_layers[[2]][,c(
      which(colnames(obj$hidden_layers[[2]]) %ni% colnames(obj$param))
    , which(colnames(obj$hidden_layers[[2]]) %in% colnames(obj$param))
  )], Jacobian_ab)
  return(Jacobian)
}
