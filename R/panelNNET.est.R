panelNNET.est <-
function(y, X, hidden_units, fe_var, maxit = 1000, lam = 0, time_var = NULL, param = NULL, parapen = rep(0, ncol(param)), parlist = NULL, verbose = FALSE, save_each_iter = FALSE, path = NULL, tag = "", gravity = 1.01, convtol = 1e-8, bias_hlayers = TRUE, RMSprop = FALSE, start.LR = .01, activation = 'tanh', inference = TRUE, doscale = TRUE, treatment = NULL, interact_treatment = TRUE){
#X <- x
#hidden_units = 10
#fe_var = NULL
#lam = 0
#time_var = NULL
#bias_hlayers = TRUE
#gravity = 1.01
#maxit = 10000
#save_each_iter = FALSE
#path = NULL
#verbose = TRUE
#parlist = NULL
#convtol = 1e-8
#tag = ''
#RMSprop = TRUE
#start.LR = .01
#param = matrix(time)
#parapen = rep(0, ncol(param))
#activation = 'tanh'
#doscale = TRUE
#treatment = treatment
#interact_treatment = TRUE
  if (doscale == TRUE){
    X <- scale(X)
    if (!is.null(param)){
      param <- scale(param)
    }
    if (!is.null(treatment)){
      scaled.treatment <- scale(treatment)
    }    
  }
  if (activation == 'tanh'){
    sigma <- tanh
    sigma_prime <- tanh_prime
    sigma_prime2 <- tanh_prime2
  }
  if (activation == 'logistic'){
    sigma <- logistic
    sigma_prime <- logistic_prime
    sigma_prime2 <- logistic_prime2
  }
  if (!is.null(path)){
    fi <- list.files(path, pattern = tag)
    if(length(fi) > 1){stop('borked tags!')}
    if(length(fi) == 1){
      fn <- paste0(path, '/', fi)
      fn <- gsub('//', '/', fn)#be careful about double slashes...
      parlist <- load_obj(fn)
      print('picked up where left off!')
    }
  }
  nlayers <- length(hidden_units)
  #get starting weights, either randomly or from a specified parlist
  if (is.null(parlist)){#random starting weights
    parlist <- vector('list', nlayers)
    for (i in 1:nlayers){
      if (i == 1){D <- ncol(X)} else {D <- hidden_units[i-1]}
      parlist[[i]] <- matrix(runif((hidden_units[i])*(D+bias_hlayers), -.7, .7), ncol = hidden_units[i])
    }
    parlist$beta <- runif(hidden_units[i], -.7, .7)
    if (is.null(param)){
      parlist$beta_param <-  NULL
    } else {
      parlist$beta_param <- runif(ncol(param), -.7, .7)
    }
    #Add the treatment effect and the interaction of the treatment with the derived variables
    if (!is.null(treatment)){
      parlist$beta_treatment <- runif(1, -.7, .7)
      if (interact_treatment == TRUE){
        parlist$beta_treatmentinteractions <- runif(hidden_units[i], -.7, .7)
      }
    }
    #add the bias term/intercept onto the front, if there are no FE's
    parlist$beta_param <- c(runif(is.null(fe_var)), parlist$beta_param)
    #if there are no FE's, append a 0 to the front of the parapen vec, to leave the intercept unpenalized
    if(is.null(fe_var)){
      parapen <- c(0, parapen)
    }
  } 
  #calculate hidden layers
  hlayers <- vector('list', nlayers)
  for (i in 1:nlayers){
    if (i == 1){D <- X} else {D <- hlayers[[i-1]]}
    if (bias_hlayers == TRUE){D <- cbind(1, D)}
    hlayers[[i]] <- sigma(D %*% parlist[[i]])
  }
  colnames(hlayers[[i]]) <- paste0('nodes',1:ncol(hlayers[[i]]))
  if (!is.null(treatment)){
    #Add treatment interactions
    if (interact_treatment == TRUE){
      ints <- sweep(hlayers[[i]], 1, treatment, '*')
      colnames(ints) <- paste0('TrInts',1:ncol(ints))
      hlayers[[i]] <- cbind(ints, hlayers[[i]])
    }
    #Add treatment dummy
    hlayers[[i]] <- cbind(scaled.treatment, hlayers[[i]])
    colnames(hlayers[[i]])[1] <- 'treatment'
  }
  if (!is.null(param)){
    hlayers[[i]] <- cbind(param, hlayers[[i]])
    colnames(hlayers[[i]])[1:ncol(param)] <- paste0('param',1:ncol(param))
  }
  if (is.null(fe_var)){hlayers[[i]] <- cbind(1, hlayers[[i]])}#add intercept if no FEs
  #get starting MSE
  if (!is.null(fe_var)){
    Zdm <- demeanlist(hlayers[[i]], list(fe_var))
    ydm <<- demeanlist(y, list(fe_var))
    fe <- (y-ydm) - as.matrix(hlayers[[i]]-Zdm) %*% as.matrix(with(parlist, c(beta_param, beta_treatment, beta_treatmentinteractions, beta)))
    yhat <- hlayers[[i]] %*% with(parlist, c(beta_param, beta_treatment, beta_treatmentinteractions, beta)) + fe    
  } else {
    yhat <- hlayers[[i]] %*% with(parlist, c(beta_param, beta_treatment, beta_treatmentinteractions, beta))
  }
  mse <- mseold <- mean((y-yhat)^2)
  #Calculate gradients.  These aren't the actual gradients, but become the gradients when multiplied by their respective layer.
  grads <- vector('list', nlayers+1)
  grads[[length(grads)]] <- getDelta(y, yhat)
  for (i in (nlayers):1){
    if (i == nlayers){outer_param = as.matrix(c(parlist$beta))} else {outer_param = parlist[[i+1]]}
    if (i == 1){lay = X} else {lay= hlayers[[i-1]]}
    if (bias_hlayers == TRUE){
      lay <- cbind(1, lay) #add bias to the hidden layer
      if (i!=nlayers){outer_param <- outer_param[-1,]}      #remove parameter on upper-layer bias term
    }
    grads[[i]] <- getS(D_layer = lay, inner_param = parlist[[i]], outer_deriv = grads[[i+1]], outer_param = outer_param, activation)
  }
  #Initialize updates
  updates <- lapply(parlist, function(x){x*0})
  #initialize G2 term for RMSprop
  if (RMSprop == TRUE){
    #Prior gradients are zero at first iteration...
    G2 <- lapply(parlist, function(x){x*0})
    #squashing all of the numeric list elements into a matrix/vector
    betas <- matrix(unlist(G2[grepl('beta', names(G2))]))
    G2 <- G2[!grepl('beta', names(G2))]    
    G2[[length(G2)+1]] <- betas
  }
  LR <- start.LR#starting LR
  D <- 1e6
  stopcounter <- iter <- 0
  LRvec <- msevec <- c()
  ###############
  #start iterating
  while(iter<maxit & stopcounter < 10){
    oldpar <- list(parlist=parlist, hlayers=hlayers, grads=grads, yhat = yhat, mse = mse, mseold = mseold, updates = updates, G2 = G2)
    #Get updated gradients
    grads <- vector('list', nlayers+1)
    grads[[length(grads)]] <- getDelta(y, yhat)
    for (i in (nlayers):1){
      if (i == nlayers){outer_param = as.matrix(c(parlist$beta))} else {outer_param = parlist[[i+1]]}
      if (i == 1){lay = X} else {lay= hlayers[[i-1]]}
      if (bias_hlayers == TRUE){
        lay <- cbind(1, lay) #add bias to the hidden layer
        if (i!=nlayers){outer_param <- outer_param[-1,]}      #remove parameter on upper-layer bias term
      }
      grads[[i]] <- getS(D_layer = lay, inner_param = parlist[[i]], outer_deriv = grads[[i+1]], outer_param = outer_param, activation)
    }
    #Calculate updates to parameters based on gradients and learning rates
    if (RMSprop == TRUE){
      newG2 <- foreach(i = 1:(length(hlayers)+1)) %do% {
        if (i == 1){D <- X} else {D <- hlayers[[i-1]]}
        if (bias_hlayers == TRUE & i != length(hlayers)+1){D <- cbind(1, D)}
          .1*(t(D) %*% grads[[i]])^2
      }
      oldG2 <- lapply(G2, function(x){.9*x})
      G2 <- mapply('+', newG2, oldG2)
      uB <- LR/sqrt(G2[[length(G2)]]+1e-10) *
        t(t(grads[[length(grads)]]) %*% hlayers[[length(hlayers)]]) + 
        LR*as.matrix(2*lam*c(parlist$beta_param*parapen, 0*parlist$beta_treatment, parlist$beta, parlist$beta_treatmentinteractions))#Treatment is always unpenalized
      updates$beta_param <- uB[1:length(parlist$beta_param)]
      updates$beta <- uB[grepl('nodes', rownames(uB))]
      if (!is.null(treatment)){
        updates$beta_treatment <- uB[rownames(uB) == 'treatment']
        if (interact_treatment == TRUE){
          updates$beta_treatmentinteractions <- uB[grepl('TrInts', rownames(uB))]
        }
      }
      for(i in nlayers:1){
        if(i == 1){lay = X} else {lay = hlayers[[i-1]]}
        if(bias_hlayers == TRUE){lay <- cbind(1,lay)}
        updates[[i]] <- LR/sqrt(G2[[i]]+1e-10) * t(t(grads[[i]]) %*% lay) + LR*t(2 * lam * t(parlist[[i]]))
      }
    } else {
      uB <- LR * t(t(grads[[length(grads)]]) %*% hlayers[[length(hlayers)]] +
        2*lam*c(parlist$beta_param*parapen, 0*parlist$beta_treatment, parlist$beta, parlist$beta_treatmentinteractions))
      updates$beta_param <- uB[1:length(parlist$beta_param)]
      updates$beta <- uB[grepl('nodes', rownames(uB))]
      if (!is.null(treatment)){
        updates$beta_treatment <- uB[rownames(uB) == 'treatment']
        if (interact_treatment == TRUE){
          updates$beta_treatmentinteractions <- uB[grepl('TrInts', rownames(uB))]
        }
      }
      for(i in nlayers:1){
        if(i == 1){lay = X} else {lay = hlayers[[i-1]]}
        if(bias_hlayers == TRUE){lay <- cbind(1,lay)}
        updates[[i]] <- t(LR * t(grads[[i]]) %*% lay + 2 * lam * t(parlist[[i]]))
      }
    }
    #Update parameters from update list
    parlist <- mapply('-', parlist, updates)
    #Update hidden layers
    for (i in 1:nlayers){
      if (i == 1){D <- X} else {D <- hlayers[[i-1]]}
      if (bias_hlayers == TRUE){D <- cbind(1, D)}
      hlayers[[i]] <- sigma(D %*% parlist[[i]])
    }
    colnames(hlayers[[i]]) <- paste0('nodes',1:ncol(hlayers[[i]]))
    if (!is.null(treatment)){
      #Add treatment interactions
      if (interact_treatment == TRUE){
        ints <- sweep(hlayers[[i]], 1, treatment, '*')
        colnames(ints) <- paste0('TrInts',1:ncol(ints))
        hlayers[[i]] <- cbind(ints, hlayers[[i]])
      }
      #Add treatment dummy
      hlayers[[i]] <- cbind(scaled.treatment, hlayers[[i]])
      colnames(hlayers[[i]])[1] <- 'treatment'
    }
    if (!is.null(param)){
      hlayers[[i]] <- cbind(param, hlayers[[i]])
      colnames(hlayers[[i]])[1:ncol(param)] <- paste0('param',1:ncol(param))
    }
    if (is.null(fe_var)){hlayers[[i]] <- cbind(1, hlayers[[i]])}#add intercept if no FEs
    #recalc MSE
    if (!is.null(fe_var)){
      Zdm <- demeanlist(hlayers[[i]], list(fe_var))
      ydm <- demeanlist(y, list(fe_var))
      fe <- (y-ydm) - as.matrix(hlayers[[i]]-Zdm) %*% as.matrix(with(parlist, c(beta_param, beta_treatment, beta_treatmentinteractions, beta)))
    yhat <- hlayers[[i]] %*% with(parlist, c(beta_param, beta_treatment, beta_treatmentinteractions, beta)) + fe    
  } else {
    yhat <- hlayers[[i]] %*% with(parlist, c(beta_param, beta_treatment, beta_treatmentinteractions, beta))
    }
    mseold <- mse
    mse <- mean((y-yhat)^2)
    #If MSE increases...
    if (mseold - mse < 0){
      parlist <- oldpar$parlist
      updates <- oldpar$updates
      G2 <- oldpar$G2
      hlayers <- oldpar$hlayers
      grads <- oldpar$grads
      yhat <- oldpar$yhat
      mse <- oldpar$mse
      mseold <- oldpar$mseold
      stopcounter <- stopcounter + 1
      LR <- LR/2
      if(verbose == TRUE){
        print("MSE increased.  halving LR")
        print(stopcounter)
      }
    } else {
      if (verbose == TRUE){
        writeLines(paste0(
            "*******************************************\n"
          , tag, "\n"
          , 'Lambda = ',lam, "\n"
          , "Hidden units -> ",paste(hidden_units, collapse = ' '), "\n"
          , " Completed ", iter, " iterations. \n"
          , "mse is ",mse, "\n"
          , "last mse was ", mseold, "\n"
          , "difference is ", mseold - mse, "\n"
          , "*******************************************\n"
        ))
        par(mfrow = c(1,3))
        plot(y, yhat, col = rgb(1,0,0,.5), pch = 19)
        abline(0,1)
  #      points(y, oldpar$yhat, col = rgb(0,0,1,.5), pch = 19)
        LRvec[iter+1] <- LR <- LR*gravity      #gravity...
#        stopcounter <- 0
        msevec[iter+1] <- mse
        plot(LRvec, type = 'l')
        plot(msevec, type = 'l')
      }
      LRvec[iter+1] <- LR <- LR*gravity      #gravity...
      msevec[iter+1] <- mse
      if (save_each_iter == TRUE){
        save(parlist, file = paste0(path, '/pnnet_int_out_',tag,'_'))  #Save the intermediate output locally
      }
      D <- mseold - mse
      if (D<convtol){
        stopcounter <- stopcounter +1
        if(verbose == TRUE){print(paste('slowing!', stopcounter))}
      }else{
        stopcounter <-0
      }
    }
    iter = iter+1
  }
  conv <- (iter<maxit)
  if(is.null(fe_var)){
    fe_output <- NULL
  } else {
    fe_output <- data.frame(fe_var, fe)
  }
  output <- list(yhat = yhat, parlist = parlist, hidden_layers = hlayers
    , fe = fe_output, converged = conv, mse = mse, lam = lam, time_var = time_var
    , X = X, y = y, param = param, fe_var = fe_var, hidden_units = hidden_units, maxit = maxit
    , used_bias = bias_hlayers, final_improvement = D, msevec = msevec, RMSprop = RMSprop, convtol = convtol
    , grads = grads, activation = activation, parapen = parapen, doscale = doscale, treatment = treatment
    , scaled.treatment = scaled.treatment, interact_treatment = interact_treatment
  )
  if(inference == TRUE){
    J <- Jacobian.panelNNET(output)
    X <- output$hidden_layers[[length(output$hidden_layers)]]
    vcs <- list(
        vc.JacHomo = tryCatch(vcov.panelNNET(output, 'Jacobian_homoskedastic', J = J), error = function(e)e, finally = NULL)
      , vc.JacSand = tryCatch(vcov.panelNNET(output, 'Jacobian_sandwich', J = J), error = function(e)e, finally = NULL)
      , vc.JacClus = tryCatch(vcov.panelNNET(output, 'Jacobian_cluster', J = J), error = function(e)e, finally = NULL)
      , vc.OLSHomo = tryCatch(vcov.panelNNET(output, 'OLS', J = X), error = function(e)e, finally = NULL)
      , vc.OLSSand = tryCatch(vcov.panelNNET(output, 'sandwich', J = X), error = function(e)e, finally = NULL)
      , vc.OLSClus = tryCatch(vcov.panelNNET(output, 'cluster', J = X), error = function(e)e, finally = NULL)
    )
    output$vcs <- vcs
  }
  return(output)
}
