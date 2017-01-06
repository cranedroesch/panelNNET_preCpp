panelNNET.est <-
function(y, X, hidden_units, fe_var, maxit, lam, time_var, param, parapen, parlist, verbose, save_each_iter, path, tag, gravity, convtol, bias_hlayers, RMSprop, start.LR, activation, inference, doscale, treatment, interact_treatment, batchsize, maxstopcounter, ...){

#y = y[r]
#X = Z[r,]
#hidden_units = c(20,20)

#fe_var = id[r]
#maxit = 1000
#lam = lam

#time_var = time[r]
#param = P
#parlist = pl
#verbose = FALSE

#save_each_iter = FALSE
#path = NULL
#tag = ""
#gravity = 1.01
#bias_hlayers = TRUE
# RMSprop = TRUE
#convtol = 1e-8
#activation = 'tanh'
#doscale = TRUE
#inference = FALSE
#batchsize = 50
#parapen = rep(0, ncol(P))
#treatment = NULL
#start.LR = .01

  if (doscale == TRUE){
    X <- scale(X)
    if (!is.null(param)){
      param <- scale(param)
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
      warning('WARNING: panelNNET for heterogeneous treatment effects is still highly experimental.  the gradient descent algorithm appears to suffer badly from local minima and the standard errors of the treatment effects appear to be extremely buggy.')
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
    hlayers[[i]] <- cbind(treatment, hlayers[[i]])
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
    fe <- (y-ydm) - as.matrix(hlayers[[i]]-Zdm) %*% as.matrix( c(parlist$beta_param, parlist$beta_treatment, parlist$beta_treatmentinteractions, parlist$beta))
    yhat <- hlayers[[i]] %*% c(parlist$beta_param, parlist$beta_treatment, parlist$beta_treatmentinteractions, parlist$beta) + fe    
  } else {
    yhat <- hlayers[[i]] %*% c(parlist$beta_param, parlist$beta_treatment, parlist$beta_treatmentinteractions, parlist$beta)
  }
  mse <- mseold <- mean((y-yhat)^2)
  loss <- lossold <- mse + lam*sum(c(parlist$beta_param*parapen, 0*parlist$beta_treatment, parlist$beta, parlist$beta_treatmentinteractions, unlist(parlist[!grepl('beta', names(parlist))]))^2)
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
  LRvec <- LR <- start.LR#starting LR
  D <- 1e6
  stopcounter <- iter <- 0
  msevec <- lossvec <- c()
  ###############
  #start iterating
  while(iter<maxit & stopcounter < maxstopcounter){
    oldpar <- list(parlist=parlist, hlayers=hlayers, grads=grads
      , yhat = yhat, mse = mse, mseold = mseold, loss = loss, lossold = lossold, updates = updates, G2 = G2)
    #Start epoch
    #Assign batches
    batchid <- sample(1:nrow(X)%/%batchsize +1)
    if (min(table(batchid))<(batchsize/2)){#Deal with orphan batches
      batchid[batchid == max(batchid)] <- sample(1:(max(batchid) - 1), min(table(batchid)))
    }
    for (bat in 1:max(batchid)) {
      curBat <- which(batchid == bat)
      #Get updated gradients
      grads <- vector('list', nlayers+1)
      grads[[length(grads)]] <- getDelta(y[curBat], yhat[curBat])
      for (i in (nlayers):1){
        if (i == nlayers){outer_param = as.matrix(c(parlist$beta))} else {outer_param = parlist[[i+1]]}
        if (i == 1){lay = X[curBat,]} else {lay= hlayers[[i-1]][curBat,]}
        if (bias_hlayers == TRUE){
          lay <- cbind(1, lay) #add bias to the hidden layer
          if (i!=nlayers){outer_param <- outer_param[-1,]}      #remove parameter on upper-layer bias term
        }
        grads[[i]] <- getS(D_layer = lay, inner_param = parlist[[i]], outer_deriv = grads[[i+1]], outer_param = outer_param, activation)
      }
      #Calculate updates to parameters based on gradients and learning rates
      if (RMSprop == TRUE){
        newG2 <- foreach(i = 1:(length(hlayers)+1)) %do% {
          if (i == 1){D <- X[curBat,]} else {D <- hlayers[[i-1]][curBat,]}
          if (bias_hlayers == TRUE & i != length(hlayers)+1){D <- cbind(1, D)}
            .1*(t(D) %*% grads[[i]])^2
        }
        oldG2 <- lapply(G2, function(x){.9*x})
        G2 <- mapply('+', newG2, oldG2)
        uB <- LR/sqrt(G2[[length(G2)]]+1e-10) *
          t(t(grads[[length(grads)]]) %*% hlayers[[length(hlayers)]][curBat,]) + 
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
          if(i == 1){lay = X[curBat,]} else {lay = hlayers[[i-1]][curBat,]}
          if(bias_hlayers == TRUE){lay <- cbind(1,lay)}
          updates[[i]] <- LR/sqrt(G2[[i]]+1e-10) * t(t(grads[[i]]) %*% lay) + LR*t(2 * lam * t(parlist[[i]]))
        }
      } else { #if RMSprop == FALSE
        uB <- LR * t(t(grads[[length(grads)]]) %*% hlayers[[length(hlayers)]][curBat,] +
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
          if(i == 1){lay = X[curBat,]} else {lay = hlayers[[i-1]][curBat,]}
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
        hlayers[[i]] <- cbind(treatment, hlayers[[i]])
        colnames(hlayers[[i]])[1] <- 'treatment'
      }
      if (!is.null(param)){
        hlayers[[i]] <- cbind(param, hlayers[[i]])
        colnames(hlayers[[i]])[1:ncol(param)] <- paste0('param',1:ncol(param))
      }
      if (is.null(fe_var)){hlayers[[i]] <- cbind(1, hlayers[[i]])}#add intercept if no FEs

        if (!is.null(fe_var)){
          Zdm <- demeanlist(hlayers[[i]], list(fe_var))
          fe <- (y-ydm) - as.matrix(hlayers[[i]]-Zdm) %*% as.matrix(c(parlist$beta_param, parlist$beta_treatment, parlist$beta_treatmentinteractions, parlist$beta))
          yhat <- hlayers[[i]] %*% c(
            parlist$beta_param, parlist$beta_treatment, parlist$beta_treatmentinteractions, parlist$beta
          ) + fe    
        } else {
          yhat <- hlayers[[i]] %*% c(parlist$beta_param, parlist$beta_treatment, parlist$beta_treatmentinteractions, parlist$beta)
        }
      mse <- mean((y-yhat)^2)
      msevec <- append(msevec, mse)
      loss <- mse + lam*sum(c(parlist$beta_param*parapen, 0*parlist$beta_treatment, parlist$beta, parlist$beta_treatmentinteractions, unlist(parlist[!grepl('beta', names(parlist))]))^2)
      lossvec <- append(lossvec, loss)
      if (verbose == TRUE){
        writeLines(paste0(
            "*******************************************\n"
          , tag, "\n"
          , 'Lambda = ',lam, "\n"
          , "Hidden units -> ",paste(hidden_units, collapse = ' '), "\n"
          , " Batch size is ", batchsize, " \n"
          , " Completed ", iter, " epochs. \n"
          , " Completed ", bat, " batches in current epoch. \n"
          , "mse is ",mse, "\n"
          , "last mse was ", oldpar$mse, "\n"
          , "difference is ", oldpar$mse - mse, "\n"
          , "loss is ",loss, "\n"
          , "last loss was ", oldpar$loss, "\n"
          , "difference is ", oldpar$loss - loss, "\n"
          , "*******************************************\n"
        ))
        par(mfrow = c(3,2))
        plot(y, yhat, col = rgb(1,0,0,.5), pch = 19, main = 'in-sample performance')
        abline(0,1)
        plot(LRvec, type = 'b', main = 'learning rate history')
        plot(msevec, type = 'l', main = 'all epochs')
        plot(msevec[(1+(iter)*max(batchid)):length(msevec)], type = 'l', ylab = 'mse', main = 'Current epoch')
        plot(lossvec, type = 'l', main = 'all epochs')
        plot(lossvec[(1+(iter)*max(batchid)):length(lossvec)], type = 'l', ylab = 'loss', main = 'Current epoch')
      }
    }
    #Finished epoch.  Assess whether MSE has increased and revert if so
    mse <- mean((y-yhat)^2)
    loss <- mse + lam*sum(c(parlist$beta_param*parapen, 0*parlist$beta_treatment, parlist$beta, parlist$beta_treatmentinteractions, unlist(parlist[!grepl('beta', names(parlist))]))^2)
    #If loss increases...
    if (oldpar$loss < loss){
      parlist <- oldpar$parlist
      updates <- oldpar$updates
      G2 <- oldpar$G2
      hlayers <- oldpar$hlayers
      grads <- oldpar$grads
      yhat <- oldpar$yhat
      mse <- oldpar$mse
      mseold <- oldpar$mseold
      loss <- oldpar$loss
      lossold <- oldpar$lossold
      stopcounter <- stopcounter + 1
      LR <- LR/2
      if(verbose == TRUE){
        print(paste0("MSE increased.  halving LR.  Stopcounter now at ", stopcounter))
      }
    } else {
      LRvec[iter+1] <- LR <- LR*gravity      #gravity...
      if (save_each_iter == TRUE){
        save(parlist, file = paste0(path, '/pnnet_int_out_',tag,'_'))  #Save the intermediate output locally
      }
      D <- lossold - loss
      if (D<convtol){
        stopcounter <- stopcounter +1
        if(verbose == TRUE){print(paste('slowing!  Stopcounter now at ', stopcounter))}
      }else{
        stopcounter <-0
      }
    }
    iter = iter+1
  } #close the while loop


  conv <- (iter<maxit)
  if(is.null(fe_var)){
    fe_output <- NULL
  } else {
    fe_output <- data.frame(fe_var, fe)
  }
  output <- list(yhat = yhat, parlist = parlist, hidden_layers = hlayers
    , fe = fe_output, converged = conv, mse = mse, loss = loss, lam = lam, time_var = time_var
    , X = X, y = y, param = param, fe_var = fe_var, hidden_units = hidden_units, maxit = maxit
    , used_bias = bias_hlayers, final_improvement = D, msevec = msevec, RMSprop = RMSprop, convtol = convtol
    , grads = grads, activation = activation, parapen = parapen, doscale = doscale, treatment = treatment
    , interact_treatment = interact_treatment, batchsize = batchsize
  )
  if(inference == TRUE){
    J <- Jacobian.panelNNET(output)
    X <- output$hidden_layers[[length(output$hidden_layers)]]
    vc.JacHomo = tryCatch(vcov.panelNNET(output, 'Jacobian_homoskedastic', J = J), error = function(e)e, finally = NULL)
    vc.JacSand = tryCatch(vcov.panelNNET(output, 'Jacobian_sandwich', J = J), error = function(e)e, finally = NULL)    
    vc.OLSHomo = tryCatch(vcov.panelNNET(output, 'OLS', J = X), error = function(e)e, finally = NULL)
    vc.OLSSand = tryCatch(vcov.panelNNET(output, 'sandwich', J = X), error = function(e)e, finally = NULL)
    if (!is.null(fe_var)){
      vc.JacClus = tryCatch(vcov.panelNNET(output, 'Jacobian_cluster', J = J), error = function(e)e, finally = NULL)
      vc.OLSClus = tryCatch(vcov.panelNNET(output, 'cluster', J = X), error = function(e)e, finally = NULL)
      vcs <- list(vc.JacHomo = vc.JacHomo, vc.JacSand = vc.JacSand , vc.JacClus = vc.JacClus , vc.OLSHomo = vc.OLSHomo, vc.OLSSand = vc.OLSSand, vc.OLSClus = vc.OLSClus)
    } else {
      vcs <- list(vc.JacHomo = vc.JacHomo, vc.JacSand = vc.JacSand, vc.OLSHomo = vc.OLSHomo, vc.OLSSand = vc.OLSSand)
    }
    output$vcs <- vcs
  }
  return(output)
}
