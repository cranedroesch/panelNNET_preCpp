panelNNET.est <-
function(y, X, hidden_units, fe_var, maxit, lam, time_var, param, parapen, parlist, verbose, save_each_iter, path, tag, gravity, convtol, bias_hlayers, RMSprop, start.LR, activation, inference, doscale, treatment, interact_treatment, batchsize, maxstopcounter, OLStrick, useOptim, optimMethod, ...){
##examplearguments for testing

#library(panelNNET)
#N <- 1000
#x <- sort(runif(N, 0, 20))
#time <- (1:N-1)%%20+1
#id <- (1:N-1)%/%20+1
#y <- id + time + x*sin(x) + rnorm(N, sd = 3)
#plot(x, y)
###y = y[r]
#X = matrix(x)
#fe_var = factor(id)
#time_var = time
#param = matrix(time)

#lam = .00001
#maxit = 1000
#hidden_units = c(50,50)
#parlist = NULL
#verbose = TRUE
#OLStrick = FALSE
#save_each_iter = FALSE
#path = NULL
#tag = ""
#gravity = 1.01
#bias_hlayers = TRUE
#RMSprop = TRUE
#convtol = 1e-8
#activation = 'tanh'
#doscale = TRUE
#inference = FALSE
#batchsize = nrow(X)
#parapen = rep(1, ncol(param))
#treatment = NULL
#start.LR = .01
#maxstopcounter = 10
#batchsize = 100
#useOptim = FALSE
#optimMethod = 'BFGS'

getYhat <- function(pl, skel = attr(pl, 'skeleton'), hlay = NULL){ 
#print((pl))
  plist <- relist(pl, skel)
  #Update hidden layers
  if (is.null(hlay)){hlay <- calc_hlayers(plist)}
  #update yhat
  if (!is.null(fe_var)){
    Zdm <- demeanlist(hlay[[length(hlay)]], list(fe_var))
    if (OLStrick == TRUE){#OLS trick!
      lamvec <- rep(lam, ncol(Zdm))
      if (is.null(fe_var)){
        pp <- c(0, parapen) #never penalize the intercept
      } else {
        pp <- parapen #parapen
      }
      lamvec[1:length(pp)] <- lamvec[1:length(pp)]*pp #incorporate parapen into diagonal of covmat
      B <- solve(t(Zdm) %*% Zdm + diag(lamvec)) %*% t(Zdm) %*% ydm
      plist$beta <- B[grepl('nodes', rownames(B))]
      plist$beta_param <- B[grepl('param', rownames(B))]
      if (!is.null(treatment)){
        plist$beta_treatment <- B[grepl('treatment', rownames(B))]
        plist$beta_treatmentinteractions <- B[grepl('TrInts', rownames(B))]
      }
    }
    fe <- (y-ydm) - as.matrix(hlay[[length(hlay)]]-Zdm) %*% as.matrix(c(
        plist$beta_param, plist$beta_treatment
      , plist$beta_treatmentinteractions, plist$beta
    ))
    yhat <- hlay[[length(hlay)]] %*% c(
      plist$beta_param, plist$beta_treatment, plist$beta_treatmentinteractions, plist$beta
    ) + fe    
  } else {
    yhat <- hlay[[length(hlay)]] %*% c(plist$beta_param, plist$beta_treatment, plist$beta_treatmentinteractions, plist$beta)
  }
  return(yhat)
}

lossfun <- function(pl, skel){
  yhat <- getYhat(pl, skel)
  mse <- mean((y-yhat)^2)
  plist <- relist(pl, skel)
  loss <- mse + lam*sum(c(plist$beta_param*parapen, 0*plist$beta_treatment, plist$beta, plist$beta_treatmentinteractions, unlist(plist[!grepl('beta', names(plist))]))^2)
  return(loss)
}

calc_hlayers <- function(parlist){
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
  if (!is.null(param)){#Add parametric terms to top layer
    hlayers[[i]] <- cbind(param, hlayers[[i]])
    colnames(hlayers[[i]])[1:ncol(param)] <- paste0('param',1:ncol(param))
  }
  if (is.null(fe_var)){
    hlayers[[i]] <- cbind(1, hlayers[[i]])#add intercept if no FEs
  }
  return(hlayers)
}


calc_grads<- function(plist, hlay = NULL, yhat = NULL, curBat = NULL){
  if (!is.null(curBat)){CB <- function(x){x[curBat,,drop = FALSE]}} else {CB <- function(x){x}}
  if (is.null(hlay)){hlay <- calc_hlayers(plist)}
  if (is.null(yhat)){yhat <- getYhat(unlist(plist), hlay = hlay)}
  grads <- vector('list', nlayers+1)
  grads[[length(grads)]] <- getDelta(CB(as.matrix(y)), yhat)
  for (i in (nlayers):1){
    if (i == nlayers){outer_param = as.matrix(c(plist$beta))} else {outer_param = plist[[i+1]]}
    if (i == 1){lay = CB(X)} else {lay= hlay[[i-1]]}
    if (bias_hlayers == TRUE){
      lay <- cbind(1, lay) #add bias to the hidden layer
      if (i!=nlayers){outer_param <- outer_param[-1,]}      #remove parameter on upper-layer bias term
    }
    grads[[i]] <- getS(D_layer = lay, inner_param = plist[[i]], outer_deriv = grads[[i+1]], outer_param = outer_param, activation)
  }
  return(grads)
}


getgr <- function(pl, skel = attr(pl, 'skeleton')){
  plist <- relist(pl, skel)
  #calculate hidden layers
  hlayers <- calc_hlayers(plist)
  #calculate gradients
  grads <- calc_grads(plist, hlay = hlayers)
  gr <- foreach(i = 1:(length(hlayers)+1)) %do% {
    if (i == 1){D <- X} else {D <- hlayers[[i-1]]}
    if (bias_hlayers == TRUE & i != length(hlayers)+1){D <- cbind(1, D)}
      (t(D) %*% grads[[i]])
  }
  return(unlist(gr))
}

  if (doscale == TRUE){
    X <- scale(X)
    if (!is.null(param)){
      param <- scale(param)
    }
  }
  if (activation == 'tanh'){
    sigma <- tanh
    sigma_prime <- tanh_prime
  }
  if (activation == 'logistic'){
    sigma <- logistic
    sigma_prime <- logistic_prime
  }
  if (activation == 'relu'){
    sigma <- relu
    sigma_prime <- relu_prime
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
    if (is.null(param)){
      parlist$beta_param <-  NULL
    } else {
      parlist$beta_param <- runif(ncol(param), -.7, .7)
    }
    parlist$beta <- runif(hidden_units[i], -.7, .7)
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
  parlist <- as.relistable(parlist)
  pl <- unlist(parlist) 
  #calculate hidden layers
  hlayers <- calc_hlayers(parlist)
  #calculate ydm and put it in global...
  if (!is.null(fe_var)){
    ydm <<- demeanlist(y, list(fe_var)) 
  }
  ###############
  #Optim approach
  if (useOptim == TRUE){
    #start optimizer
    out <- optim(par = pl, fn = lossfun, gr = getgr
      , control = list(trace  =verbose*6, maxit = 10000)
      , method = optimMethod, skel = attr(pl, 'skeleton')
    )
    parlist <- relist(out$par)  
    #Update hidden layers
    hlayers <- calc_hlayers(parlist)
    #update yhat
    yhat <- getYhat(out$par, hlay = hlayers)
    #calc fixed effects
    if (!is.null(fe_var)){
      Zdm <- demeanlist(hlayers[[i]], list(fe_var))
      fe <- (y-ydm) - as.matrix(hlayers[[i]]-Zdm) %*% as.matrix(c(
          parlist$beta_param, parlist$beta_treatment
        , parlist$beta_treatmentinteractions, parlist$beta
      ))
      fe_output <- data.frame(fe_var, fe)
    } else {
      fe_output <- NULL
    }
    #pars for output
    mse <- mean((y-yhat)^2)
    conv <- out$convergence == 0
    loss <- out$value
    grads <- msevec <- NULL
##################################
  } else { #if useOptim  == FALSE
  #get starting MSE
    yhat <- getYhat(pl, hlay = hlayers)
    mse <- mseold <- mean((y-yhat)^2)
    loss <- mse + lam*sum(c(parlist$beta_param*parapen, 0*parlist$beta_treatment, parlist$beta, parlist$beta_treatmentinteractions, unlist(parlist[!grepl('beta', names(parlist))]))^2)
    #Calculate gradients.  These aren't the actual gradients, but become the gradients when multiplied by their respective layer.
    grads <- calc_grads(parlist, hlayers, yhat)
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
        , yhat = yhat, mse = mse, mseold = mseold, loss = loss, updates = updates, G2 = G2)
      #Start epoch
      #Assign batches
      batchid <- sample(1:nrow(X)%/%batchsize +1)
      if (min(table(batchid))<(batchsize/2)){#Deal with orphan batches
        batchid[batchid == max(batchid)] <- sample(1:(max(batchid) - 1), min(table(batchid)), replace = TRUE)
      }
      for (bat in 1:max(batchid)) {
        curBat <- which(batchid == bat)
        #Get updated gradients
        grads <- calc_grads(parlist, lapply(hlayers, function(x){x[curBat,]}), yhat[curBat], curBat = curBat)
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
        parlist <- as.relistable(mapply('-', parlist, updates))
        pl <- unlist(parlist)
        #Update hidden layers
        hlayers <- calc_hlayers(parlist)
        #update yhat
        yhat <- getYhat(pl, attr(pl, 'skeleton'), hlay = hlayers)
        mse <- mean((y-yhat)^2)
        msevec <- append(msevec, mse)
        loss <- mse + lam*sum(c(parlist$beta_param*parapen, 0*parlist$beta_treatment, parlist$beta, parlist$beta_treatmentinteractions, unlist(parlist[!grepl('beta', names(parlist))]))^2)
        lossvec <- append(lossvec, loss)
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
        stopcounter <- stopcounter + 1
        loss <- oldpar$loss
        LR <- LR/2
        if(verbose == TRUE){
          print(paste0("Loss increased.  halving LR.  Stopcounter now at ", stopcounter))
        }
      } else {
        LRvec[iter+1] <- LR <- LR*gravity      #gravity...
        if (save_each_iter == TRUE){
          save(parlist, file = paste0(path, '/pnnet_int_out_',tag,'_'))  #Save the intermediate output locally
        }
        D <- oldpar$loss - loss
        if (D<convtol){
          stopcounter <- stopcounter +1
          if(verbose == TRUE){print(paste('slowing!  Stopcounter now at ', stopcounter))}
        }else{
          stopcounter <-0
        }
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
      iter = iter+1
    } #close the while loop
    conv <- (iter<maxit)
    if(is.null(fe_var)){
      fe_output <- NULL
    } else {
      Zdm <- demeanlist(hlayers[[length(hlayers)]], list(fe_var))
      fe <- (y-ydm) - as.matrix(hlayers[[length(hlayers)]]-Zdm) %*% as.matrix(c(
          parlist$beta_param, parlist$beta_treatment
        , parlist$beta_treatmentinteractions, parlist$beta
      ))
      fe_output <- data.frame(fe_var, fe)
    }
  } #ifelse optim or not
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


