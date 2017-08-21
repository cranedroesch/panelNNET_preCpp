panelNNET.est <-
function(y, X, hidden_units, fe_var, maxit, lam, time_var, param, parapen, parlist
         , verbose, para_plot, report_interval, gravity, convtol, bias_hlayers, RMSprop
         , start.LR, activation, doscale, treatment, interact_treatment
         , batchsize, maxstopcounter, OLStrick, initialization, dropout_hidden
         , dropout_input, test_set, ...){

# y <- y[r]
# X <- Z[r,]
# hidden_units = arch
# fe_var = id[r]
# maxit = 1000
# lam = lam
# time_var = time[r]
# param = P[r,]
# RMSprop = TRUE
# convtol = 1e-6
# activation = 'relu'
# doscale = TRUE
# inference = FALSE
# parapen = rep(1, ncol(P))#penalize the time trends
# gravity = 1.01
# OLStrick = FALSE
# initialization = 'enforce_normalization'
# parlist = NULL
# verbose = TRUE
# report_interval = 10
# test_set = list(y_test = y[v], x_test = Z[v,], fe_test = id[v], test_params = P[v, , drop = FALSE])
# bias_hlayers <- TRUE
# batchsize = nrow(X)  
# dropout_hidden <- dropout_input <- 1
# para_plot <- FALSE
# treatment <- NULL
# start.LR <- .01
  ##########
  #Define internal functions

  getYhat <- function(pl, skel = attr(pl, 'skeleton'), hlay = NULL){ 
  #print((pl))
  #pl <- parlist
  #skel = attr(pl, 'skeleton')
  #hlay <- hlayers
    plist <- relist(pl, skel)
    #Update hidden layers
    if (is.null(hlay)){hlay <- calc_hlayers(plist)}
    #update yhat
    if (!is.null(fe_var)){
      Zdm <- demeanlist(hlay[[length(hlay)]], list(fe_var))
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

  lossfun <- function(pl, skel, lam, parapen){
    yhat <- getYhat(pl, skel)
    mse <- mean((y-yhat)^2)
    plist <- relist(pl, skel)
    loss <- mse + lam*sum(c(plist$beta_param*parapen, 0*plist$beta_treatment, plist$beta, plist$beta_treatmentinteractions, unlist(plist[!grepl('beta', names(plist))]))^2)
    return(loss)
  }

  calc_hlayers <- function(parlist, normalize = FALSE){
    hlayers <- vector('list', nlayers)
    for (i in 1:nlayers){
      if (i == 1){D <- X} else {D <- hlayers[[i-1]]}
      if (bias_hlayers == TRUE){D <- cbind(1, D)}
      if (normalize == TRUE){
        hli <- activ(D %*% parlist[[i]])
        v <- sd(as.numeric(hli))
        hlayers[[i]] <- hli/v 
      } else {
        hlayers[[i]] <- activ(D %*% parlist[[i]])
      }
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

  calc_grads<- function(plist, hlay = NULL, yhat = NULL, curBat = NULL, droplist = NULL, dropinp = NULL){
    #subset the parameters and hidden layers based on the droplist
    if (!is.null(droplist)){
      Xd <- X[,dropinp, drop = FALSE]
      #drop from parameter list emanating from input
      plist[[1]] <- plist[[1]][c(TRUE,dropinp),droplist[[1]]]
      #drop from subsequent parameter matrices
      for (i in 2:nlayers){
        plist[[i]] <- plist[[i]][c(TRUE, droplist[[i-1]]), droplist[[i]], drop = FALSE]
      }
      plist$beta <- plist$beta[droplist[[nlayers]]]
    } else {Xd <- X}#for use below...  X should be safe given scope, but extra assignment is cheap here
    if (!is.null(curBat)){CB <- function(x){x[curBat,,drop = FALSE]}} else {CB <- function(x){x}}
    if (is.null(hlay)){hlay <- calc_hlayers(plist)}
    if (is.null(yhat)){yhat <- getYhat(unlist(plist), hlay = hlay)}
    #empty list of gradients, one for each hidden layer, plus one for
    grads <- vector('list', nlayers+1)
    grads[[length(grads)]] <- getDelta(CB(as.matrix(y)), yhat)
    for (i in (nlayers):1){
      if (i == nlayers){outer_param = as.matrix(c(plist$beta))} else {outer_param = plist[[i+1]]}
      if (i == 1){lay = CB(Xd)} else {lay= CB(hlay[[i-1]])}
      if (bias_hlayers == TRUE){
        lay <- cbind(1, lay) #add bias to the hidden layer
        if (i != nlayers){outer_param <- outer_param[-1,, drop = FALSE]}      #remove parameter on upper-layer bias term
      }
      grads[[i]] <- getS(D_layer = lay, inner_param = plist[[i]], outer_deriv = grads[[i+1]], outer_param = outer_param, activation)
    }
    return(grads)
  }

  
  ###########################
  #start fitting
  if (doscale == TRUE){
    X <- scale(X)
    if (!is.null(param)){
      param <- scale(param)
    }
  }
  if (activation == 'tanh'){
    activ <- tanh
    activ_prime <- tanh_prime
  }
  if (activation == 'logistic'){
    activ <- logistic
    activ_prime <- logistic_prime
  }
  if (activation == 'relu'){
    activ <- relu
    activ_prime <- relu_prime
  }
  if (activation == 'lrelu'){
    activ <- lrelu
    activ_prime <- lrelu_prime
  }

  nlayers <- length(hidden_units)
  #get starting weights, either randomly or from a specified parlist
  if (is.null(parlist)){#random starting weights
    parlist <- vector('list', nlayers)
    for (i in 1:nlayers){
      if (i == 1){D <- ncol(X)} else {D <- hidden_units[i-1]}
      if (initialization %ni% c('XG', 'HZRS')){#random initialization schemes
        ubounds <- .7 #follows ESL recommendaton
      } else {
        if (initialization == 'XG'){
          ubounds <- sqrt(6)/sqrt(D+hidden_units[i]+2*bias_hlayers)
        }
        if (initialization == 'HZRS'){
          ubounds <- 2*sqrt(6)/sqrt(D+hidden_units[i]+2*bias_hlayers)
        }
      }
      parlist[[i]] <- matrix(runif((hidden_units[i])*(D+bias_hlayers), -ubounds, ubounds), ncol = hidden_units[i])
    }
    if (is.null(param)){
      parlist$beta_param <-  NULL
    } else {
      parlist$beta_param <- runif(ncol(param), -ubounds, ubounds)
    }
    parlist$beta <- runif(hidden_units[i], -ubounds, ubounds)
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
    if (initialization == 'enforce_normalization'){
      hlayers <- calc_hlayers(parlist, normalize = TRUE)
    }
  } else { #if a parlist is provided
    hlayers <- calc_hlayers(parlist)
  }
  parlist <- as.relistable(parlist)
  pl <- unlist(parlist) 
  #calculate ydm and put it in global...
  if (!is.null(fe_var)){
    ydm <<- demeanlist(y, list(fe_var)) 
  }
  #####################################
  #start setup
  #get starting mse
  yhat <- getYhat(pl, hlay = hlayers)
  mse <- mseold <- mean((y-yhat)^2)
  loss <- mse + lam*sum(c(parlist$beta_param*parapen
    , 0*parlist$beta_treatment, parlist$beta
    , parlist$beta_treatmentinteractions
    , unlist(parlist[!grepl('beta', names(parlist))]))^2
  )
  #Calculate gradients.  These aren't the actual gradients, but become the gradients when multiplied by their respective layer.
  grads <- calc_grads(parlist, hlayers, yhat, droplist = NULL, dropinp = NULL)
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
  msevec <- lossvec <- msetestvec <- c()
  #initialize list for plotting parameters during training
  if (para_plot == TRUE){
    para_plot_list <- lapply(parlist, function(x){
      x <- as.matrix(as.numeric(unlist(x)))
      l <- length(x)
      if (l > 30){ #if many parameters in a layer, take their quantiles and their mean
        q <- quantile(x, probs = seq(.05, .95, by = .1))
        mu = mean(x)
        return(c(q, mu = mu))
      } else { #if not, just take their absolute values
        return(x)
      }
    })
  }
  ###############
  #start iterating
  while(iter < maxit & stopcounter < maxstopcounter){
    oldpar <- list(parlist=parlist, hlayers=hlayers, grads=grads
      , yhat = yhat, mse = mse, mseold = mseold, loss = loss, updates = updates, G2 = G2
      , msevec = msevec, lossvec = lossvec)
    #Start epoch
    #Assign batches
    batchid <- sample(1:nrow(X)%/%batchsize +1)
    if (min(table(batchid))<(batchsize/2)){#Deal with orphan batches
      batchid[batchid == max(batchid)] <- sample(1:(max(batchid) - 1), min(table(batchid)), replace = TRUE)
    }
    for (bat in 1:max(batchid)) { #run minibatch 
      curBat <- which(batchid == bat)
      hlay <- hlayers#hlay may have experienced dropout, as distinct from hlayers
      #if using dropout, generate a droplist
      if (dropout_hidden < 1){
        droplist <- lapply(hlayers, function(x){
          todrop <- as.logical(rbinom(ncol(x), 1, dropout_hidden))
          if (all(todrop==FALSE)){#ensure that at least one unit is present
            todrop[sample(1:length(todrop))] <- TRUE
          }
          return(todrop)
        })
        #remove the parametric terms
        droplist[[nlayers]] <- droplist[[nlayers]][(ncol(param)+1):length(droplist[[nlayers]])]
        todrop <- rbinom(ncol(X), 1, dropout_input)
        if (all(todrop==FALSE)){#ensure that at least one unit is present
          todrop[sample(1:length(todrop))] <- TRUE
        }
        dropinp <- as.logical(todrop)
        for (i in 1:nlayers){
          hlay[[i]] <- hlay[[i]][,droplist[[i]], drop = FALSE]
        }
        Xd <- X[,dropinp]
      } else {Xd <- X; droplist = NULL}
      #Get updated gradients
      grads <- calc_grads(plist = parlist, hlay = hlay
        , yhat = yhat[curBat], curBat = curBat, droplist = droplist, dropinp = dropinp)
      #Pad the gradients with zeros to scale it back to the original size
      if (dropout_hidden < 1){
        for (i in 1:(length(grads)-1)){
          gr <- matrix(rep(0, length(curBat)*length(droplist[[i]])), nrow = length(curBat))
          gr[,droplist[[i]]] <- grads[[i]]
          grads[[i]] <- gr
        }
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
          LR*as.matrix(2*lam*c(parlist$beta_param*parapen#penalty/weight decay...
            , 0*parlist$beta_treatment, parlist$beta
            , parlist$beta_treatmentinteractions)
          )#Treatment is always unpenalized
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
      #OLS trick!
      if (OLStrick == TRUE){
        parlist <- OLStrick_function(parlist = parlist, hidden_layers = hlayers, y = y
          , fe_var = fe_var, lam = lam, parapen = parapen, treatment = treatment
        )
        pl <- unlist(parlist)
      }
      #update yhat
      yhat <- getYhat(pl, attr(pl, 'skeleton'), hlay = hlayers)
      mse <- mean((y-yhat)^2)
      msevec <- append(msevec, mse)
      loss <- mse + lam*sum(c(parlist$beta_param*parapen
        , 0*parlist$beta_treatment, parlist$beta
        , parlist$beta_treatmentinteractions
        , unlist(parlist[!grepl('beta', names(parlist))]))^2)
      lossvec <- append(lossvec, loss)
      #Finished epoch.  Assess whether MSE has increased and revert if so
      mse <- mean((y-yhat)^2)
      loss <- mse + lam*sum(c(parlist$beta_param*parapen
        , 0*parlist$beta_treatment, parlist$beta
        , parlist$beta_treatmentinteractions
        , unlist(parlist[!grepl('beta', names(parlist))]))^2
      )
      #If loss increases...
      if (oldpar$loss <= loss){
        parlist <- oldpar$parlist
        updates <- oldpar$updates
        G2 <- oldpar$G2
        hlayers <- oldpar$hlayers
        grads <- oldpar$grads
        yhat <- oldpar$yhat
        mse <- oldpar$mse
        stopcounter <- stopcounter + 1
        loss <- oldpar$loss
        msevec <- oldpar$msevec
        lossvec <- oldpar$lossvec
        LR <- LR/2
        if(verbose == TRUE){
          print(paste0("Loss increased.  halving LR.  Stopcounter now at ", stopcounter))
        }
      } else {
        LRvec[iter+1] <- LR <- LR*gravity      #gravity...
        D <- oldpar$loss - loss
        if (D<convtol){
          stopcounter <- stopcounter +1
          if(verbose == TRUE){print(paste('slowing!  Stopcounter now at ', stopcounter))}
        }else{
          stopcounter <-0
      }
      if  (verbose == TRUE & iter %% report_interval == 0 & (!is.null(test_set))){
        if (!is.null(test_set)){ 
          #we create a fitted prediction object with test set objects
          Zdm <- demeanlist(hlayers[[length(hlayers)]], list(fe_var))
          fe <- (y-ydm) - as.matrix(hlayers[[length(hlayers)]]-Zdm) %*%
            as.matrix(c(parlist$beta_param, 
                        parlist$beta_treatment, 
                        parlist$beta_treatmentinteractions, 
                        parlist$beta))
          fe_output <- data.frame(fe_var, fe)
          pr_test <- list(parlist = parlist, yhat = yhat, activation = activation, fe = fe_output
                          , fe_var = fe_var, X = X, doscale = doscale, param = param, fe_var = fe_var
                          , hidden_units = hidden_units, used_bias = bias_hlayers) 
          pr_within <- predict.panelNNET(pr_test, 
                                         newX = test_set$x_test, 
                                         fe.newX = test_set$fe_test, 
                                         new.param = test_set$test_params, 
                                         se.fit = FALSE)
          #predicted_mse
          mse_test <- mean((pr_within-test_set$y_test)^2)
          msetestvec[iter] <- mse_test
print(msetestvec)
        }
        writeLines(paste0(
          "*******************************************\n"
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
          , "Testset MSE ", mse_test, "\n"
          , "*******************************************\n"  
        ))
      }
      if (verbose == TRUE & iter %% report_interval == 0 & (is.null(test_set))){
        writeLines(paste0(
          "*******************************************\n"
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
          , "no training set provided\n"
          , "*******************************************\n"
        ))
      }
      if (para_plot == TRUE){#additional plots if plotting parameter evolution
        par(mfrow = c(ceiling(length(parlist)/2)+3,2))
      } else {
        par(mfrow = c(3,2))
      }
      plot(y, yhat, col = rgb(1,0,0,.5), pch = 19, main = 'in-sample performance')
      abline(0,1)
      plot(LRvec, type = 'b', main = 'learning rate history')
      plot(msevec, type = 'l', main = 'all epochs', ylim = range(c(msevec, msetestvec), na.rm = TRUE))
      lines(msetestvec, type = "l",col = "blue")
      plot(msevec[(1+(iter)*max(batchid)):length(msevec)], type = 'l', ylab = 'mse', main = 'Current epoch')
      plot(lossvec, type = 'l', main = 'all epochs')
      plot(lossvec[(1+(iter)*max(batchid)):length(lossvec)], type = 'l', ylab = 'loss', main = 'Current epoch')
        if (para_plot == TRUE){
          #update para plot list
          for (lay in 1:length(para_plot_list)){
            x <- as.matrix(as.numeric(parlist[[lay]]))
            l <- length(x)
            if (l > 30){
              q <- quantile(x, probs = seq(.05, .95, by = .1))
              mu = mean(x)
              para_plot_list[[lay]] <- cbind(para_plot_list[[lay]], c(q, mu = mu))
      
              plot(para_plot_list[[lay]]['mu',], ylim = range(para_plot_list[[lay]])
                , type = 'l', col = 'red', main = names(parlist)[[lay]], ylab = 'weights'
              )
              apply(para_plot_list[[lay]][-nrow(para_plot_list[[lay]]),], 1, lines, col = 'black')
              abline(h = 0, lty = 2)
            } else {
              para_plot_list[[lay]] <- cbind(para_plot_list[[lay]], x)
              plot(apply(para_plot_list[[lay]], 2, mean), ylim = range(para_plot_list[[lay]])
                , type = 'l', col = 'red', main = names(parlist)[[lay]], ylab = 'weights'
              )
              apply(para_plot_list[[lay]], 1, lines, col = 'grey')
              abline(h = 0, lty = 2)
            }
          }
        }
      } # this closes verbose command 
    } # closes gravity command 
    iter <- iter+1
  } #closes the while loop
  #If trained with dropput, weight the layers by expectations
  if(dropout_hidden<1){
    for (i in nlayers:1){
      if (i == 1){
        parlist[[i]] <- parlist[[i]] * dropout_input
      } else {
        parlist[[i]] <- parlist[[i]] * dropout_hidden
      }
    }
    parlist$beta <- parlist$beta * dropout_hidden
    if (OLStrick == TRUE){
      parlist <- OLStrick_function(parlist = parlist, hidden_layers = hlayers, y = y
        , fe_var = fe_var, lam = lam, parapen = parapen, treatment = treatment
      )
    }
    #redo the hidden layers based on the new parlist
    hlayers <- calc_hlayers(parlist)
    yhat <- getYhat(unlist(parlist), hlay = hlayers)
  }
  conv <- (iter<maxit)#Did we get convergence?
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
  output <- list(yhat = yhat, parlist = parlist, hidden_layers = hlayers
    , fe = fe_output, converged = conv, mse = mse, loss = loss, lam = lam, time_var = time_var
    , X = X, y = y, param = param, fe_var = fe_var, hidden_units = hidden_units, maxit = maxit
    , used_bias = bias_hlayers, final_improvement = D, msevec = msevec, RMSprop = RMSprop, convtol = convtol
    , grads = grads, activation = activation, parapen = parapen, doscale = doscale, treatment = treatment
    , interact_treatment = interact_treatment, batchsize = batchsize, initialization = initialization)
  return(output) # list 
}



