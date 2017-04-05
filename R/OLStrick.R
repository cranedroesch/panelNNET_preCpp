

OLStrick <- function(parlist, hidden_layers, y, fe_var, lam, parapen, treatment){
#parlist <- pnn$parlist
#hidden_layers <- pnn$hidden_layers
#y = pnn$y
#fe_var <- pnn$fe_var
#lam <- pnn$lam
#parapen <- pnn$parapen
#treatment = NULL
#hidden_layers <- hlayers
#hidden_layers <- hlayers
  constraint <- sum(c(parlist$beta_param*parapen, parlist$beta)^2)
  #getting implicit regressors depending on whether regression is panel
  if (!is.null(fe_var)){
    Zdm <- demeanlist(hidden_layers[[length(hidden_layers)]], list(fe_var))
    targ <- demeanlist(y, list(fe_var))
  } else {
    Zdm <- hidden_layers[[length(hidden_layers)]]
    targ <- y
  }
  #set up the penalty vector
  D <- rep(1, ncol(Zdm))
  if (is.null(fe_var)){
    pp <- c(0, parapen) #never penalize the intercept
  } else {
    pp <- parapen #parapen
  }
  if (!is.null(treatment)){pp <- append(pp, 0)}#treatment always follows parametric terms and will not be penalized
  D[1:length(pp)] <- D[1:length(pp)]*pp #incorporate parapen into diagonal of covmat
  #function to find implicit lambda
  f <- function(lam){
    bi <- glmnet(y = targ, x = Zdm, lambda = lam, alpha = 0, intercept = FALSE, penalty.factor = D, standardize = FALSE)
    bi <- as.matrix(coef(bi))[-1,]
    (t(bi*D) %*% (bi*D) - constraint)^2
  }
  #optimize it
  o <- optim(par = lam, f = f, method = 'Brent', lower = lam, upper = 1e9)
  #new lambda
  newlam <- o$par
  #New top-level params
  br <- glmnet(y = targ, x = Zdm, lambda = newlam, alpha = 0, intercept = FALSE, penalty.factor = D, standardize = FALSE)
  b <- as.matrix(coef(br))[-1,,drop = FALSE]
  parlist$beta_param <- b[grepl('param', rownames(b))]
  parlist$beta <- b[grepl('nodes', rownames(b))]
  return(parlist)
}





#mean((targ - Zdm %*% c(parlist$beta_param, parlist$beta))^2)
#mean((targ - Zdm %*% b)^2)
