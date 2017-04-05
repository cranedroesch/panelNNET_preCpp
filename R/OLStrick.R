

OLStrick <- function(parlist, hidden_layers, y, fe_var, lam, parapen, treatment){
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
  D <- rep(lam, ncol(Zdm))
  if (is.null(fe_var)){
    pp <- c(0, parapen) #never penalize the intercept
  } else {
    pp <- parapen #parapen
  }
  if (!is.null(treatment)){pp <- append(pp, 0)}#treatment always follows parametric terms and will not be penalized
  D[1:length(pp)] <- D[1:length(pp)]*pp #incorporate parapen into diagonal of covmat
  #function to find implicit lambda
  f <- function(lam){
    bi <- solve(t(Zdm) %*% Zdm + diag(D)) %*% t(Zdm) %*% targ
    (t(bi) %*% bi - constraint)^2
  }
  #optimize it
  o <- optim(par = lam, f = f, method = 'Brent', lower = lam, upper = 1e9)
  #new lambda
  newlam <- o$par
  #New top-level params
  D[D!=0] <- newlam
  b <- solve(t(Zdm) %*% Zdm + diag(D)) %*% t(Zdm) %*% targ
  parlist$beta_param <- b[grepl('param', rownames(b))]
  parlist$beta <- b[grepl('nodes', rownames(b))]
  return(parlist)
}
