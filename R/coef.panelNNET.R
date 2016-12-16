coef.panelNNET <- function(obj, rescale = TRUE){
  parm <- obj$parlist$beta_param
  if (rescale == TRUE & obj$doscale == TRUE){
    parm <- parm/attr(obj$param, "scaled:scale")
  }
  if (!is.null(obj$vcs)){
    se <- lapply(obj$vcs, function(x){
      se <- sqrt(diag(x$vc))[1:length(parm)]
      if (rescale == TRUE & obj$doscale == TRUE){
        se <- se/attr(obj$param, "scaled:scale")
      }
      return(se)
    })
    rn <- names(se)
    se <- t(matrix(unlist(se), length(se[[1]])))
    colnames(se) <- colnames(obj$param)
    rownames(se) <- rn
  } else {
    se <- 'No parameter covariance matrix in fitted model!'
  }
  return(list(parm = parm, se = se))
}

