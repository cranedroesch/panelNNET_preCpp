summary_table_element <-
function(vc, parm){#Function used internally to the infernce == TRUE argument
  if (inherits(vc, 'error')){
    'Error!  Probably an ill-conditioned covariance matrix'
  } else {
    se <- sqrt(diag(vc$vc))[1:length(parm)]
    p <- 2*pnorm(-abs(parm/se))
    stars <- rep('',length(parm))
    stars[p<.1] <- '*'
    stars[p<.05] <- '**'
    stars[p<.01] <- '***'
    list(signif(se, 4), signif(p,4), stars)
  }
}
