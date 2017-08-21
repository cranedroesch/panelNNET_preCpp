
#function to make a mask for convolutional nets

makeMask <- function(X, topology, span, step){
  stops <- seq(round(span/2), (max(topology, na.rm = T)), by = step)
  # make a matrix of zeros, of dimension equal to the number of inputs by the number of outputs (which is a function of the span)
  TVmask <- foreach(i = 1:length(topology), .combine = rbind) %do% {
    interval <- topology[i] + round(span/2) * c(-1, 1)
    as.numeric(stops> interval[1] & stops<interval[2])
  }
  # Variables that don't have a topology should be NA -- they will get set to zero
  TVmask[is.na(TVmask)] <- 0
  colnames(TVmask) <- stops
  # Add on a block diagonal matrix for the non-time-varying terms
  NTVmask <- rbind(matrix(rep(0, length(topology[!is.na(topology)]) *
                                length(topology[is.na(topology)])), ncol = length(topology[is.na(topology)])),
                   diag(rep(1, length(topology[is.na(topology)]))))
  colnames(NTVmask) <- colnames(X)[is.na(topology)]
  mask <- cbind(TVmask, NTVmask)
  rownames(mask) <- colnames(X)
  return(mask)
}