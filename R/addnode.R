

#function to add a node to a layer of a network
addnode <- function(obj, layer){
  obj$hidden_layers[[layer]] <- cbind(obj$hidden_layers[[layer]], runif(nrow(obj$hidden_layers[[layer]]), -.7, .7))  
  obj$parlist[[layer]] <- cbind(obj$parlist[[layer]], runif(nrow(obj$parlist[[layer]]), -.7, .7))
  if (layer == length(obj$hidden_layers)){
    obj$parlist$beta <- c(obj$parlist$beta, runif(1, -.7, .7))
  } else {
    obj$parlist[[layer - 1]] <- rbind(obj$parlist[[layer - 1]],runif(ncol(obj$parlist[[layer-1]]), -7, 7))
  }
  obj$hidden_layers[[layer]] <- newlayer
  return(obj)
}

