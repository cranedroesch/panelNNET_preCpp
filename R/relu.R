relu <-
function(v){
  v[v<0] <- 0
  v
}

relu_prime <- 
function(v){
  v[v<0] <- 0
  v[v>=0] <- 1
  v
}
