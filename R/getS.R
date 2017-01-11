getS <-
function(D_layer, inner_param, outer_deriv, outer_param, activation){
  if (activation == 'tanh'){
    sigma_prime <- function(x){1- tanh(x)^2}
  }
  if (activation == 'logistic'){
    sigma_prime <- function(x){logistic(x) * (1- logistic(x))}
  }
  if (activation == 'relu'){
    sigma_prime <- relu_prime
  }
  sigma_prime(D_layer %*% inner_param) * outer_deriv %*% t(outer_param)
}
