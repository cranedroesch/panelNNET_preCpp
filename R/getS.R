getS <-
function(D_layer, inner_param, outer_deriv, outer_param, activation){
  if (activation == 'tanh'){
    activ_prime <- function(x){1- tanh(x)^2}
  }
  if (activation == 'logistic'){
    activ_prime <- function(x){logistic(x) * (1- logistic(x))}
  }
  if (activation == 'relu'){
    activ_prime <- relu_prime
  }
  if (activation == 'lrelu'){
    activ_prime <- lrelu_prime
  }
  activ_prime(D_layer %*% inner_param) * outer_deriv %*% t(outer_param)
}
