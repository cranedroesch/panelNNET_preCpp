tanh_prime2 <-
function(x){
  v = .5*(log(1+x) - log(1-x))
  tanh_prime(v)
}
