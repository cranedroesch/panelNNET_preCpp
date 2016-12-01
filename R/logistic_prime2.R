logistic_prime2 <-
function(x){#Function to go from an activated term to its derivative
  v = log(x) - log(1-x)
  logistic_prime(v)
}
