esn_validity <- function(esn) {
  warnings <- character()

  #Check the different validity criteria
  if(length(esn@leaking.rate) != 1){
    warnings <- c(warnings,paste("Leaking rate is length", esn@leaking.rate, ".  Should be 1", sep = " "))
  }
  if(length(esn@spectral.radius) != 1){
    warnings <- c(warnings,paste("Spectral radius is length", esn@spectral.radius,". Should be 1",sep = " "))
  }
  if(esn@spectral.radius < 0){
    warnings <- c(warnings,paste("Spectral radius is",esn@spectral.radius,". Should be positive",sep = " "))
  }

  if(length(warnings) == 0) TRUE else warnings
}

#' @param leaking.rate The rate which the net is learning at
#' @param lambda The lambda of the ESN
#' @param spectral.radius The spectral radius of the network
#' @param W_in W_in of the network
#' @param W_out W_out of the network
#' @param W The reservoir of the matrix
#' @param U The input matrix for the network
#' @param Y The response signal for the network to be trained with
#' @author Matthias Adriaens
setClass("ESN",representation(leaking.rate = "numeric",
                              lambda = "numeric",
                              spectral.radius = "numeric",
                              n.neurons = "numeric",
                              W_in = "matrix",
                              W_out = "matrix",
                              W = "matrix",
                              U = "matrix",
                              Y = "matrix",
                              X = "matrix",
                              regCoef = "numeric"),
         prototype(leaking.rate = 0.2,
                   lambda = 0.5,
                   spectral.radius = 0.5),
         validity = esn_validity
)

init_W_in <- function(N,K){
  #N number of reservoir units, K is number of inputs
  W_in <- matrix(runif(N*(K+1), -0.5, 0.5),
                 nrow = N,
                 ncol = (K+1))
  return(W_in)
}

init_W <- function(N){
  #N is number of reservoir units
  W <- matrix(runif(N*N, -0.5, 0.5),
              nrow = N,
              ncol = N)
  cat('Computing spectral radius...')
  rhoW = abs(eigen(W,only.values=TRUE)$values[1])
  print('done.')
  W = W * 1.25 / rhoW
  return(W)
}

init_W_out <- function(K,N,L){
  #L = number of outputs
  W_out <- matrix(runif(L*(K+N+1), -0.5, 0.5),
                  nrow = L,
                  ncol = (K+N+1))
  return(W_out)
}

init_reservoir <- function(N,K,L){

  init_res <- list()
  init_res[["W_in"]] <- init_W_in(N,K)
  init_res[["W"]] <- init_W(N)
  init_res[["W_out"]] <- init_W_out(K,N,L)
  return(init_res)
}

createESN <- function(leaking.rate =0.2,
                    lambda = 0.5,
                    spectral.radius = 0.5,
                    n.neurons = 1000,
                    U,
                    Y){
  N <- n.neurons
  K <- ncol(U)
  L <- ncol(Y)
  init_res <- list()
  init_res <- init_reservoir(N,K,L)

  X <- matrix(0,1+ncol(U) + n.neurons,nrow(Y))

  esn <- new("ESN",
             leaking.rate = leaking.rate,
             lambda = lambda,
             spectral.radius = spectral.radius,
             n.neurons = n.neurons,
             W_in = init_res[["W_in"]],
             W_out = init_res[["W_out"]],
             W = init_res[["W"]],
             U = U,
             Y =  Y,
             X = X,
             regCoef = 1e-2)
  return(esn)
}

#######################################
####TRAINING THE ECHO STATE NETWORK####
#######################################

setGeneric("train", function(esn) 0)
#Matrix runs the reservoir and collects the reservoir states for a given initilized echo state network
setMethod("train", signature(esn = "ESN"), function(esn) {
  x <- matrix(0,nrow = esn@n.neurons,ncol =1)

  for(i in 1:nrow(esn@Y)){
    #Update equation for the reservoir states
    x <-  (1-esn@leaking.rate)*x + tanh(esn@W_in%*%t(t(c(1,esn@U[i,])))+  esn@W%*%x)
    #Collecting all the reservoir states
    #Wash out the initial set up
    if(i > 100){
      esn@X[,i] <- c(1,esn@U[i,],as.matrix(x))
    }
  }
  #Print the regularization coefficient to the user
  print(esn@regCoef)
  #Train W_out in a linear way using Ridge regression
  esn@W_out <- t(esn@Y)%*%t(esn@X)%*%solve(esn@X%*%t(esn@X) + esn@regCoef*diag(nrow(esn@X)))
  esn
})

#######################################
####PREDICTING WITH ECHO STATE NET#####
#######################################

setGeneric("predict", function(esn, U) 0)
#Method predicts an an output matrix for a given input matrix and a trained ESN
setMethod("predict", signature(esn = "ESN", U = "matrix"), function(esn,U) {
  #Init output matrix for prediction
  Yp <- matrix(0, nrow = nrow(U) , ncol = ncol(esn@Y))
  #Init single reservoir state
  x <- matrix(0,nrow = esn@n.neurons,ncol =1)

  for (i in 1:(nrow(U) - 1)) {
    #Calculate reservoir state with given inputs
    x <- (1-esn@leaking.rate)*x + tanh(esn@W_in%*%t(t(c(1,U[i,])))+ esn@W%*%x)
    #Predict output with trained w_out layer
    Yp[i+1,] <- esn@W_out %*% c(1,esn@U[i,],as.matrix(x))
  }
  #Return the output
  Yp
})








