\name{createESN}
\alias{createESN}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
  Creates an echo state network
}
\description{
  Creates an echo state network form the ESN-type
}
\usage{
createESN(leaking.rate = 0.35,
								lambda = 1.25,
								n.neurons = 1000,
								wash.out = 100,
								feedback = FALSE,
								regCoef = 1e-8,
								resCon = 1,
								U = net_u,
								Y = net_Yt)
}
\arguments{
	\item{leaking.rate}{}Leaking rate of the ESN, cfr. learning rate}
	\item{lambda}{Scales the spectral radius of the reservoir}
	\item{n.neurons}{Number of neurons in the reservoir}
	\item{wash.out}{Number of instances that is chosen to wash out initalization}
	\item{feedback}{Logical value for incorporating output feedback into the matrix}
	\item{regCoef}{Regularization coefficient of the network}
	\item{resCon}{Density factor for non-zero elements in the reservoir}
	\item{U}{Input matrix, may be mulit-variate}
	\item{Y}{Target matrix, holds the labels}
}
\value{
S4 ESN class object
}
\author{
Matthias Adriaens
}


\examples{
trainLen = 2000
testLen = 2000
initLen = 100

#Data to be found in the /data folder in package:

data = as.matrix(read.table('MackeyGlass_t17.txt'))

net_u = as.matrix(data[1:trainLen])
net_Yt = matrix(data[2:(trainLen+1)])


net <- EchoStateNet::createESN(leaking.rate = 0.35,
								lambda = 1.25,
								n.neurons = 1000,
								wash.out = 100,
								feedback = FALSE,
								regCoef = 1e-8,
								resCon = 1,
								U = net_u,
								Y = net_Yt)

trained_net <- EchoStateNet::train(net)

Ypred <- EchoStateNet::predict(trained_net,
								U = as.matrix(to.predict)),
								generative = FALSE,
								genNum = 2000)
}

