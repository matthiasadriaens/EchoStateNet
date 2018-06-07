## Echo State Network implementation

# Install in your R session:

install.packages("devtools")
library(devtools)
devtools::install_github("matthiasadriaens/EchoStateNet")
library(EchoStateNet)


# Simple example of Mackey-Glass system:
```{r}
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
```


