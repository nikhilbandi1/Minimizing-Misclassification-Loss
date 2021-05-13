set.seed(1)
data <- read.csv("https://raw.githubusercontent.com/nikhilbandi1/Minimizing-Misclassification-Loss/main/180476.csv")
#data <- read.csv("https://dvats.github.io/assets/data/180476.csv")
y <- as.matrix(data$y)
X <- as.matrix(data[,2:51])
beta <- as.matrix(rep(0,50))

#test <- read.csv("https://dvats.github.io/assets/data/180380.csv")
#X.new<- as.matrix(test[,2:51])
#Y.new<-as.matrix(test$y)

newton <- function(func, initial_values,tol = 1e-16){
  params <- initial_values
  check <-1
  while(check > tol){
    func_eval <- func(params)
    params <- params -  solve(func_eval$ddf)%*%func_eval$df
    check <- sqrt(t(func_eval$df)%*%func_eval$df)
  }
  return(params)
}

L_L <- function(y,beta,X){
  # Calculate the vector of probabilities:
  pi <- exp(X%*%beta)/(1+exp(X%*%beta))
  f <- prod(ifelse(y==1,pi,1-pi))
  # First derivative:
  df <- t(t(y- pi)%*%X )
  # Second derivative:
  W <- diag(as.vector(pi*(1-pi)))
  ddf <- -t(X)%*%W%*%X
  # Output:
  output_list <- list(f = f, df = df,ddf = ddf)
  return(output_list)
}

admission <- as.matrix(data$y)
vars <- X
my_coefs <- newton(function(input){L_L(y = admission,X = vars,beta = input)},initial_values = beta,tol = 1e-13)
fromGLM <- glm(formula = admission~vars-1,family = binomial())
GLMcoefficients <- fromGLM$coefficients
my_results <- L_L(y,my_coefs,X)
glm_results <- L_L(y,GLMcoefficients,X)
L_L_Difference <- my_results$f - glm_results$f
DL_L_Difference <- t(my_results$df - glm_results$df)%*%(my_results$df - glm_results$df)

print(data.frame(matrix(GLMcoefficients),my_coefs))

beta<-as.matrix(my_coefs)

# p<- 1/(1+exp(-(X.new%*%beta)))

est.y <- function(X.new, beta)
{
  pr <- 1/(1+exp(-(X.new%*%beta)))
  y.pred<-as.matrix(as.integer(pr>=0.5))
  return(y.pred)
}

# loss<-function(Y.new,y.pred)
# {
#   l<-sum(abs(Y.new-y.pred))
#   return (l)
# }
# 
# loss(Y.new, est.y(X.new, beta))
# print(loss)
#save(est.y, beta, file = "180476.Rdata")
