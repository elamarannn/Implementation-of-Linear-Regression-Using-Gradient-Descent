# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard python libraries for Gradient design.
2. Introduce the variables needed to execute the function.
3. Use function for the representation of the graph.
4. Using for loop apply the concept using the formulae.
5. Execute the program and plot the graph.
6. Predict and execute the values for the given conditions. 
 

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: Elamaran S E
RegisterNumber:212222230036

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1.txt",header=None)

print("Profit prediction graph:")
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,1000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  """
  Take in a numpy array X,y,theta and generate the cost function in a linear regression model
  """
  m=len(y) #length of the training data
  h=X.dot(theta) #hypothesis
  square_err=(h-y)**2

  return 1/(2*m) * np.sum(square_err) #returning
  
  data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

print("Compute cost value:")
computeCost(X,y,theta) #Call the function

def gradientDescent(X,y,theta,alpha,num_iters):
  """
  Take in numpy array X,y and theta and update theta by taking number with learning rate of alpha

  return theta and the list of the cost of theta during each iteration
  """
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions -y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))

  return theta,J_history  
  
theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) value:")
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

print("Cost function using Gradient Descent graph:")
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

print("Profit prediction graph:")
plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
  """
  Take in numpy array of x and theta and return the predicted value of y based on theta
  """

  predictions= np.dot(theta.transpose(),x)

  return predictions[0]
  
predict1=predict(np.array([1,3.5]),theta)*10000
print("Profit for the population 35,000:")
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("Profit for the population 70,000:")
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))


```

## Output:
### Profit Prediction Graph:
![268232479-673c0ffa-ae7f-4702-8e11-78393b4d8a2d](https://github.com/elamarannn/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497531/76198b69-aa61-4ff5-9e92-baab701dfa24)
### Compute Cost Value:
![268232556-fe50f74a-2f2c-456f-a2b8-44b58ca77e76](https://github.com/elamarannn/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497531/2be25e87-cf1e-4038-b91e-cfd9f52eebb8)
### h(x) value:
![268232581-d306795e-1849-4a28-b815-b086ede8eabf](https://github.com/elamarannn/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497531/480167fc-7db0-49ea-bca6-8cff8bc9ec0b)
### Cost function using Gradient Descent graph:
![268232608-a76d12a7-ba8f-45aa-8490-e9264c013503](https://github.com/elamarannn/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497531/0654d31d-1a7e-4063-b2d7-03b89b41fd2b)
### Profit prediction graph:
![268232650-b9ca5e73-3d15-4d12-87b0-e830a3adc08f](https://github.com/elamarannn/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497531/8665c3fc-c7e5-4342-944b-60a62a0ddb0d)
### Profit for the population 35,000:
![268232709-74ac0a17-259b-4ce6-9320-32c514484fcc](https://github.com/elamarannn/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497531/4b7d6ed1-4c28-4ac8-997a-6030b191e8c6)
### Profit for the population 70,000:
![268232737-7b57e4cd-a671-49c9-9b52-ca83128eaf47](https://github.com/elamarannn/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497531/239021b4-9e04-44ee-a29c-1e82779afde2)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
