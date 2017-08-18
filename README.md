# Artificial Neural Networks, Deep Learning Churn Modeling

Business Problem: Dataset of a bank with 10,000 customers measured lots of attributes of the customer and is seeing unusual churn rates at a high rate. Want to understand what the problem is, address the problem, and give them insights. 10,000 is a sample, millions of customer across Europe. Took a sample of 10,000 measured six months ago lots of factors (name, credit score, grography, age, tenure, balance, numOfProducts, credit card, active member, estimated salary, exited, etc.). For these 10,000 randomly selected customers and track which stayed or left.

Goal: create a geographic segmentation model to tell which of the customers are at highest risk of leaving. 

Valuable to any customer-oriented organisations. Geographic Segmentation Modeling can be applied to millions of scenarios, very valuable. (doesn't have to be for banks, churn rate, etc.). Same scenario works for (e.g. should this person get a loan or not? Should this be approved for credit => binary outcome, model, more likely to be reliable). Fradulant transactions (which is more likely to be fradulant) 

- Binary outcome with lots of independent variables you can build a proper robust model to tell you which factors influence the outcome. 

![alt text](https://github.com/kevincwu0/deep-learning-ann-churn-modeling/blob/master/data_screenshot.png)

Problem: Classification problem with lots of independent variables (credit score, balance, number of products) and based on these variables we're predicting which of these customers will leave the bank. Artificial Neural Networks can do a terrific job with Classification problems and making those kind of predictions.


### Libraries used:
1. Theano
  - numerical computation library, very efficient for fast numerical computations based on Numpy syntax
  - GPU is much more powerful than CPU, as there are many more cores and run more floating points calculations per second
  - GPU is much more specialized for highly intensive computing tasks and parallel computations, exactly for the case for neural networks
  - When we're forward propogating the activations of the different neurons in the neural network thanks to the activation function well that involves parallel computations
  - When errors are backpropagated to the neural networks that again involves parallel computation
  - GPU is a much better choice for deep neural network than CPU - simple neural networks, CPU is sufficient
  - Created by Machine Learning group at the Univeristy of Montreal
2. Tensorflow
  - Another numerical computation library that runs very fast computations that can run on your CPU or GPU
  - Google Brain, Apache 2.0 license
  - Theano & Tensorflow are used primarily for research and development in the deep learning field
  - Deep Learning neural network from scratch, use the above
  - Great for inventing new deep learning neural networks, deep learning models, lots of line of code
3. Keras
  - Wrapper for Theano + Tensorflow
  - Amazing library to build deep neural networks in a few lines of code
  - Very powerful deep neural networks in few lines of code
  - based on Theano and Tensorflow
  - Sci-kit Learn (Machine Learning models), Keras (Deep Learning models)

Installing Theano, Tensorflow in three steps with Anaconda installed:
1. `$ pip install theano`
2. `$ pip install tensorflow`
3. `$ pip install keras`
4. `$ conda update --all`
