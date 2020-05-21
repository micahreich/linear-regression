# Linear Regression
This github repository contains two simple linear regression machine learning models which can be applied to linear data to make predictions. One model works with single-feature inputs while the other is adapted for multivariate and multi-featured predictions. 

### Linear Regression Algorithm
Assuming we have a dataset we want to model linearly, we can make predictions using the best-fit line to the data. To create the best-fit line without using an iterative algorithm (and not using the normal equations), we can use gradient descent.

Our prediction of the best-fit line is in the form of:

<p align="center">
  <img src="https://quicklatex.com/cache3/72/ql_6223a11493be6753f8144eed6db1b372_l3.png">
</p>

We will be given an input x, and we must figure out m and b to be able to make an accurate prediction. For simplicity, let's call b and m theta 0 and theta 1, respectively, and we will call our prediction h(x). So, our prediction is in the form of:

<p align="center">
  <img src="https://quicklatex.com/cache3/dd/ql_14122867106a62343e50c3f6517039dd_l3.png">
</p>

where additional theta parameters and inputs x are added if the prediction is to be multivariate. To measure the error between our prediction and the true value within our dataset, we will use the least squares expression to model our error over the entire dataset. Our error J(theta) looks like:

<p align="center">
  <img src="https://quicklatex.com/cache3/97/ql_ce536cde80f40de5339ca2b3d4a8ae97_l3.png">
</p>

### Gradient Descent Algorithm
Starting with our theta parameters as zero or as randomly initialized, the gradient descent algorithm seeks to minimize J(theta) as much as possible by finding the correct parameters to minimize error. To find the best parameters possible, we will take an iterative approach. For one iteration of training, we will look at how small changes in our parameters theta affect the error, J(theta). After taking note of how to change theta to lower the error, we will change our parameters to reflect our findings. An update to a parameter theta will be formalized as:

<p align="center">
  <img src="https://quicklatex.com/cache3/cf/ql_893566d3b8ed63f1395b7cea5f0062cf_l3.png">
</p>

where the partial derivitive of J(theta) represents how J(theta) changes with respect to the parameter theta j (either theta 0 or theta 1 if doing single variable regression). Over multiple training iterations, the parameters are refined until J(theta) is minimized. The alpha is known as the learning rate, and it represents the magnitude with which theta should be changed during the training iteration.

Note: changing theta by looking at the partial derivitive of the loss over the entire dataset is known as batch gradient descent.

