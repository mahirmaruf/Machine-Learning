## Exercise 3

### 1 Multi-class classification

The dataset is from ex3data1.mat, which is a training set of 5000 handwritten digits. Each example is an unraveled 20x20 pixel picture, where each grid is a floating point number corresponding to the pixel’s brightness at that location. So X is a [5000 x 400] matrix.

The Y is each example’s label (from 0-9). It is a vector with digits 1-10 which correspond to the hand written number. Octave indexes from 1, so the labels also start at 1. Then the label ‘0’ corresponds to 10 in Y. Thus, Y is a [5000 x 1] matrix. 

#### 1.3 Vectorizing logistic regression

Here, we will be editing the `lrCostFunction.m` file. This should look similar to the code written in the Example 2. 

For brief review, we first defined the function, and that J and grad are the outputs of that function. 

We called the function `lrCostFunction`. And its inputs are theta, X, y, and lambda. Thats how we get:
``` 
function [J, grad] = lrCostFunction(theta, X, y, lambda)
```
Check the notes for Exercise 2 for a more detailed explanation of this function. But the important thing to remember is the *purpose* of this function, that namely, we supply the parameters, the features (X), the output (y), and the $\lambda$ (for regularization), and the the function spits out the cost and the gradient at that point. 

Now that we have programmed this function, we can move on to the next part. 

#### 1.4 Once vs all Classification

This gets a little dicey. In this part of the assignment, we will use the `oneVsAll.m` file. The only output of this function is `all_theta` which is contains the parameters, $\theta$, for each pixel for each number label. There are 10 number labels (0-9) and each observation have 400 features. Thus this is a [10 x 401] matrix (k x n+1). 

We'll go over the lines of code now. 

First, we will define the function. By now, we should be able to understand what the components are of this code:
```
function [all_theta] = oneVsAll (X, y, num_labels, lambda)
```

But to review, `all_theta` is our output of the function. `oneVsAll` is the name of the function that we call in `ex3.m`. And the `X`, `y`, `num_labels`, and `lambda` are the inputs. You will not see `num_labels` defined until `ex3.m` so as a spoiler alert, it is set equal to 10 because we have 10 labels.

Next, `m` and `n` are initialzed. There are 5000 observations and 400 features. 
```
m = size(X, 1)
n = size(X, 2)
```
And this makes m = 5000, and n = 400. 

Because we have a $\theta$$_{0}$, we need to add a column of 1s to our X. We do that with this line of code: `X = [ones(m, 1) X]`. This adds a column of 1s to the beginning of X. So `X` is now a [5000 x 401] matrix. 

We now initialize the `initial_theta` with the code `initial_theta = zeros(n+1, 1)` to create a [401 x 1] column vector of 0s. 

The options we set will display the gradient and set the maximum number of iterations to 50. 

***
Ok, lets take a step back to talk about what were want to do. We want to create a set of parameters ($\theta$) for each number label in *k*, such that when logistic regression is applied to classify a set of 400 pixels as that number *k* or not, it will result in the lowest cost. 

That probably didn't clarify much. Simply, we are training a set of $\theta$s to classify a set of pixels as either the corresponding number *k* or not that number *k*. So those pixels are either that number or they are not. "Yes" or "no". 

Lets take an example. Say we have a single observation of the unrolled pixels of a hand drawn "8". And the label of this picure is also an "8". Our goal would be to find a set of parameters that would result in the lowest cost of our hypothesis: $h_\theta$(x) = g($\theta$$^T$X), where g(z) = $\frac{1}{1+e^z}$.

The problem is that the y is not "yes" or "no". In this example, the y are the labels in the *k*, so they are numbers from 0-9. This won't work for logistic regression. So how will we have a binary y? And moreover, how will we have a different binary y for each value of *k*? 

Here is where the `for` loop comes in. 
***

Let's break this seamingly simple code down:
```
for c = 1:num_labels
  all_theta(c,:) = fmincg ( @(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options)
 end
 ```
 First: in each loop of the code, `c` goes from `1` through `num_labels`. Like we mentioned earlier, `num_labels` is 10, which is defined in `ex3.m`.  So this is going to loop from 1-10, and each one corresponds to a label *k*. 

 Let's skip the `all_theta(c,:)` part and come back to that. Let's discuss the `fmincg` function. This is a built-in function to optimize a a specific argument. From the *help*, it says "Minimize a continuous differentialble multivariate function". So we will use this to find $\theta$s that minimize the `lrCostFunction` we wrote. 

 We use the `@(t)` to denote which variable we want to optimize - that would be `t`, which we are coming to. After that, we specify the function that we are minimizing - that would be our cost function. Now, inside of our cost function, we have to specify its arguments. The first is `t`. This is the function that we want to optimize, and in the `lrCostFunction.m` file, this argument is actually `theta`. So we know we are going to be optimizing the `theta`. 

 Then, as we saw from `lrCostFunction.m`, the next argument is X. Nothing special. 

 Now. Here is where things get interesting - when we supply the y argument. We say here `(y==c)`. Why? What does that do?

 Well, as we go through the `for` loop, `c` is going to loop go from 1:10. And now for each observation, the `y` is going to be evaluated against the `c`. If they are the same, `1` is returned. If they are not the same, `0` is returned. So now we have a y that is binary for logistic regression! And this binary value changes for each number in *k*!

Ok next for the `lrCostFunction` we supply the `lambda` for regularization. Close the parentheses, and move onto the next argument.

The next argument for `fmincg` is the `initial_theta` which as we remember is a [401 x 1] matrix. This is the starting point used in the `lrCostFunction`. Now add the `options`.

What will this return for us? It will give us 10 [401 x1] matrices, one for each *k*. This results in all_theta being a [k x n+1] matrix.

When we run `ex3.m`, it will give us the cost for each k at 50 iterations. 




