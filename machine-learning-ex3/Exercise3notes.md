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

#### 1.4 One vs all Classification

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

##### 1.4.1 One-vs-all Prediction

Now that we have calculated the optimum set of parameters for each letter in *k*, we can apply these parameters to our dataset and see how well our parameters predict the letters in our data. Since the parameters were trained on this dataset, it should be pretty accurte. 

We will use the `predictOneVsAll.m` file. This creates a function whose output is `p` and inputs are `all_theta` (which we optimized previously) and `X`. 

Here we will define `m` as a 5000, which is the number of examples in `X`. Then we define `num_labels` as the number of of rows in `all_theta` (which is 10).

Our output, `p`, is going to be the predicted letter for each example. So this is a vector of 5000 elements. 

Now that we have defined and initialized the variables, we can start the prediction. So the first thing that we have to do is add the "place holder" to X. Recall that X is a [5000 x 400] matrix. We can add a column of 1s in the beginnin with the code `X = [ones(m, 1) X]`

The actual calculation is actually pretty easy. But the understanding is a little difficult. We'll go through both though. 

We will define `z` as `X` multiplied by `all_theta'`. Again `X` is [5000 x 401] and transposed `all_theta` is [401 x 10]. So `z` becomes a [5000 x 10] matrix. Then we take the sigmoid of each element with the `sigmoid.m` function. 

What does this represent? This is the "probability" of each example belonging to *k* for each $\theta_k$. So each example has 10 probabilities. What have done is written a hypothesis, $h_\theta(x)$, for each $\theta_k$ and applied it to each example. That means each example has *k* number of probabilities in the rows.

What is interesting is that the index of highest probability represents the predicted number. So if the 3rd index for that example has the highest probability, the predicted *k* is 3. That means we need to extract the index for each example. 

We can use the  in-built `max()` function to do this. When 2 outputs are extracted from `max()`, the first output, which we defined as `prob` is the highest probability, and the second output, `p`, is the index (which is also the predicted number!).

The first argument of `max()` is the matrix or vector to find the maximum of. `A` in this case. Apparently to return maximum of the row, the second argument is blank, and `2` is the third argument. 

Now we have `p`! This is a vector of 5000 elements that has the predicted value.

### 2 Neural Networks

In this exercise, we will use the `ex3_nn.m`, and load the data `ex3weights.mat`. 

Be sure to look at Figure 2 in the pdf. That is going to help a lot. Here, we are going to use the `predict.m` file. And just like the last section, we are going to predict the number in *k* for each example. But instead of doing a 1 vs all logistic regression we will use a neural network. So already, we don't need 10 sets of $\theta$s. 

For this example, we also don't need to calculate the the theta like we had to do for the last part. The $\Theta$ are already supplied to us in `Theta1` and `Theta2`. 

Lets define what we know:  
m = 5000  
num_labels = 10  
X = [5000 x 401]  
Theta1 = [25 x 401]  
Theta2 = [5000 x 10]  

The first thing we need to do is as the bias to our input layer. This is the same as adding a column of 1s: `X = [ones(m,1) X]`. And remember that we can also call our input layer a1.

That means our second layer, the hidden layer is $a^{(2)}$. How do we go from one layer to the next? Its actually pretty simple, its essentially a logistic regression of the of the prior layer and $\Theta$s. 

So `z2` will be `X * Theta1'`  
This is a [5000 x 401] x [401 x 25] multiplication. This hidden layer has 25 nodes. The resultant `z2` is **[5000 x 25]**. And now we take *g($z^{(2)}$)* to get $a^{(2)}$ (using the sigmoid function). 

But now we add the bias layer to $a^{(2)}$, just like we did with X:
`a2 = [ones(m,1) a2]`.

Ok now we can work on the output layer. We know that $z^{(3)}$ = $\Theta^{(2)}$$a^{(2)}$. And we have $\Theta^{(2)}$ and $a^{(2)}$! 

So we use the code: `z3 = a2 * Theta2'` to get $z^{(3)}$, then get the sigmoid of that to get $a^{(3)}$. This results in a [5000 x 10] matrix. 

Now, just like the last example, we can use the `max()` function to determine the location of the maximum hypothesis. 

Thats it!



