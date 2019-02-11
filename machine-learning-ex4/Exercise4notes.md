## Exercise 4

### 1 Neural Networks

The goal of this exercise is to handwritten digits, but we will create an algorithm to learn the weights of the parameters. In the last exercise, the weights were given to us. 

The data is from `the ex3data1.mat` which is a 5000 x 400 matrix. It has 5000 examples of 400 pixel brightnesses (of a 20x20 pixel picture of a number). 

The 'y' is a 5000x1 vector of labels from 0-9 which is the correct label for the corresponding training example.

Our neural network has 3 layers: an input layer, a hidden layer, and an output layer. The input layer has 400 units, the number of parameters of the training examples. 

The hidden layer has 25 units. Each unit of the hidden layer will receive input from each of the units of the input layer. That means `Theta1` is a 25x401 matrix if the bias unit is included. 

The output layer has 10 layers (one for each answer). So each layer will take an input from the hidden layer (and bias). Thus `Theta2` is a 10x26 matrix. 

#### 1.3 Feedforward and cost function

Here we will be working with the `nnCostFunction.m` file. This function calculates the cost function for a neural network with two layers. 

The *outputs* are `J` which is the cost, and `grad` which is the gradient. 

There are a couple inputs:  
- **nn_params** = unrolled neural network parameters. 
  - In the `ex4.m` file, this is created with the code `nn_params = [Theta1(:) ; Theta2(:)]` which unrolls all the parameters into one vector. This vector is (25x401)+(10x26) elements long. 
- **input_layer_size** = 400, this is number of input parameters
- **hidden_layer_size** = 25, number of units int he hidden layer
- **num_labels** = 10, the number of output classes
- **X** = 5000x400 matrix of examples
- **y** = 5000x1 matrix of the labels (from 0-9)
- **lambda** = this is for regularization

Here is the equation for the cost function:

J($\theta$) = $(1/m)$ * $\sum_{i=1}^{m}$ $\sum_{k=1}^{K}$ [-$y_{k}^{(i)}$ $log(h_{\theta}x^{(i)})_{k}$ - (1-$y_{k}^{(i)}$)*(1-$log(h_{\theta}x^{(i)})_{k}$)]  

Essentially, this is cost of a logistic regression, but for each example $m$ at each y label $K$. 

To make sure that works, we need to make y into a format that is workable. We will come back to this. 

The first thing we need to do is to create a matrix of the parameters. the `ex4.m` file has a line of that unrolls the parameters, so we need to roll the parameters into the respective matrices. 

we will use the `reshape()` function for this. This takes 3 arguments.  
1. the range of number from that vector to take from. We are going from 1 to the size of the hidden layer (25) * input layer size plus 1 (400+1). 
2. number of matrix rows, `hidden_layer_size` is 25
3. number of matrix colums `input_layer_size + 1` is 401

This will give us `Theta1`, a [25x401 matrix]. We do the same this to get `Theta2`. 

The rest of the variables are initialized. Now we move onto the actual part of the exercise.  

##### Part 1 of nnCostFunction - Feedforward

In part 1, we implement a feedforward neural network to return the cost `J`. We need to calculate the hypothesis of the neural network. This is similar to the previous exercise, so we won't spend as much time on this: 

`X = [ones(size(X,1),1)  X];` adds a column of 1s to the X matrix. Now a [5000 x 401 matrix].  

`z2 = X * Theta1';`  
`a2 = sigmoid(z2);`  
`Theta1` is a [25x401] matrix. So we inverse that when we multiply it by the input `X`. Then take the sigmoid of that to get the activation matrix of the hidden layer, a [5000 x 25] matrix. 

Then we add a layer of 1s to make a [5000 x 26] matrix:  
`a2 = [ones(size(a2, 1),1)  a2];`  

We multiple the activations of the hidden layer by the inverse of `Theta2`.  This is a [5000 x 26] x [26 x 10] matrix mulitplication, so now we get z3 which is a [5000x10] matrix. We take the sigmoid of that which is the activation of the third layer!  This is also our hypothesis. 
`z3 = a2 * Theta2';`  
`a3 = sigmoid(z3);`  

**What exactly does this represent?**   
Well these are the probabilities that the example $m$ belongs to each class in $K$. The $K$ is the column of the matrix, where each column represents a number label. For example column 1 represents the label $1$, and a high probability there means that the example is likely a 1. the column 3 represents the label $3$ and so on. And of note, column 10 represents the label $0$. 

Now this is *not a trained model*. So all these are really going to be random percentages. We need to train this model against the correct labels. 

This means need labels that our model can understand. The current format of `y` is vector of numbers from $0-9$. Instead, what we'll give each example 10 columns, where a $1$ indicates the label, and a $0$ is given for all the other numbers. For example, $3$ would be {0, 0, 1, 0, 0, 0, 0, 0, 0, 0}.

We will use boolian logic to do that. Our method is vectorized, rather than using a for loop:  
`y_label = (1:num_labels) == y;`  
This creates a [5000x10] matrix of y labels where the column where the 1 (TRUE) is, indicates the label.  

Now we can use the cost equation to calculate the cost:  
`J = (-1/m) * sum(sum( y_label .* log(hx) + ((1-y_label) .* log(1-hx))));`

This does an element muliplication of the actual label `y_label` and the predicted label `log(hx)` for each example. Then it adds up the cost of all examples (m, the first `sum`) and then for each column example (k, the second `sum`).

**Cost With regularization**  
This just adds the regularization term:  
`+ (lambda/(2*m)) * (sum(sum(Theta1(:,2:end) .^2)) + sum(sum(Theta2(:,2:end) .^2 )));` 
Remember that the first term of each `theta1` is the bias, as that is not regularized. So we use `Theta( : , 2:end)` to subset all the rows, but only use the columns from 2 to the end. 

### 2 Backpropogation

##### 2.1 Sigmoid gradient  
First we need to write a function that takes the tells you the *gradient* of a sigmoid value, like we did for logistic regression. 

Think about a sigmoid curve. If the input is a large number, like 100, the gradient (or derivative) is 0. 

In `sigmoidGradient.m`, output `g` is returned from input `z`. And this should work for vectors or matrices, so we'll initialize `g` as zeros as the size of `z`. 

first we take the sigmoid of the input. Then we calculate the derivative: `g = gz .* (1-gz);`

##### 2.2 Random initialization  
The file `randInitializeWeights.m` is a function that creates random inital weights for $\Theta$. It takes two arguments:  
`L_in` which is $s_l$  
`L_out` which is $s_{l+1}$

Then we create a matrix of size [`L_out` x `L_in + 1`] and mulitply/subtract by an initial epsilon.  

When the `Theta1` is initialized, it is initialized like this:  
`initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);`

And `Theta2 is initialized like this:  
`initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);`  

This way, a [25x401] matrix of random numbers is created for `initial_Theta1` and [10x26] matrix of random numbers is created for `initial_Theta2`.  

##### 2.3 back propogation 
##### Part 2 of nnCostFunction - Backpropogation  
Likely unnecesary, but for the first few lines, I rewrote the forward propogation code to get `a3`, the activation of the output layer. Recall that this is a [5000x10] matrix, and represents the activation of each example for each of the output nodes (labels of 0-9).

Now that we have `a3`, the activation of the last layer, we need to calculate the error for each layer, the $\delta_{j}^{(l)}$.

$\delta_{j}^{(3)}$, the error in layer 3 is actually easy to calculate. Its just $a_{j}^{(3)}$ - $y_k$.

Lets take a quick step back and recall that we made a new variable `y_label` from `y`. This is a [5000x10] matrix with the 1 indicating the column that the label is. 

Ok, going back to $\delta_{j}^{(3)}$ = $a_{j}^{(3)}$ - $y_k$  
We use this code to calculate that:
`delta3 = a3 - y_label;`  

Thats easy.  

The $\delta_j$ of the other layers is a little harder. Basically, we get $\delta_j$ of layer $l$ by multiplying the next layer's error, the $\delta_j^{l+1}$, by the current layer's $\Theta^l$. And then multiplying that by the gradient of the activation at layer $l$. 

*What does that look like??*  
Here:  
$\delta_j^{2}$ = $(\Theta^{(2)})^T$ * $\delta_j^{3}$ * $g'(z^{(2)})$

Lets break each components down: 
- $(\Theta^{(2)})^T$ = parameters of the second later
- $\delta_j^{3}$ = Error of the third layer
- $g'(z^{(2)})$ = gradient of the sigmoid activation the second layer

Now lets look at the code:  
`delta2 = (delta3 * Theta2) .* [ones(size(z2,1),1) sigmoidGradient(z2)]; `

Remember that `delta3` is a [5000x10] matrix, and `Theta2` is a [10x26] matrix. We use matrix multiplication to get a [5000x26] matrix. 

We do an element multiplication by the gradient for that parameter. Remeber that we need to add ones for the bias.  

What is this doing? Well this adjusts each parameter based on sigmoid activation. A high activation (far right on the sigmoid curve) will provide a gradient (slope) of 0. 

By itself, `sigmoidGradient(z2)` will provide a [5000x25] matrix, but we use `ones(size(z2,1),1)` to add ones, making a [5000x26] matrix. So we can use it in the element-wise multiplation.  

**Using $\delta^{(2)}$ to calulate the gradients of $\Theta^{(1)}$**  
Then we need to remove the first column of bias  
`delta2 = delta2(:, 2:end);`

Now there is no error for the first layer because inputs don't have errors.  

Lets talk about what we are doing. `delta2` or $\delta^{(2)}$ is the error of each of the 5000 training examples along each of the 25 nodes in the hidden layer. We multiply this by activation of the previous layer (which is just X in this case). The activation of the previous layer is the brightness of each pixel for each 5000 examples. 

If we mulitply the error of layer 2 by the activations in layer 1, we get the gradient of $\Theta^{(1)}$

Here's the code:  
`Theta1_grad = (1/m) * (delta2' * a1);`
`delta2` is a [5000x25] matrix because it is the error for each example at each node. And we inverse that to get [25x5000]  
`a1` is just `X`. Thats a [5000x401] matrix. 

When we multiply `delta2` [25x5000] by `a1` [5000x401] we get a [25x401] matrix of the gradients for $\Theta^{(1)}$. Notice this matrix is the same size at `Theta1` too!

 **Using $\delta^{(3)}$ to calulate the gradients of $\Theta^{(2)}$**  
To calculate the *gradients* of $\Theta^{(2)}$, `Theta2_grad`, we essentially do the same thing. Here's the formula: `Theta2_grad = (1/m) * (delta3' * a2);`  

Remember that `delta3` is [5000x10],so we need to inverse it. And `a2` is a [5000x26] matrix, so `delta3 * a2` is a [10x5000]x[5000x26] matrix multiplication, resulting in a `Theta2_grad` that is [10x26], exactly the dimensions of `Theta2`

##### Part 3 of nnCostFunction - Backpropogation  with regularization

This part is actually pretty easy. Just remember that the first term is not regularized. So we only need to regularize everything byt the first column. 

With this line of code:
`Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda/m) * Theta1(:,2:end));`  
We are adding the $\lambda/m * \Theta_1$ to all the gradients except the first column. 