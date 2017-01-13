## Introduction
### Supervised Learning
In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

Supervised learning problems are categorized into "regression" and "classification" problems. In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.

### UnSupervised Learning
Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

We can derive this structure by clustering the data based on relationships among the variables in the data.

With unsupervised learning there is no feedback based on the prediction results.

## Gradient Descent(Parameter Learning)
So we have our hypothesis function and we have a way of measuring how well it fits into the data. Now we need to estimate the parameters in the hypothesis function. That's where gradient descent comes in.

The gradient descent algorithm is:

repeat until convergence:
$$
\theta_j :=  \theta_j - \alpha \frac{\partial{J(\Theta)}}{\partial{\theta_j}}
$$
where j represents the feature index number.

(At each iteration j, one should simultaneously update the parameters.)

We should adjust our parameter α to ensure that the gradient descent algorithm converges in a reasonable time. Failure to converge or too much time to obtain the minimum value imply that our step size is wrong.

- If α is too small, gradient can be slow.
- If α is too large, gradient decent can overshoot the minimum. It may fail to converge, or even diverge.

Gradient descent can converge to local minimum, even with learning rate fixed. Because $\frac{\partial{J(\theta)}}{\partial{\theta_j}}$ will be small when close to the minimum.

## Multivariable Linear Regression
Linear regression with multiple variables is also known as "multivariate linear regression".

$$
\begin{align*}x_j^{(i)} &= \text{value of feature } j \text{ in the }i^{th}\text{ training example} \newline x^{(i)}& = \text{the column vector of all the feature inputs of the }i^{th}\text{ training example} \newline m &= \text{the number of training examples} \newline n &= \left| x^{(i)} \right| ; \text{(the number of features)} \end{align*}
$$

The multivariable form of the hypothesis function accommodating these multiple features is as follows:
$$
\begin{align*}
h_\theta (x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \cdots + \theta_n x_n \newline h_\theta(x) =\begin{bmatrix}\theta_0 \hspace{2em} \theta_1 \hspace{2em} ... \hspace{2em} \theta_n\end{bmatrix}\begin{bmatrix}x_0 \newline x_1 \newline \vdots \newline x_n\end{bmatrix}= \theta^T x
\end{align*}
$$

Remark: Note that for convenience reasons in this course we assume $x_0^\left(i\right)=1$ for $x_{0}^{(i)} =1 \text{ for } (i\in { 1,\dots, m } )$. This allows us to do matrix operations with theta and x. Hence making the two vectors $\theta$ and $x^\left(i\right)$ match each other element-wise (that is, have the same number of elements: n+1).

### Cost function
$$
J(\theta) = \frac{1}{2m}\sum\limits_{i=1}^{m}\left(h_\theta(x^\left(i\right)) - y^\left(i\right))\right)^2
$$

The mean is halved ($1/2$) as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the $1/2$ term.

### Gradient descent
When specifically applied to the case of linear regression, a new form of the gradient descent equation can be derived. We can substitute our actual cost function and our actual hypothesis function and modify the equation to :
$$
\begin{align*} & \text{repeat until convergence:} \; \lbrace \newline \; & \theta_0 := \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_0^{(i)}\newline \; & \theta_1 := \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_1^{(i)} \newline \; & \theta_2 := \theta_2 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_2^{(i)} \newline & \cdots \newline \rbrace \end{align*}
$$

Note that we have separated out the two cases for $\theta_j$ into separate equations for $j = 0$ and $j != 0$ ($x_0$ which value is 1 is added to features corresponding to $\theta_0$ for computation convenience)

### Feature Scaling
We can speed up gradient descent by having each of our input values in roughly the same range. This is because θ will descend quickly on small ranges and slowly on large ranges, and so will oscillate inefficiently down to the optimum when the variables are very uneven.

The way to prevent this is to modify the ranges of our input variables so that they are all roughly the same. Ideally: $-1\leq x_j\leq1$ or $-0.5\leq x_j\leq 0.5$

These aren't exact requirements; we are only trying to speed things up. The goal is to get all input variables into roughly one of these ranges, give or take a few.

Two techniques to help with this are **feature scaling** and **mean normalization**. Feature scaling involves dividing the input values by the range (i.e. the maximum value minus the minimum value) of the input variable, resulting in a new range of just 1. Mean normalization involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero.

### Polynomial Regression
We can combine multiple features into one. Our hypothesis function need not be linear (a straight line) if that does not fit the data well. We can change the behavior or curve of our hypothesis function by making it a quadratic, cubic or square root function (or any other form).

## Normal Equation
In the "Normal Equation" method, we will minimize J by explicitly taking its derivatives with respect to the θj ’s, and setting them to zero. This allows us to find the optimum theta without iteration. The normal equation formula is given below:
$$
\theta = \left(X^T X\right)^{-1}X^Ty
$$

There is no need to do feature scaling with the normal equation.

The following is a comparison of gradient descent and the normal equation:

| Gradient Descent | Normal Equation |
| ---------------- | --------------- |
| Need to choose alpha | No need to choose alpha|
| Needs many iterations | No need to iterate |
| Works well when n is large | Slow if n is very large|
| $O(k n^2)$| $O(n^3)$, need to calculate inverse of $X^TX$|

With the normal equation, computing the inversion has complexity (n3). So if we have a very large number of features, the normal equation will be slow. In practice, when n exceeds 10,000 it might be a good time to go from a normal solution to an iterative process.

## Logistic Regression Model
### Hypothesis Representation
***Binary classification***

We could approach the classification problem ignoring the fact that y is discrete-valued, and use our old linear regression algorithm to try to predict y given x. It also doesn’t make sense for $h_\theta(x)$ to take values larger than 1 or smaller than 0 when we know that $y ∈ {0, 1}$. To fix this, let’s change the form for our hypotheses $h_\theta(x)$ to satisfy $0 \leq h_\theta(x) \leq 1$. This is accomplished by plugging $\theta^Tx$ into the Logistic Function.

"Sigmoid Function," also called the "Logistic Function":
$$
\begin{align*}& h_\theta (x) = g ( \theta^T x ) \newline \newline& z = \theta^T x \newline& g(z) = \dfrac{1}{1 + e^{-z}}\end{align*}
$$

### Decision Boundary
In order to get our discrete 0 or 1 classification, we can translate the output of the hypothesis function as follows:
$$
\begin{align*}& h_\theta(x) \geq 0.5 \rightarrow y = 1 \newline& h_\theta(x) < 0.5 \rightarrow y = 0 \newline\end{align*}
$$

When the logistic function input greater than 0, its output is greater than or equal to 0.5

$$ g(z) \geq 0.5 $$ when $$ z \geq 0\ $$

So $$ \theta^Tx = 0 $$ is the decision boundary

### Cost function && Gradient Descent

Non-convex function has many local optima. So we cannot use the same cost function that we use for linear regression because the Logistic Function is not convex, will cause the output to be wavy, causing many local optima.

We use cost function like this:
$$
J(\theta) = - \frac{1}{m} \displaystyle \sum_{i=1}^m [y^{(i)}\log (h_\theta (x^{(i)})) + (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))]
$$

Corresponding gradient descent is identical the one used in linear regression:
$$
\begin{align*} \theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}\end{align*}
$$

See also: BFGS, L-BFGS, Conjugate gradient, fminunc(Octave)

### Multiclass Classification: One-vs-all
Train a logistic regression classifier $h_\theta(x)$ for each class￼ to predict the probability that ￼$￼y = i$ ￼.

To make a prediction on a new x, pick the class ￼that maximizes $h_\theta(x)$

### Solving Overfitting
Add regularization to cost function
$$
\dfrac{1}{2m}\ \left[ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\ \sum_{j=1}^n \theta_j^2 \right]
$$

We will modify our gradient descent function to separate out $\theta_0$ from the rest of the parameters because we do not want to penalize $\theta_0$.

The λ, or lambda, is the regularization parameter. It determines how much the costs of our theta parameters are inflated.

### Linear Regression
$$
\begin{align*} \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \newline \theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right]\end{align*}
$$

### Normal Equation
$$
\begin{align*}& \theta = \left( X^TX + \lambda \cdot L \right)^{-1} X^Ty \newline& \text{where}\ \ L = \begin{bmatrix} 0 & & & & \newline & 1 & & & \newline & & 1 & & \newline & & & \ddots & \newline & & & & 1 \newline\end{bmatrix}\end{align*}
$$

### Logistic Regression
The same as linear regression

## Neural Network
### Forward Propagation
$$
\begin{align*}& a_i^{(j)} = \text{"activation" of unit $i$ in layer $j$} \newline& \Theta^{(j)} = \text{matrix of weights controlling function mapping from layer $j$ to layer $j+1$}\end{align*}
$$

Lets $a_k^j=g(z_k^j)$. For layer $j=i+1$ and node k, the variable z will be:
$$
z_k^{(i+1)} = \Theta_{k,0}^{(i)}x_0 + \Theta_{k,1}^{(i)}x_1 + \cdots + \Theta_{k,n}^{(i)
  }x_n
$$
We also can write the equation as:
$$
z^{(j)} = \Theta^{(j-1)}a^{(j-1)}
$$

We are multiplying our matrix $\Theta^{j-1}$ with dimensions $s^j × (n+1)$ (where $s^j$ is the number of our activation nodes) by our vector $a^{j-1}$ with height $(n+1)$. This gives us our vector $z(j)$ with height $s^j$.

### Cost Function
Variables that we will need to use:

- L = total number of layers in the network
- $s_l$ = number of units (not counting bias unit) in layer l
- K = number of output units/classes

Recall that in neural networks, we may have many output nodes. We denote $h_\Theta(x)_k$ as being a hypothesis that results in the $k^{th}$ output. Our cost function for neural networks is going to be a generalization of the one we used for logistic regression.
$$
\begin{gather*} J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j, i}^{(l)})^2\end{gather*}
$$

### Back Propagation
"Backpropagation" is neural-network terminology for minimizing our cost function, just like what we were doing with gradient descent in logistic and linear regression. Our goal is to compute:
$$
\min_\Theta J(\Theta)
$$
That is, we want to minimize our cost function J using an optimal set of parameters in theta. In this section we'll look at the equations we use to compute the partial derivative of $J(\Theta)$:
$$
\dfrac{\partial}{\partial \Theta_{i,j}^{(l)}}J(\Theta)
$$

Given training set {(x(1),y(1))⋯(x(m),y(m))}

Set $\Delta^{(l)}_{i,j}$ := 0 for all (l,i,j), (hence you end up having a matrix full of zeros)

For training example t =1 to m:

1. Set $a^{(1)}:=x^{(t)}$
2. Perform forward propagation to compute $a^{(l)}$ for l=2,3,…,L
3. Using $y^{(t)}$, compute $\delta^{(L)} = a^{(L)} - y^{(t)}$

   Where L is our total number of layers and a(L) is the vector of outputs of the activation units for the last layer. So our "error values" for the last layer are simply the differences of our actual results in the last layer and the correct outputs in y. To get the delta values of the layers before the last layer, we can use an equation that steps us back from right to left:

4. Compute $\delta^{(L-1)}, \delta^{(L-2)},\dots,\delta^{(2)}$ using $\delta^{(l)} = ((\Theta^{(l)})^T \delta^{(l+1)})\ .*\ a^{(l)}\ .*\ (1 - a^{(l)})$

The delta values of layer l are calculated by multiplying the delta values in the next layer with the theta matrix of layer l. We then element-wise multiply that with a function called $g'$, or g-prime, which is the derivative of the activation function $g$ evaluated with the input values given by $z^{(l)}$.

The g-prime derivative terms can also be written out as:
$$
g'(z^{(l)}) = a^{(l)}.* (1 - a^{(l)})
$$

5. $\Delta^{(l)}_{i,j} := \Delta^{(l)}_{i,j} + a_j^{(l)} \delta_i^{(l+1)}$ or with vectorization, $\Delta^{(l)} := \Delta^{(l)} + \delta^{(l+1)}(a^{(l)})^T$

Hence we update our new Δ matrix.

- $D^{(l)}_{i,j} := \dfrac{1}{m}\left(\Delta^{(l)}_{i,j} + \lambda\Theta^{(l)}_{i,j}\right)$, if j ≠ 0
- $D^{(l)}_{i,j} := \dfrac{1}{m}\Delta^{(l)}_{i,j}$, if j =0

The capital-delta matrix D is used as an "accumulator" to add up our values as we go along and eventually compute our partial derivative. Thus we get $\frac \partial {\partial \Theta_{ij}^{(l)}} J(\Theta)$

### Gradient Checking
Gradient checking will assure that our backpropagation works as intended. We can approximate the derivative of our cost function with:
$$
\dfrac{\partial}{\partial\Theta_j}J(\Theta) \approx \dfrac{J(\Theta_1, \dots, \Theta_j + \epsilon, \dots, \Theta_n) - J(\Theta_1, \dots, \Theta_j - \epsilon, \dots, \Theta_n)}{2\epsilon}
$$

A small value for ϵ (epsilon) such as $ϵ=10^{−4}$, guarantees that the math works out properly. If the value for ϵ is too small, we can end up with numerical problems.

### Random Initialization

Initializing all theta weights to zero does not work with neural networks. When we backpropagate, all nodes will update to the same value repeatedly. Instead we can randomly initialize our weights for our $\Theta$ matrices in $[-\epsilon,\epsilon]$

## Evaluating a Learning Algorithm
The model fitting training set well doesn't mean it's a good model(maybe overfit).

Training: 60% Cross validation: 20% Testing: 20%. Using cross validation to choose the model and Using test set to measure the model.

High Bias & High Variance:

- High bias (underfit): $J_{train}(\theta)$ is high and $J_{cv}(\theta)\approx J_{train}(\theta)$
- High variance problem(overfit): $J_{train}(\theta)$ is low and $J_{cv}(\theta) >> J_{train}(\theta)$

The lambda $\lambda$ of Regularization:

- Large $\lambda$ ~ high bias: $\Theta$ is close to 0
- Small $\lambda$ ~ high variance

The training size:

- High Bias: If a learning algorithm is suffering from high bias, getting more training data will be not help much. After a certain training size, the error $J_{cv}(\theta)\approx J_{train}(\theta)$ and $J_{train}(\theta)$ is high
- High Variance: If a learning algorithm is suffering from high variance, getting more training data is likely to help. There is always a large gap between $J_{cv}(\theta)$ and $J_{train}(\theta)$

### Error analysis on skewed classes
$$ Precision = \frac{TP}{TP+FP}$$
$$ Recall = \frac{TP}{TP+FN}$$

- Suppose we want to predict $y=1$ only if very confident: Higher precision, lower recall
- Suppose we want to avoid missing too many case(false negative): higher recall, lower precision

$F_1$ Score: $\frac{2PR}{P+R}$

## Support Vector Machine

### Cost Function
$$
C\sum\limits_{i=1}^{m}\left[y^{(i)}cost_1(\theta^T x^{(i)}) + (1-y^{(i)})cost_0(\theta^T x^{(i)})\right]+\frac{1}{2}\sum\limits_{i=1}^{n}\theta_i^2
$$

We could look $C$ like $\frac{1}{\lambda}$

If $y=1$, we want $\theta^Tx \geq 1$

If $y=0$, we want $\theta^Tx \leq -1$

How SVM get the proper margin(large margin)?

$$\theta^Tx^{(i)} = p^{(i)} . ||\theta||$$

$p^{(i)}$ is the $x^{(i)}$ projection length to $\theta$

When $y=1$, we want $\theta^Tx \geq 1$ ($p^{(i)} . ||\theta|| \geq 1$) and we also want to minimize to cost function. So we need the $p^{(i)}$ should be large.

Notice that the vector $\theta$ is vertical to decision boundary.

### Kernels
1. Choose several landmarks $l^{(1)}, l^{(2)}, ..., l^{n}$ (Choose from training set)
2. for each $x^{(i)}$, compute the similarity between $x^{(i)}$ and $l^{(j)}$ to generate new feature $f_j=exp\left(-\frac{||x-l^{(j)}||^2}{2\sigma ^2}\right)$ (Gaussian kernel, perform feature scaling before using it)
3. Train $\theta$ based on the new feature
4. Predict "1" when $\theta ^T f \geq 0$

$\sigma ^2$

- Large $\sigma ^2$: feature f vary more smoothly. higher bias, lower variance
- Small $\sigma ^2$: feature f vary less smoothly. lower bias, higher variance

### Logistic regression VS. SVMs
- When $n$ is large relative to $m$, using Logistic regression or SVM without kernel
- When $n$ is small and $m$ is intermediate, using SVM with Gaussian kernel
- When $n$ is small and $m$ is large, create more features, then use logistic regression or SVM without kernel

## PCA
Reduce from n-dimension to k-dimension: Find k vectors $u^{(1)}, u^{(2)}, ..., u^{(k)}$ on which to project the data so as to minimize the project error.

1. Data Preprocessing: *mean normalization*, feature scaling
2. Compute convariance matrix $\Sigma = \frac{1}{m}\sum\limits_{i=1}^{n}(x^{(i)})(x^{(i)})^T$ ($\Sigma = X^T X$)
3. Compute the eigenvectors of matrix $\Sigma$ $[U, S, V] = svd(Sigma)$
4. Select the first $k_{th}$ column of U as $u^{(1)}, u^{(2)}, ..., u^{(k)}$
5. Reduce origin data to k-dimension: $Z = [u^{(1)}, u^{(2)}, ..., u^{(k)}]^T X$ (reconstruct the data, $x = [u^{(1)}, u^{(2)}, ..., u^{(k)}] z$)

## Multivariable Gaussian Distribution
$$f_Y(x)=\frac{1}{\sqrt{(2\pi)^n|\boldsymbol\Sigma|}}
\exp\left(-\frac{1}{2}({x}-{u})^T{\boldsymbol\Sigma}^{-1}({x}-{u})
\right)$$

$$
\begin{align}
u = \frac{1}{m}\sum\limits_{i=1}^{m}x^{(i)} \newline
\Sigma = \frac{1}{m}\sum\limits_{i=1}^{m}(x^{(i)}- u)(x^{(i)}-u)^T
\end{align}
$$

## Stochastic Gradient Descent, Mini-Batch Gradient Descent
