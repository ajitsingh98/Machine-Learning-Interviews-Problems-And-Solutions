# Calculus and Convex Optimization Questions


1. Differentiable functions
    1.  What does it mean when a function is differentiable?
    1. Give an example of when a function doesn’t have a derivative at a point.
    1. Give an example of non-differentiable functions that are frequently used in machine learning. How do we do backpropagation if those functions aren’t differentiable?
2. Convexity
    1. What does it mean for a function to be convex or concave? Draw it.
    1. Why is convexity desirable in an optimization problem?
    1. Show that the cross-entropy loss function is convex.
3. Given a logistic discriminant classifier:
$
p(y=1|x)=σ(w^Tx)
$
where the sigmoid function is given by:
$
σ(z)=(1+exp(−z))^{−1}
$
The logistic loss for a training sample $x_i$  with class label $y_i$  is given by $L(yi,xi;w)=−logp(y_i|x_i)$
    1. Show that  $p(y=−1|x)=σ(−w^Tx)$.
    1. Show that  $Δ_wL(y_i,x_i;w)=−y_i(1−p(y_i|x_i))x_i$.
    1. Show that  $Δ_wL(y_i,x_i;w)$  is convex.
4. Most ML algorithms we use nowadays use first-order derivatives (gradients) to construct the next training iteration.
    1. How can we use second-order derivatives for training models?
    1. Pros and cons of second-order optimization.
    1. Why don’t we see more second-order optimization in practice?
5. How can we use the Hessian (second derivative matrix) to test for critical points?
6. Jensen’s inequality forms the basis for many algorithms for probabilistic inference, including Expectation-Maximization and variational inference.. Explain what Jensen’s inequality is.
7. Explain the chain rule.
8. Let $x∈R_n$ , $L=crossentropy(softmax(x),y)$ in which $y$  is a one-hot vector. Take the derivative of $L$  with respect to $x$.
9. Given the function $f(x,y)=4x^2−y$  with the constraint $x^2+y^2=1$. Find the function’s maximum and minimum values.