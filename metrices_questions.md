
1. Why do we say that matrices are linear transformations?
1. What’s the inverse of a matrix? Do all matrices have an inverse? Is the inverse of a matrix always unique?
1. What does the determinant of a matrix represent?
1. What happens to the determinant of a matrix if we multiply one of its rows by a scalar  $t×R$ ?
1. A $4×4$  matrix has four eigenvalues $3,3,2,−1$. What can we say about the trace and the determinant of this matrix?
1. Given the following matrix:

| 1 | 4 | -2 |
| -1 | 3 | 2 |
| 3 | 5 | -6 |

Without explicitly using the equation for calculating determinants, what can we say about this matrix’s determinant?
1. What’s the difference between the covariance matrix $A^TA$  and the Gram matrix $AA^T$ ?
1. Given $A∈R^{n×m}$  and $b∈R^n$ 

    1. Find $x$ such that: $Ax=b$.
    1. When does this have a unique solution?
    1. Why is it when $A$ has more columns than rows, $Ax=b$ has multiple solutions?
    1. Given a matrix $A$ with no inverse. How would you solve the equation  $Ax=b$ ? What is the pseudoinverse and how to calculate it?
1. Derivative is the backbone of gradient descent.
    1. What does derivative represent?
    1. What’s the difference between derivative, gradient, and Jacobian?
1. Say we have the weights $w∈R^{d×m}$  and a mini-batch $x$  of $n$  elements, each element is of the shape $1×d$  so that $x∈R^{n×d}$. We have the output $y=f(x;w)=xw$. What’s the dimension of the Jacobian $\frac{δy}{δx}$?
1. Given a very large symmetric matrix $A$ that doesn’t fit in memory, say $A∈R^{1M×1M}$  and a function $f$ that can quickly compute $f(x)=Ax$ for $x∈R1M$. Find the unit vector $x$ so that $x^TAx$  is minimal.