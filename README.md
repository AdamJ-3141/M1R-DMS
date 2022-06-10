# M1R-DMS
Discrete minimal surfaces on closed boundary loops using gradient descent.

Requires: NumPy, Matplotlib
Download the python file, change the variables at the end of the code to change the output.

`f`: Functions R to R³ in terms of parameter t. Must be written in pythonic form, i.e. `math.sin(2*x)` rather than `sin(2x)`.
`domain`: 2-tuple (a,b) denoting the minimum and maximum t-value. Note that f(a) must equal f(b)
`num_iters`: Number of iterations used
`n`: Number triangles along one side of the square matrix, detail parameter - increase to large values at your own peril.
`a`: The value for alpha as described in the poster.
