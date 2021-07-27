# DynAutoDiff - A C++ 20 Header-Only Dynamic Automatic Differential Library for Statistics and ML
[![purple-pi](https://img.shields.io/badge/Rendered%20with-Purple%20Pi-bd00ff?style=flat-square)](https://github.com/nschloe/purple-pi?activate)

![Introduction](Main.png)

## Introduction 

### Why Another Auto-Diff library?

There have been many excellent automatic differential libraries: `autodiff`, `Adept`, `stan-math`, `pytorch`, `tensorflow`, `FastAD`... But none of them servers as my ideal:

1. Some are too big (`pytorch`, `tensorflow`) and depend on many external libraries. A series problem is that these libraries may be conflict with other libraries which I need due to linking problems. One example is `pytorch` can't be linked together with `Ceres` (an optimizing library) or there will be segmentation fault.

2. Some lacking abilities to deal with matrix differential (e.g. *matrix inverse*, *det*, ...). As far as I know, except `FastAD` and `stan-math`, other libraries don't deal with matrix differential.

3. `stan-math` has bad documentations. I can't even successfully run an example. Also it depends on TBB and SUNDIALS which I seldom need.

4. `FastAD` is inconvenient. It uses expression template technique and is better for static expressions. It's hard to manipulate expressions programmatically. However it's fast.

That's why I decide to write an automatic differential library by myself. The design of this library is inspired by `pytorch` and `FastAD`. **Dynamic** is the first goal, so naturally it should be slower than `FastAD`. If you are sensitive to speed, I suggest you to try `FastAD`.

### Core Idea

Data is represented by a matrix: a scalar is a $1\times1$ matrix, a column is a $n\times 1$ matrix. The matrix is store as `Eigen::Map` to support both automatic memory management or passing external storage.

The type for variable is `Var<T=double>`. It stores value and gradient data, evaluation functor, gradient functor. But the real type that involved in expressions is `shared_ptr<Var<T>>`. When an expression `x+y` is created, a graph is created and no computation is done.

The actual computation is stared when user call `eval()` and then call `backward()` for gradient computation. In this stage, evaluation and gradient functors will be called.

There is a `GraphManager` that helps manage a graph. For example it can track all previous nodes given a final node. It can clear gradient data (before a fresh evaluation is begin, gradient data should be set to zero). A variable has a flag `_evaluated` to avoid repeating computation. This flag can also be cleared by `GraphManager`.


## Usage and Examples

### Installation

First clone this project:
```
git clone https://github.com/kilasuelika/DynAutoDiff.git
```

Then run the hello world example (see bellow):
```
cmake .
make
```

The main part of this library is a header-only library and depend on `Eigen, boost.json, boost.math` which are also header-only libraries. So just copy the header files folder `DynAutoDiff` into your projects. You compiler should support **c++20** (`constexpr, <concepts>, <numbers> ...` are used).

There are also a series of wrapper classes for `ceres` (CeresOptimizer.hpp), `NLopt` (nloptoptimizer.hpp). They are not included in `DynAutoDiff.hpp`. To use these, you need to install them by yourselves.

There is a `GDOptimizer.hpp` that implements `NaiveGD, Adam` that can be used to train a neural network. It has been included in the main library.

If you want to run tests, then you need `boost`.

If you want to run some examples, then you need to install more. For example the example *ceres_neural_network* requires `ceres` or more.

### A Hello World Example

```cpp
#include "DynAutoDiff/DynAutoDiff.hpp"

using namespace std;
using namespace DynAutoDiff;

int main() {
    // Create variables and expressions.
    auto x = vec<double>({1, 2}), Sigma = pmat<double>({2, 1, 1, 2}, 2, 2),
         y = pvec<double>({2, 3});
    auto z = transpose(x) * inv(Sigma) * y;
    GraphManager gm(z);

    // Run automatic differential.
    gm.run();
    cout << z->v() << endl << Sigma->val() << endl << Sigma->grad() << endl;

    // Save and load.
    gm.save("graph.json");
    auto z1 = gm.load("graph.json");
};
```

### Linear Regression Models

Compare OLS regression and MLE regression. You need ceres. See `examples/ceres_linear_regression`:

```cpp
#include "../../DynAutoDiff/CeresOptimizer.hpp"

using namespace std;
using namespace DynAutoDiff;

int main() {
    auto X = std::make_shared<Var<>>("../../test/X.txt");
    auto y = std::make_shared<Var<>>("../../test/y.txt");

    // Set parameters.
    auto theta_ols = pvec(X->cols());
    auto theta_mle = pvec(X->cols());
    auto c_ols = psca();
    auto c_mle = psca();

    // Losses.
    auto mse = mse_loss(X * theta_ols + c_ols, y);
    auto ll = -ln_normal_den(y - X * theta_mle - c_mle, csca(0.0), psca(1.0));

    CeresOptimizer ols_opt(mse);
    ols_opt.run();
    CeresOptimizer mle_opt(ll);
    mle_opt.run();

    cout << "OLS theta: " << endl << theta_ols->val() << endl;
    cout << "OLS c: " << endl << c_ols->val() << endl;

    cout << "MLE theta: " << endl << theta_mle->val() << endl;
    cout << "MLE c: " << endl << c_mle->val() << endl;
}
```

One would find that the result will be the same (except negligible numerical error).

### GMM Model

The full code is in `examples/ceres_GMM`. The dataset has 10 columns and 5000 samples. Assuming the data comes from two Gaussian distributions, the following code computes negative log-likelihood:

```cpp
auto X = std::make_shared<Var<>>("../../test/gmm_datat5000_10.txt");

auto I = cmat(10, 10);
*I = TMat<>::Identity(10, 10); //Set data of I matrix.

auto mu1 = prowvec<double>({10, 10, 10, 10, 10, 10, 10, 10, 10, 10});
auto sigma1 = pvec<double>( {});
auto mu2 = prowvec<double>({-10, -10, -10, -10, -10, -10, -10, -10, -10, -10});
auto sigma2 = pvec<double>( {});

auto w1 = psca(0.5);
auto p = sigmoid(w1);

auto S1 = rsdot(ivecl(sigma1)) + I; //LL^T+I for positive definite matrices.
auto S2 = rsdot(ivecl(sigma2)) + I;

auto Xmu1 = offset(X, -mu1), Xmu2 = offset(X, -mu2);
auto p1v = (1.0 / sqrt(pow(2 * numbers::pi, 10))) / sqrt(det(S1)) *
               exp(-0.5 * diag(Xmu1 * inv(S1), transpose(Xmu1)));
auto p2v = (1.0 / sqrt(pow(2 * numbers::pi, 10))) / sqrt(det(S2)) *
               exp(-0.5 * diag(Xmu2 * inv(S2), transpose(Xmu2)));
auto ll = -sum(ln(p * p1v + (1 - p) * p2v));
GraphManager<> gm(ll);
```

### Neural Network Regression

This example locates in `examples/neural_network`:

```cpp
#include "../../DynAutoDiff/CeresOptimizer.hpp"
#include <algorithm>

using namespace Eigen;
using namespace std;
using namespace DynAutoDiff;

template <typename T = double> struct NN {
    int k;
    shared_ptr<Var<T>> A1 = pmat(20, k), b1 = pvec(20), A2 = pmat(50, 20), b2 = pvec(50),
                       A3 = pmat(20, 50), b3 = pvec(20), A4 = pmat(1, 20), b4 = pvec(1);

    NN(int k) : k(k) {
        // He parameter initialization.
        for (auto &node : {A1, A2, A3, A4}) {
            node->setRandom();
            *node = node->val() / sqrt(node->cols() / 2);
        };
    };

    auto make_model(const TMat<T> &X) {
        auto y1 = relu(offset(A1 * mat(X), b1)); // 20*N
        auto y2 = relu(offset(A2 * y1, b2));     // 100*N
        auto y3 = relu(offset(A3 * y2, b3));     // 20*N
        auto y4 = relu(offset(A4 * y3, b4));     // 1*N

        return y4;
    };
};

template <typename T = double> void evaluate(const TMat<T> &test_X, const TMat<T> &test_y) {}
int main() {
    // Create train and test dataset.
    auto data = load_mat("../../test/cancer_reg.txt");
    int N = data.rows(), k = data.cols();
    data.transposeInPlace();
    for (int i = 0; i < k; ++i) {
        data.row(i).normalize();
    }

    // Generate random sample ids for training and testing.
    vector<int> all(N);
    iota(all.begin(), all.end(), 0);
    default_random_engine eng(0);
    shuffle(all.begin(), all.end(), eng);

    vector<int> train_id(all.begin(), all.begin() + 2200);
    vector<int> test_id(all.begin() + 2200, all.end());
    TMat<> train_X = data(seqN(1, k - 1), train_id);
    TMat<> train_y = data(seqN(0, 1), train_id);
    TMat<> test_X = data(seqN(1, k - 1), test_id);
    TMat<> test_y = data(seqN(0, 1), test_id);

    // Expression for training.
    NN<> net(train_X.rows());
    auto y4 = net.make_model(train_X);
    auto mse = mse_loss<Reduction::Mean>(y4, mat(train_y));
    GraphManager<> gm(mse);
    gm.auto_bind_parm();

    // Expression for testing.
    auto y4t = net.make_model(test_X);
    auto test_mse = mse_loss<Reduction::Mean>(y4t, mat(test_y));
    GraphManager<> gm1(test_mse);

    ConstLR c_scheduler(1e-6);
    Adam adam_opt(&gm, &c_scheduler);

    for (int i = 0; i < 50000; ++i) {
        adam_opt.step();
        if (i % 100 == 0) {
            gm1.run();
            cout << "Iteration " << i << ", learning rate: " << c_scheduler.lr()
                 << ", train RMSE: " << sqrt(mse->v()) << ", test RMSE: " << sqrt(test_mse->v())
                 << endl;
        }
    }

    gm.save("graph.json");
}
```

Because the shape of intermediate is decided before `gm.run()`, so we need to store the parameters in a class to easily generate graph for training and testing.

## Reference Guide

### Conventions and Be Careful

1. This library only implements *reverse mode* automatic differential.
2. `Eigen` uses `ColumnMajor` by default. But it's a little counter-intuitive. So `RowMajor` is used by this library. The internal type used is:
```cpp
template <typename T = double>
using TMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
```
3. When I say *vector*, it's default to be a column vector. Use column vectors as much as possibles to avoid confusing.
4. The default template element type are `double`, but sometime you need to explicitly specify it:
```cpp
auto x=rowvec({1,2,3}) // Warning: element will be int
auto x=rowvec<double>({1,2,3})  //Correct.
```
5. The initial values and gradients are guaranteed to be 0 for automatically allocated memory. Sometimes this may be problematic. For example, when optimizing, the target function may be undefined for 0 inputs. You must provide initial values at this situation.
6. The default reduction method for loss functions is **Sum**, not mean.
7. There will be range checks for `v(), g()` which are used to get value and gradient by index.

### Initializing Variables

#### Automatic memory management

There are a series of convinence functions for creating specific size variables: `mat, vec, rowvec, sca`. Each one has a c-version (constant, thus `requires_grad=false`) and p-version  (parameter, thus `requires_grad=true`).

1. Only specify size, allo values are initialized to be 0s:

```cpp
auto S=cmat(2,2); //2*2 size matrix
auto S=pmat(2,2); //2*2 size parameter matrix
auto v=vec(5); //column vector
auto s=csca(); // scalar.
```
2. Pass data when crating a variable:

```cpp
auto S=cmat<double>({1,2,3,1}, 2, 2); //Don't forget to set variable type to double.
auto S=cmat({1.1, 2.0, 3.0, 3.9}, 2, 2); //double type.
auto v=vec<double>({1,2,3}); //No need to set size for vector. 3*1 vector.
```

#### Passing External Storage

You can pass data pointers when initilizing:
```cpp
auto S=std::make_shared<Var<T=double>>(T* val, T* gard, rows, cols);
```

Or you can `bind` later:

```cpp
auto S=cmat(2,2);
S->bind(T* val, T* grad, rows, cols);
```

#### Reading Data from File
```
auto x=std::make_shared<Var<>>("data.txt", requires_grad=false);
```
can create a variable with values read from "data.txt". It should be a tab-delimited file without header row, not csv format. Shape is automatically decided. 

### Expressions and Operators

#### Basic Math Functions

1. `+, -, *, /`: *+, -, /* are element-wise operation (one of them can be a scalar). *\** is matrix product (or scalar-matrix product). `eprod(X,Y)`: element wise product, *X,Y* must have the same shape (thus not a scalar and a matrix). `offset(X, x)`: *X* is a matrix, *x* is a column or row vector. If *x* is a column vector, then add it to each column of *X* (thus shape must match), vice versa. `scale(X, x)`, like *offset*, but now is element-wise times.
1. Element-wise functions: `exp`, `ln`, `sqrt`,`pow(x, d)`, `sin`, `cos`,`tan`,`sinh`, `cosh`, `tanh`, `sigmoid`, `relu`.
2. `lpnorm(X, Dim=-1)`: `Dim=-1 | 0 | 1`, `-1`(all, return a scalar), `0` (norm of each row, return a column vector), `1` (return a row vector).
3. `sum(X, Dim=-1)`.

#### Matrix Operators

1. `inv(X)`: matrix inverse $X^{-1}$.
2. `transpose(X)`: matrix transpose $X^T$.
3. `trace(X)`: trace of a matrix. `trace(X, Y)`: trace of $XY$.
4. `det(X), logdet(X)`: determinant and natural log determinant.
5. `lsdot(X)`: left-self-dot $XX^T$. `rsdot(X)`: right-self-dot $X^TX$. They can be used to construct a semi-positive definitive matrix.
6. `diag(X)`: extract diagonal elements to form a column vector. `diag(X, Y)`: $\text{diag}(XY)$. `diagonal(v)`: construct a diagonal matrix by a vector (both column and row).
7. `vec(X)`: stack columns to form a column vector. `vech(X)`: stack columns of lower part to form a column vector. $X$ must be a square matrix. `ivech(v)`: unstack a vector to form a symmetric matrix. For example, if *v* has size 6, then the result will be a $2\times 2$ symmetric matrix ($\frac{(1+3)\times 3}{2}=6$).
This is usefull when you want to optimize a symmetric matrix.
$$
[1,2,3,4,5,6]\implies \begin{bmatrix}
1 & 2 &3  \\
2 & 4 & 5\\
3 & 5 & 6
\end{bmatrix}
$$

Note $1,2,3$ forms the first column (or the first row).
`ivecl(v)`: like *ivech(v)* but leave upper part to be zero (thus return a lower triangular matrix). `ivecu(v)`: leave lower part to be zero.

$$
\text{ivecl(v)}: [1,2,3,4,5,6]\implies \begin{bmatrix}
1 & 0 & 0  \\
2 & 4 & 0\\
3 & 5 & 6
\end{bmatrix}
\text{ivecu(v)}: [1,2,3,4,5,6]\implies \begin{bmatrix}
1 & 2 & 3 \\
0 & 4 & 5 \\
0 & 0 & 6
\end{bmatrix}
$$

#### Distributions

Only computes **log density**. For example,

```cpp
ln_mvnormal_den(X, mu, Sigma); //default None reduction.
ln_mvnormal_den(X, cvec(5), Sigma); // mu is a zero vector.
ln_mvnormal_den(X, mu, Sigma, DynAutoDiff::Mean); //mean reduction.
```

computes log multi-variate normal density. Arguments `X` is a data matrix( for multivariate distributions) or a column vector( for univariate distributions), each row represents a sample.

The function parameter `reduction=DynAutoDiff::None | Mean | Sum` is used to control reduction, default is `Sum` (then return a scalar). If `None`, return a column vector. The template parameter `true` is used to control whether to include constants. default is `false` which is not to include constants.


1. `ln_mvnormal_den(X,mu,Sigma)`: multivariate normal, which has been introduced above.
2. `ln_normal_den(X,mu, sigma)`: univariate normal. `X` must be a column vector.
3. `ln_mvt_den(X, mu, Sigma, nu)`: multivariate $t$ distribution. *nu* is a scalar.
4. `ln_t_den(X, mu, sigma, nu)`: univariate $t$ distribution. `mu, sigma, nu` are scalars.

#### Loss Functions

Remember the input order is `(input, target)`. The loss function has a template parameter `Reduction`, whose allowed values are `Reduction::None, Reduction::Sum, Reduction::Mean`. The default is `Sum`. Usage:

```
auto loss = binary_cross_entropy(sigmoid(X * theta + c), y); //Logistic regression.
```

1. `binary_cross_entropy(input, taret)`.
2. `mse_loss(input,target)`.


### Evaluation and Backward

To carry out a fresh computation, use

```cpp
GraphManager<> gm(y); //y is a variable.
gm.run(clear_leaf_grad=true);
```

It's equivalent to:

```cpp
zero_all(clear_leaf_grad); // Clear all gradient.
_root->eval();
_root->backward();
```

If `clear_leaf_grad=true`, then it will also clear gradient of leaf nodes  (parameters).

Sometimes you want to loop over samples and accumualate gradients on leaf nodes. Then just use

```
gm.run(false);
```

### GraphManager

#### Reallocate Contiguous Memory

Sometimes you want the memory of parameter nodes to be contiguous. Then you can use:

```cpp
auto [val, grad]& = gm.auto_bind_parm();
```

Here `val, grad` both are column vectors with type `TMap<T>`. 