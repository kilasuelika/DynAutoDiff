#include "../DynAutoDiff/CeresOptimizer.hpp"
#include "../DynAutoDiff/DynAutoDiff.hpp"
#include <algorithm>
#include <boost/test/tools/old/interface.hpp>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <ostream>
#include <vector>

#define BOOST_TEST_MODULE Normal_Test
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#define TL 1e-10

using namespace Eigen;
using namespace std;
using namespace DynAutoDiff;

BOOST_AUTO_TEST_SUITE(Eigen_Helper_Test)

BOOST_AUTO_TEST_CASE(load_mat_test) {
    auto X = std::make_shared<Var<>>("X.txt");
    auto y = std::make_shared<Var<>>("y.txt");
    auto theta = pvec(X->cols()), c = psca();
    auto loss = mse_loss(X * theta + c, y);

    CeresOptimizer opt(loss);
    opt.run();
    std::cout << theta->val() << endl << c->v() << endl;
}

BOOST_AUTO_TEST_CASE(logistic_regression) {
    auto X = std::make_shared<Var<>>("Xb.txt");
    auto y = std::make_shared<Var<>>("yb.txt");
    auto theta = pvec(X->cols()), c = psca();
    auto loss = binary_cross_entropy(sigmoid(X * theta + c), y);

    CeresOptimizer opt(loss);
    opt.run();
    std::cout << theta->val() << endl << c->v() << endl;
}
BOOST_AUTO_TEST_SUITE_END()
