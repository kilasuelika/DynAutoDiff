#include "../DynAutoDiff/Distributions.hpp"
#include "../DynAutoDiff/Var.hpp"
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

BOOST_AUTO_TEST_SUITE(Initialize_Variable_Test)

BOOST_AUTO_TEST_CASE(init_var_test) {
    auto X = load("X.txt");

    BOOST_CHECK_EQUAL(X->rows(), 1000);
    BOOST_CHECK_EQUAL(X->cols(), 5);
    BOOST_CHECK_CLOSE(X->v(0, 0), -0.343003152130103, TL);
    BOOST_CHECK_CLOSE(X->v(5, 4), -0.0470153684815493, TL);
    BOOST_CHECK_CLOSE(X->v(99, 4), 0.303972116239077, TL);

    auto y = load("y.txt");

    BOOST_CHECK_EQUAL(y->rows(), 1000);
    BOOST_CHECK_EQUAL(y->cols(), 1);
    BOOST_CHECK_CLOSE(y->v(99, 0), -1.70565299307618, TL);
}
BOOST_AUTO_TEST_CASE(ivecl_test) {}
BOOST_AUTO_TEST_CASE(times_test) {}

BOOST_AUTO_TEST_CASE(division_test) {}
BOOST_AUTO_TEST_SUITE_END()
