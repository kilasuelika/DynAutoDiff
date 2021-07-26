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

BOOST_AUTO_TEST_SUITE(Eigen_Helper_Test)

BOOST_AUTO_TEST_CASE(load_mat_test) {
    auto X=load("X.txt");
    
    BOOST_CHECK_EQUAL(X->rows(), 100);
    BOOST_CHECK_EQUAL(X->cols(), 5);
    BOOST_CHECK_CLOSE(X->v(0, 0), 0.814723686393179, TL);
    BOOST_CHECK_CLOSE(X->v(5, 4), 0.699887849928292, TL);
    BOOST_CHECK_CLOSE(X->v(99, 4), 0.613460736812875, TL);

    auto y=load("y.txt");
   
    BOOST_CHECK_EQUAL(y->rows(), 100);
    BOOST_CHECK_EQUAL(y->cols(), 1);
    BOOST_CHECK_CLOSE(y->v(99, 0), 7.62241903286546, TL);
}
BOOST_AUTO_TEST_CASE(ivecl_test) {}
BOOST_AUTO_TEST_CASE(times_test) {}

BOOST_AUTO_TEST_CASE(division_test) {}
BOOST_AUTO_TEST_SUITE_END()
