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

BOOST_AUTO_TEST_SUITE(Losses_Test)

BOOST_AUTO_TEST_CASE(bce_test) {
    vector<TMat<>> v(2);
    v[0] = TMat<>(2, 2);
    // cout << v[0] << endl;

    auto p = vec<double>({0.2, 0.3, 0.4, 0.5, 0.8}, true), y = vec<double>({0, 0, 1, 1, 1});

    auto y1 = binary_cross_entropy(p, y);
    GraphManager gm1(y1);
    gm1.run();

    BOOST_CHECK_CLOSE(y1->v(), 2.4123999590012524, 1e-5);
    BOOST_CHECK_CLOSE(p->g(), 5.0 / 4, 1e-5);
    BOOST_CHECK_CLOSE(p->g(1), 1.4285714382902825, 1e-3);
    BOOST_CHECK_CLOSE(p->g(3), -2, 1e-10);
}
BOOST_AUTO_TEST_CASE(ivecl_test) {}
BOOST_AUTO_TEST_CASE(times_test) {}

BOOST_AUTO_TEST_CASE(division_test) {}
BOOST_AUTO_TEST_SUITE_END()
