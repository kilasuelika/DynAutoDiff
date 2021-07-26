#include "../DynAutoDiff/DynAutoDiff.hpp"

#define BOOST_TEST_MODULE IO_Test
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#define TL 1e-10

using namespace Eigen;
using namespace std;
using namespace DynAutoDiff;

BOOST_AUTO_TEST_SUITE(Eigen_Helper_Test)

BOOST_AUTO_TEST_CASE(load_mat_test) {
    auto x1 = rowvec({1.0, 5.0}), s1 = psca(5.0), Sigma = mat({6.0, 7.0, 8.0, 9.0}, 2, 2, true),
         x2 = vec({2.0, 6.0});
    auto y1 = (s1 - x1) * x2 + s1;
    GraphManager gm(y1);
    gm.run();
    gm.save("graph.json");

    auto y2 = gm.load("graph.json");
    gm.run();
    BOOST_CHECK_EQUAL(y1->v(), y2->v());
}
BOOST_AUTO_TEST_SUITE_END()
