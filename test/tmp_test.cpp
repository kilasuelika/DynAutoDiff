#include "../DynAutoDiff/DynAutoDiff.hpp"
#include <algorithm>
#include <boost/test/tools/old/interface.hpp>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <vector>

#define BOOST_TEST_MODULE ArithMetic_Test
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#define TL 1e-10

using namespace Eigen;
using namespace std;
using namespace DynAutoDiff;

BOOST_AUTO_TEST_SUITE(test)
BOOST_AUTO_TEST_CASE(multi_branch_test) {
    auto x1=psca(2.0),x2=psca(3.0),x3=psca(4.0),x4=psca(5.0);
	auto y1=x1*x2+x3;
	auto y2=y1*x4+y1;

	GraphManager<> m1(y2);
	m1.run();
	BOOST_CHECK_CLOSE(y2->v(), 60.0, 1e-5);
	BOOST_CHECK_CLOSE(x1->g(), 18.0, 1e-5);
}
BOOST_AUTO_TEST_SUITE_END()
