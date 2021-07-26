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
BOOST_AUTO_TEST_CASE(exp_test) {
    auto x1 = csca(1.0), x = psca(2.0);
    auto y1 = exp(-0.5 * (x1 - x));
    GraphManager<> m1(y1);
    m1.run();
    BOOST_CHECK_CLOSE(y1->val().coeff(0, 0), 1.6487212707, 1e-5);
    BOOST_CHECK_CLOSE(x->grad().coeff(0, 0), 0.82436063535, 1e-5);

    auto v1 = vec({1.0, 2.0}), v2 = pvec({2.0, 3.0});
    auto z1 = sum(exp(-0.5 * (v1 - v2)));
    GraphManager<> m2(z1);
    m2.run();
    BOOST_CHECK_CLOSE(v2->g(), 0.82436063535, 1e-5);

    auto z2 = exp(-0.5 * transpose(v1) * v2);
    GraphManager<> m3(z2);
    m3.run();
    BOOST_CHECK_CLOSE(v2->g(), -0.0091578194, 1e-5);

    auto z3 = exp(-0.5 * transpose(v1 - v2) * (v1 - v2));
    GraphManager<> m4(z3);
    m4.run();
    BOOST_CHECK_CLOSE(v2->g(), -0.367879441171, 1e-5);

    auto z4 = sum(exp(-0.5 * diag(v1 - v2, transpose(v1 - v2))));
    GraphManager<> m5(z4);
    m5.run();
    BOOST_CHECK_CLOSE(v2->g(), -0.606530659713, 1e-5);
}

BOOST_AUTO_TEST_CASE(compound_test) {
    auto x1 = rowvec({1.0, 5.0}), s1 = psca(5.0), Sigma = mat({6.0, 7.0, 8.0, 9.0}, 2, 2, true),
         x2 = vec({2.0, 6.0});
    auto y1 = (s1 - x1) * x2 + s1;
    GraphManager<> m1(y1);
    m1.zero_all();
    y1->eval();
    y1->backward();
    BOOST_CHECK_EQUAL(y1->val().coeff(0, 0), 13);
    BOOST_CHECK_EQUAL(s1->grad().coeff(0, 0), 9);

    auto y2 = (s1 - x1) * Sigma * (x2 - s1) + s1;
    GraphManager<> m2(y2);
    m2.zero_all();
    y2->evalb();
    BOOST_CHECK_EQUAL(y2->val().coeff(0, 0), -39);
    BOOST_CHECK_EQUAL(s1->grad().coeff(0, 0), -77);
    BOOST_CHECK_EQUAL(Sigma->grad().coeff(0, 0), -12);
}
BOOST_AUTO_TEST_CASE(negation_test) {
    auto x1 = rowvec({1.0, 5.0}, true), s1 = psca(5.0),
         Sigma = mat({6.0, 7.0, 8.0, 9.0}, 2, 2, true), x2 = vec({2.0, 6.0});

    auto y1 = -x1 * x2;
    GraphManager<> m1(y1);
    m1.zero_all();
    y1->evalb();
    BOOST_CHECK_EQUAL(y1->val().coeff(0, 0), -32);
    BOOST_CHECK_CLOSE(x1->g(0, 0), -2, 1e-8);
}
BOOST_AUTO_TEST_CASE(times_test) {
    auto x1 = rowvec({1.0, 5.0}), s1 = psca(5.0), Sigma = mat({6.0, 7.0, 8.0, 9.0}, 2, 2, true),
         x2 = vec({2.0, 6.0});

    auto y1 = x1 * x2;
    GraphManager<> m1(y1);
    m1.zero_all();
    y1->eval();
    BOOST_CHECK_EQUAL(y1->val().coeff(0, 0), 32);
    // BOOST_CHECK_EQUAL(y1->grad().coeff(0, 0), 404);

    // auto y2 = 5 * x1 * s1 * Sigma * x2;
    auto y2 = x1 * Sigma * x2;
    GraphManager<> m2(y2);
    m2.zero_all();
    y2->eval();
    y2->backward();
    // BOOST_CHECK_EQUAL(y2->val().coeff(0, 0), 10100);
    // BOOST_CHECK_EQUAL(s1->grad().coeff(0, 0), 2020);
    BOOST_CHECK_EQUAL(Sigma->grad().coeff(0, 0), 2);
    BOOST_CHECK_EQUAL(Sigma->grad().coeff(1, 1), 30);

    auto y3 = x1 * s1 * Sigma * x2;
    GraphManager<> m3(y3);
    m3.zero_all();
    // cout << "y3 grad before: " << endl << Sigma->grad() << endl;
    y3->eval();
    y3->backward();
    BOOST_CHECK_EQUAL(y3->val().coeff(0, 0), 2020);
    BOOST_CHECK_EQUAL(s1->grad().coeff(0, 0), 404);
    BOOST_CHECK_EQUAL(Sigma->grad().coeff(0, 0), 10);
    BOOST_CHECK_EQUAL(Sigma->grad().coeff(1, 1), 150);

    auto y4 = s1 * x1 * Sigma * x2;
    GraphManager<> m4(y4);
    m4.zero_all();
    y4->eval();
    y4->backward();

    BOOST_CHECK_EQUAL(y4->val().coeff(0, 0), 2020);
    BOOST_CHECK_EQUAL(s1->grad().coeff(0, 0), 404);
    BOOST_CHECK_EQUAL(Sigma->grad().coeff(0, 0), 10);
    BOOST_CHECK_EQUAL(Sigma->grad().coeff(1, 1), 150);
}

BOOST_AUTO_TEST_CASE(division_test) {
    auto x1 = rowvec({1.0, 5.0}), s1 = psca(5.0), Sigma = mat({6.0, 7.0, 8.0, 9.0}, 2, 2, true),
         x2 = vec({2.0, 6.0}, true);
    auto y1 = s1 / (x1 * Sigma * x2);
    GraphManager<> m1(y1);
    m1.zero_all();
    y1->eval();
    y1->backward();
    BOOST_CHECK_CLOSE(y1->val().coeff(0, 0), 5.0 / 404, TL);
    BOOST_CHECK_CLOSE(s1->grad().coeff(0, 0), 1.0 / 404, TL);
    BOOST_CHECK_CLOSE(Sigma->grad().coeff(0, 0), -5.0 / 81608, TL);
    BOOST_CHECK_CLOSE(Sigma->grad().coeff(1, 1), -75.0 / 81608, TL);
}

BOOST_AUTO_TEST_CASE(power_test) {
    auto x1 = psca(5.0), x2 = psca(6.0), x3 = psca(-7.0);
    auto y1 = pow(x1, 2) + x3;
    GraphManager<> m1(y1);
    m1.zero_all();
    y1->eval();
    y1->backward();
    BOOST_CHECK_CLOSE(y1->v(), 18.0, TL);
    BOOST_CHECK_CLOSE(x1->g(), 10.0, TL);

    auto y2 = pow(x1, 2) * x3 + pow(x2, 3) * x1;
    GraphManager<> m2(y2);
    m2.zero_all();
    y2->eval();
    y2->backward();
    BOOST_CHECK_CLOSE(y2->v(), 905, TL);
    BOOST_CHECK_CLOSE(x1->g(), 146, TL);
    BOOST_CHECK_CLOSE(x2->g(), 540, TL);
}

BOOST_AUTO_TEST_CASE(sigmoid_test) {
    auto x1 = psca(5.0), x2 = psca(6.0), x3 = psca(-7.0);
    auto y1 = sigmoid(x1) + sigmoid(x2 - 5.5);
    GraphManager<> m1(y1);
    m1.zero_all();
    y1->eval();
    y1->backward();
    BOOST_CHECK_CLOSE(y1->v(), 1.61577, 1e-2);
    BOOST_CHECK_CLOSE(x1->g(), 0.00664806, 1e-2);

    auto y2 = sigmoid(x1 + x3) + x2 * sigmoid(x2 + x3);
    GraphManager<> m2(y2);
    m2.zero_all();
    y2->eval();
    y2->backward();
    BOOST_CHECK_CLOSE(y2->v(), 1.73285, 1e-2);
    BOOST_CHECK_CLOSE(x1->g(), 0.104994, 1e-2);
    BOOST_CHECK_CLOSE(x2->g(), 1.44861, 1e-2);

    auto y3 = pow(sigmoid(x1 + x2 / 2 + x3) + x1 - x2, 2);
    GraphManager<> m3(y3);
    m3.zero_all();
    y3->eval();
    y3->backward();
    BOOST_CHECK_CLOSE(y3->v(), 0.0723295, 1e-2);
    BOOST_CHECK_CLOSE(x1->g(), -0.643637, 1e-2);

    auto s1 = psca(0.5);
    auto p = sigmoid(s1);
    // auto y4 = ln((1 - p) * 0.9);
    auto y4 = ln(p * 0.2);
    GraphManager<> m4(y4);
    m4.run();
    BOOST_CHECK_CLOSE(s1->g(), 0.377540668798, 1e-2);

    auto y5 = ln((1 - p) * 0.9);
    GraphManager<> m5(y5);
    m5.run();
    BOOST_CHECK_CLOSE(s1->g(), -0.622459331202, 1e-2);

    auto y6 = ln(p * 0.2 + (1 - p) * 0.9);
    GraphManager<> m6(y6);
    m6.run();
    BOOST_CHECK_CLOSE(s1->g(), -0.354318819035, 1e-2);

    auto y7 = ln(s1 * 0.2 + (1 - s1) * 0.9);
    GraphManager<> m7(y7);
    m7.run();
    BOOST_CHECK_CLOSE(s1->g(), -1.27272727, 1e-2);

    auto s2 = 2 * s1;
    auto y8 = ln(s2 * 0.2 + (1 - s2) * 0.9);
    GraphManager<> m8(y8);
    m8.run();
    BOOST_CHECK_CLOSE(s1->g(), -7, 1e-2);
}
BOOST_AUTO_TEST_CASE(sigmoid_test_with_mat) {
    auto x1 = rowvec({.1, .5}), s1 = psca(.5), Sigma = mat({.6, .7, .8, .9}, 2, 2, true),
         x2 = vec({.2, .6}, true);
    auto y1 = x1 * sigmoid(Sigma * x2);
    GraphManager<> m1(y1);
    m1.zero_all();
    y1->eval();
    y1->backward();
    BOOST_CHECK_CLOSE(y1->v(), 0.397275, 1e-2);
    BOOST_CHECK_CLOSE(Sigma->g(0, 0), 0.00465251, 1e-2);
    BOOST_CHECK_CLOSE(x2->g(0, 0), 0.102643, 1e-2);

    auto y2 = x1 * (sigmoid(Sigma * x2) + x2);
    GraphManager<> m2(y2);
    m2.zero_all();
    y2->eval();
    y2->backward();
    BOOST_CHECK_CLOSE(y2->v(), 0.717275, 1e-2);
    BOOST_CHECK_CLOSE(Sigma->g(0, 0), 0.00465251, 1e-2);
    BOOST_CHECK_CLOSE(x2->g(0, 0), 0.202643, 1e-2);
}
BOOST_AUTO_TEST_CASE(log_division_test) {
    auto x1 = psca(2.0), x2 = psca(5.0);
    auto y1 = ln(x1 / (x1 + x2));
    GraphManager<> m1(y1);
    m1.zero_all();
    y1->eval();
    y1->backward();
    BOOST_CHECK_EQUAL(y1->val().coeff(0, 0), std::log(2.0 / 7.0));
    BOOST_CHECK_EQUAL(x1->g(), 0.5 - 1.0 / 7);
}
BOOST_AUTO_TEST_CASE(sqrt_test) {
    auto x1 = psca(2.0), x2 = psca(5.0);
    auto y1 = 1.0/sqrt(x1);
    GraphManager<> m1(y1);
    m1.run();
    BOOST_CHECK_EQUAL(y1->val().coeff(0, 0), 1.0/std::sqrt(2.0));
    BOOST_CHECK_CLOSE(x1->g(), -0.176776695297, 1e-5);

	auto y2 = 1.0/x1;
    GraphManager<> m2(y2);
    m2.run();
    BOOST_CHECK_EQUAL(y2->val().coeff(0, 0), 0.5);
    BOOST_CHECK_CLOSE(x1->g(), -0.25, 1e-5);
}

BOOST_AUTO_TEST_CASE(relu_test) {
    auto m=pmat<double>({-2,-1,1,2},2,2);
	auto y=relu(m);
    GraphManager<> m1(y);
    m1.run();
    BOOST_CHECK_EQUAL(y->v(), 0);
    BOOST_CHECK_CLOSE(y->v(1,0),1, 1e-5);
	BOOST_CHECK_CLOSE(m->g(),0, 1e-5);
	BOOST_CHECK_CLOSE(m->g(1,1),1, 1e-5);
}
BOOST_AUTO_TEST_SUITE_END()
