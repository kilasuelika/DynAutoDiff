#include "../DynAutoDiff/DynAutoDiff.hpp"
#include <algorithm>
#include <boost/test/tools/old/interface.hpp>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <vector>

#define BOOST_TEST_MODULE MatrixOperator_Test
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#define TL 1e-10

using namespace Eigen;
using namespace std;
using namespace DynAutoDiff;

BOOST_AUTO_TEST_SUITE(test)

BOOST_AUTO_TEST_CASE(ivech_test) {

    // auto x=rowvec({1,2,3,4,5,6}, true);
    auto v1 = rowvec<double>({1, 2, 3, 4}), v2 = vec<double>({2, 3, 4, 5});
    auto S = rowvec<double>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, true);

    auto y1 = ivech(S);
    GraphManager gm1(y1);
    gm1.run();
    BOOST_CHECK_EQUAL(y1->val().coeff(0, 0), 1);
    BOOST_CHECK_EQUAL(y1->val().coeff(0, 3), 4);
    BOOST_CHECK_EQUAL(y1->val().coeff(2, 1), 6);
    BOOST_CHECK_EQUAL(y1->val().coeff(3, 3), 10);

    auto y2 = v1 * ivech(S) * v2;
    GraphManager gm2(y2);
    gm2.run();
    BOOST_CHECK_EQUAL(y2->v(), 959);
    BOOST_CHECK_EQUAL(S->g(0, 3), 13);
    BOOST_CHECK_EQUAL(S->g(0, 5), 17);
}
BOOST_AUTO_TEST_CASE(ivecl_test) {

    auto v1 = rowvec<double>({1, 2, 3, 4}), v2 = vec<double>({2, 3, 4, 5});
    auto S = rowvec<double>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, true);

    auto y1 = ivecl(S);

    GraphManager gm1(y1);
    gm1.run();
    // cout<<y1->val()<<endl;
    BOOST_CHECK_EQUAL(y1->val().coeff(0, 0), 1);
    BOOST_CHECK_EQUAL(y1->val().coeff(3, 0), 4);
    BOOST_CHECK_EQUAL(y1->val().coeff(2, 1), 6);
    BOOST_CHECK_EQUAL(y1->val().coeff(3, 3), 10);

    auto y2 = v1 * ivecl(S) * v2;
    GraphManager gm2(y2);
    gm2.run();
    BOOST_CHECK_EQUAL(y2->v(), 668);
    BOOST_CHECK_EQUAL(S->g(0, 0), 2);
    BOOST_CHECK_EQUAL(S->g(0, 1), 4);
    BOOST_CHECK_EQUAL(S->g(0, 3), 8);
    BOOST_CHECK_EQUAL(S->g(0, 5), 9);
}
BOOST_AUTO_TEST_CASE(lpnorm_test) {
    auto X = pmat<double>({1, 2, -2, 3, 1, 3}, 3, 2);

    auto z = lpnorm<2, 0>(X);
    GraphManager gm(z);
    gm.run();
    BOOST_CHECK_EQUAL(z->v(), std::sqrt(5));
    BOOST_CHECK_EQUAL(z->v(2), std::sqrt(10));
    BOOST_CHECK_CLOSE(X->g(0, 0), 1.0 / std::sqrt(5), 1e-10);
    BOOST_CHECK_CLOSE(X->g(0, 1), 2.0 / std::sqrt(5), 1e-10);
    BOOST_CHECK_CLOSE(X->g(1, 0), -2.0 / std::sqrt(13), 1e-10);

    auto z1 = lpnorm<2, -1>(X);
    GraphManager gm1(z1);
    gm1.run();
    BOOST_CHECK_EQUAL(z1->v(), std::sqrt(28));
    BOOST_CHECK_CLOSE(X->g(), 1.0 / std::sqrt(28), 1e-10);
    BOOST_CHECK_CLOSE(X->g(2, 0), 1.0 / std::sqrt(28), 1e-10);
    BOOST_CHECK_CLOSE(X->g(1, 0), -2.0 / std::sqrt(28), 1e-10);
    BOOST_CHECK_CLOSE(X->g(2, 1), 3.0 / std::sqrt(28), 1e-10);

    auto z2 = lpnorm<3, 0>(X);
    GraphManager gm2(z2);
    gm2.run();
    BOOST_CHECK_CLOSE(z2->v(), std::pow(9, 1.0 / 3), 1e-10);
    BOOST_CHECK_CLOSE(z2->v(1), std::pow(35, 1.0 / 3), 1e-10);
    BOOST_CHECK_CLOSE(X->g(0, 0), 1.0 / 9 * std::pow(9, 1.0 / 3), 1e-10);
    BOOST_CHECK_CLOSE(X->g(0, 1), 4.0 / 9 * std::pow(9, 1.0 / 3), 1e-10);
    BOOST_CHECK_CLOSE(X->g(1, 0), -4.0 / 35 * std::pow(35, 1.0 / 3), 1e-10);
}

BOOST_AUTO_TEST_CASE(trace_test) {
    auto X = prowvec<double>({1, 2, 3, 4}), Y = pvec<double>({2, 1, 4, 3});
    auto z = trace(X, Y);
    GraphManager gm(z);
    gm.run();
    BOOST_CHECK_EQUAL(z->v(), 28);
    for (int i = 0; i < 4; ++i) {
        BOOST_CHECK_EQUAL(X->g(i), Y->v(i));
        BOOST_CHECK_EQUAL(Y->g(i), X->v(i));
    }
}

BOOST_AUTO_TEST_CASE(diag_test) {
    auto X = pmat<double>({1, 2, 3, 4, 5, 6}, 2, 3), Y = pmat<double>({9, 8, 7, 6, 5, 4}, 3, 2);
    auto v = prowvec<double>({2, -6});
    auto z = v * diag(X, Y);
    GraphManager gm(z);
    gm.run();

    //
    double zv = -440;
    TMat<> _X(2, 3), _Y(3, 2), _v(1, 2);
    _X << 18, 14, 10, -48, -36, -24;
    _Y << 2, -24, 4, -30, 6, -36;
    _v << 38, 86;

    BOOST_CHECK_EQUAL(z->v(), zv);
    BOOST_CHECK(X->grad() == _X);
    BOOST_CHECK(Y->grad() == _Y);
    BOOST_CHECK(v->grad() == _v);
}

BOOST_AUTO_TEST_CASE(sum_test) {
    auto X = pmat<double>({1, 2, 3, 4, 5, 6}, 2, 3), Y = pmat<double>({9, 8, 7, 6, 5, 4}, 3, 2);
    auto v = prowvec<double>({2, -6});
    auto z = sum<1>(X * Y) * transpose(v);
    GraphManager gm(z);
    gm.run();

    //
    double zv = -430;
    TMat<> _X(2, 3), _Y(3, 2), _v(1, 2);
    _X << -30, -22, -14, -30, -22, -14;
    _Y << 10, -30, 14, -42, 18, -54;
    _v << 139, 118;

    BOOST_CHECK_EQUAL(z->v(), zv);
    BOOST_CHECK(X->grad() == _X);
    BOOST_CHECK(Y->grad() == _Y);
    BOOST_CHECK(v->grad() == _v);
}

BOOST_AUTO_TEST_CASE(linear_test) {
    auto A = pmat({1.0, 2.0, 3.0, 4.0}, 2, 2), x = pvec({2.0, 3.0}), b = pvec({5.0, -2.0});
    auto y = sum(linear(A, x, b));

    GraphManager<> gm(y);
    gm.run();
    BOOST_CHECK_EQUAL(y->v(), 29.0);
    BOOST_CHECK_EQUAL(A->g(0, 0), 2);
    BOOST_CHECK_EQUAL(A->g(0, 1), 3);
    BOOST_CHECK_EQUAL(b->g(0, 0), 1);

    auto s = psca(9.0);
    auto y1 = sum(linear(A, x, s));

    GraphManager<> gm1(y1);
    gm1.run();
    BOOST_CHECK_EQUAL(y1->v(), 44.0);
    BOOST_CHECK_EQUAL(A->g(0, 0), 2);
    BOOST_CHECK_EQUAL(A->g(0, 1), 3);
    BOOST_CHECK_EQUAL(s->g(), 2);
}

BOOST_AUTO_TEST_CASE(mean_test) {
    auto A = pmat({1.0, 2.0, 3.0, 4.0, -1.0, 6.0}, 3, 2), x = prowvec({2.0, 3.0, -6.0});
    auto y = x * mean<Dim::Row>(A);

    GraphManager<> gm(y);
    gm.run();
    BOOST_CHECK_EQUAL(y->v(), -3.0 / 2);
    BOOST_CHECK_EQUAL(A->g(0, 0), 1);
    BOOST_CHECK_EQUAL(A->g(0, 1), 1);
    BOOST_CHECK_EQUAL(x->g(0, 0), 3.0 / 2);
}

BOOST_AUTO_TEST_CASE(variance_test) {
    auto A = pmat({1.0, 2.0, 3.0, 4.0, -1.0, 6.0}, 3, 2), x = prowvec({2.0, 3.0, -6.0}),
         s = psca(2.0);

    auto y = x * variance<Dim::Row>(A);
    GraphManager<> gm(y);
    gm.run();
    BOOST_CHECK_EQUAL(y->v(), -289.0 / 4);
    BOOST_CHECK_EQUAL(A->g(0, 0), -1);
    BOOST_CHECK_EQUAL(A->g(0, 1), 1);
    BOOST_CHECK_EQUAL(x->g(0, 0), 1.0 / 4);

    auto y1 = s * variance(A);
    GraphManager<> gm1(y1);
    gm1.run();
    BOOST_CHECK_CLOSE(y1->v(), 59.0 / 6, 1e-10);
    BOOST_CHECK_CLOSE(A->g(0, 0), -1, 1e-10);
    BOOST_CHECK_CLOSE(A->g(0, 1), -1.0 / 3, TL);
    BOOST_CHECK_CLOSE(s->g(0, 0), 59.0 / 12, TL);
}
BOOST_AUTO_TEST_SUITE_END()
