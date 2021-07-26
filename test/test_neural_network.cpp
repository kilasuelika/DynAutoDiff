#include "../DynAutoDiff/DynAutoDiff.hpp"
#include <algorithm>
#include <boost/test/tools/old/interface.hpp>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <vector>

#define BOOST_TEST_MODULE NeuralNetwork_Test
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#define TL 1e-10

using namespace Eigen;
using namespace std;
using namespace DynAutoDiff;

BOOST_AUTO_TEST_SUITE(test)

BOOST_AUTO_TEST_CASE(neuralnet_test) {
    struct NN {
        TMat<> X, y;
        shared_ptr<Var<>> A1, b1, A2, b2, A3, b3, xi, yi, y_pred, residual;
        GraphManager<> gm;

        NN(const TMat<> &X, const Eigen::VectorXd &y)
            : X(X), y(y), A1(pmat(3, 2)), b1(pmat(3, 1)), A2(pmat(3, 3)), b2(pmat(3, 1)),
              A3(pmat(1, 3)), b3(pmat(1, 1)), xi(cmat(2, 1)), yi(cmat(1, 1)) {
            auto x1 = A1 * xi + b1;
            auto y1 = sigmoid(x1);
            auto y2 = sigmoid(A2 * y1 + b2) + x1;
            y_pred = A3 * y2 + b3;
            residual = pow(yi - y_pred, 2);
            gm.set_root(residual);
        };

        double loss(const double *parm, double *grad) {

            bind_seq(const_cast<double *>(parm), grad, {A1, b1, A2, b2, A3, b3});

            double loss = 0;
            for (int i = 0; i < X.rows(); ++i) {
                *xi = X.row(i).transpose();
                *yi = y.row(i);

                gm.run(false);
                loss += residual->v();
            }
            return loss;
        };
    };

    // Create data matrix.
    TMat X(5, 2);
    X << 1, 10, 2, 20, 3, 30, 4, 40, 5, 50;
    TVecd y(5);
    y << 32, 64, 96, 128, 160; // y=2*x1+3*x2

    // Generating parameter buffer and NN.
    TVecd parm_data(25);
    parm_data << 0.043984, -0.800526, 0.960126, -0.0287914, -0.520941, 0.635809, 0.584603,
        -0.443382, 0.224304, 0.97505, 0.666392, -0.911053, -0.824084, -0.498828, -0.230156, 0.2363,
        -0.781428, -0.136367, 0.263425, 0.841535, 0.920342, 0.65629, 0.848248, -0.748697, 0.21522;

    TVecd grad_data(25);

    NN net(X, y);
    auto loss = net.loss(parm_data.data(), grad_data.data());

    BOOST_CHECK_CLOSE(loss, 92030.0564, 1e-2);
    BOOST_CHECK_CLOSE(net.A1->g(0, 0), -2953.05, 1e-2);
    BOOST_CHECK_CLOSE(net.b3->g(0, 0), -1226.47, 1e-2);
}
BOOST_AUTO_TEST_CASE(minus_test) {}
BOOST_AUTO_TEST_CASE(times_test) {}

BOOST_AUTO_TEST_CASE(division_test) {}
BOOST_AUTO_TEST_SUITE_END()
