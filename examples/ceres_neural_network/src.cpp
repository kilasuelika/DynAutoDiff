#include "../../DynAutoDiff/Var.hpp"
#include "ceres/ceres.h"
#include <algorithm>
#include <boost/test/tools/old/interface.hpp>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <vector>

using namespace Eigen;
using namespace std;
using namespace DynAutoDiff;

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
        //	residual=norm(yi-y_pred);
        residual = pow(yi - y_pred, 2);
        gm.set_root(residual);
    };
    void bind(double *parm, double *grad) {
        A1->bind(parm, grad, 3, 2);
        // std::cout << A1.get() << std::endl;
        b1->bind(parm + 6, grad + 6, 3, 1);
        A2->bind(parm + 9, grad + 9, 3, 3);
        b2->bind(parm + 18, grad + 18, 3, 1);
        A3->bind(parm + 21, grad + 21, 1, 3);
        b3->bind(parm + 24, grad + 24, 1, 1);
    };
    double loss(const double *parm, double *grad) {
        bind(const_cast<double *>(parm), grad);
        gm.zero_all();
        // Var<double, mat> y3(1, 1);
        // data buffer
        Eigen::VectorXd x_row_buffer = X.row(0);
        xi->bind(x_row_buffer.data(), nullptr, X.cols(), 1);
        Eigen::VectorXd y_row_buffer = y.row(0);
        yi->bind(y_row_buffer.data(), nullptr, y.cols(), 1);

        // Loop over each row to calulate loss.
        double loss = 0;

        for (int i = 0; i < X.rows(); ++i) {
            x_row_buffer = X.row(i);
            y_row_buffer = y.row(i);

            gm.zero_all(false);
            residual->eval();
            residual->backward();

            loss += residual->v();
        }
        return loss;
    };
};
// Ceres functor.
class NNfunctor : public ceres::FirstOrderFunction {
  private:
    NN &net;

  public:
    NNfunctor(NN &net) : net(net){};
    virtual bool Evaluate(const double *parameters, double *cost, double *gradient) const {
        *cost = net.loss(parameters, gradient);
        return true;
    }
    virtual int NumParameters() const { return 25; }
};
int main() {
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
	
        // Train
    google::InitGoogleLogging("FastAD with ceres");
    ceres::GradientProblemSolver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 1000;
    ceres::GradientProblemSolver::Summary summary;
    ceres::GradientProblem problem(new NNfunctor(net));
    ceres::Solve(options, problem, parm_data.data(), &summary);
    std::cout << summary.FullReport() << "\n";
    std::cout << "Initial x[0]: " << 0.043984 << " x[1]: " << 0.960126 << "\n";
    std::cout << "Final   x[0]: " << parm_data.coeff(0) << " x[1]: " << parm_data.coeff(1) << "\n";

}