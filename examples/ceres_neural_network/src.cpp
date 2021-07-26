#include "../../DynAutoDiff/CeresOptimizer.hpp"
#include "ceres/ceres.h"
#include <algorithm>
#include <boost/test/tools/old/interface.hpp>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <random>
#include <vector>
using namespace Eigen;
using namespace std;
using namespace DynAutoDiff;

int main() {
    // Create train and test dataset.
    auto data = load_mat("../../test/cancer_reg.txt");
    int N = data.rows(), k = data.cols();
    data.transposeInPlace();
    for (int i = 1; i < k; ++i) {
        data.row(i).normalize();
    }
    vector<int> all(N);
    iota(all.begin(), all.end(), 0);
    default_random_engine eng;
    shuffle(all.begin(), all.end(), eng);

    vector<int> train_id(all.begin(), all.begin() + 2200);
    vector<int> test_id(all.begin() + 2200, all.end());
    TMat<> train_X = data(seqN(1, k - 1), train_id);
    TMat<> train_y = data(seqN(0, 1), train_id);
    TMat<> test_X = data(seqN(1, k - 1), test_id);
    TMat<> test_y = data(seqN(0, 1), test_id);

    // Model.
    auto A1 = pmat(50, k - 1), b1 = pvec(50), A2 = pmat(100, 50), b2 = pvec(100),
         A3 = pmat(100, 100), b3 = pvec(100), A4 = pmat(50, 100), b4 = pvec(50), A5 = pmat(50, 50),
         b5 = pvec(50), A6 = pmat(25, 50), b6 = pvec(25), A7 = pmat(1, 25), b7 = pvec(1);

    for (auto &node : {A1, b1, A2, b2, A3, b3, A4, b4, A5, b5, A6, b6, A7, b7}) {
        node->setRandom();
        // cout<<node->val()<<endl;
    };

    auto y1 = relu(offset(A1 * mat(train_X), b1)); // 50*N
    auto y2 = relu(offset(A2 * y1, b2));           // 100*N
    auto y3 = relu(offset(A3 * y2, b3));           // 100*N
    // cout << 5 << endl;
    auto y4 = relu(offset(A4 * y3, b4) + y1); // 50*N
    // cout << 5 << endl;
    auto y5 = relu(offset(A5 * y4, b5)); // 50*N
    // cout << 5 << endl;
    // cout<<"y5: "<<y5->rows()<<" "<<y5->cols()<<endl;
    auto y6 = relu(offset(A6 * y5, b6)); // 25*N
    // cout << 5 << endl;
    auto y7 = offset(A7 * y6, b7); // 1*N

    // cout << y7->rows() << " " << y7->cols() << endl;

    auto mse = mse_loss<Reduction::Mean>(y7, mat(train_y));

    CeresOptimizer opt(mse);
    opt.run();

    opt.gm.save("model.json");
}