#include "../../DynAutoDiff/CeresOptimizer.hpp"
#include <algorithm>
#include <chrono>

using namespace Eigen;
using namespace std;
using namespace DynAutoDiff;

template <typename T = double> struct NN {
    int k;
    shared_ptr<Var<T>> A1 = pmat(20, k), b1 = pvec(20), A2 = pmat(50, 20), b2 = pvec(50),
                       A3 = pmat(20, 50), b3 = pvec(20), A4 = pmat(1, 20), b4 = pvec(1);

    NN(int k) : k(k) {
        // He parameter initialization.
        for (auto &node : {A1, A2, A3, A4}) {
            node->setRandom();
            *node = node->val() / sqrt(node->cols() / 2);
        };
    };

    auto make_model(const TMat<T> &X) {
        auto y1 = relu(offset(A1 * mat(X), b1)); // 20*N
        auto y2 = relu(offset(A2 * y1, b2));     // 100*N
        auto y3 = relu(offset(A3 * y2, b3));     // 20*N
        auto y4 = relu(offset(A4 * y3, b4));     // 1*N

        return y4;
    };
};

template <typename T = double> void evaluate(const TMat<T> &test_X, const TMat<T> &test_y) {}
int main() {
    // Create train and test dataset.
    auto data = load_mat("../../test/cancer_reg.txt");
    int N = data.rows(), k = data.cols();
    data.transposeInPlace();
    for (int i = 0; i < k; ++i) {
        data.row(i).normalize();
    }

    // Generate random sample ids for training and testing.
    vector<int> all(N);
    iota(all.begin(), all.end(), 0);
    default_random_engine eng(0);
    shuffle(all.begin(), all.end(), eng);

    vector<int> train_id(all.begin(), all.begin() + 2200);
    vector<int> test_id(all.begin() + 2200, all.end());
    TMat<> train_X = data(seqN(1, k - 1), train_id);
    TMat<> train_y = data(seqN(0, 1), train_id);
    TMat<> test_X = data(seqN(1, k - 1), test_id);
    TMat<> test_y = data(seqN(0, 1), test_id);

    // Expression for training.
    NN<> net(train_X.rows());
    auto y4 = net.make_model(train_X);
    auto mse = mse_loss<Reduction::Mean>(y4, mat(train_y));
    GraphManager<> gm(mse);
    gm.auto_bind_parm();

    // Expression for testing.
    auto y4t = net.make_model(test_X);
    auto test_mse = mse_loss<Reduction::Mean>(y4t, mat(test_y));
    GraphManager<> gm1(test_mse);

    ConstLR c_scheduler(1e-6);
    Adam adam_opt(&gm, &c_scheduler);

    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < 5000; ++i) {
        adam_opt.step();
        if (i % 100 == 0) {
            gm1.run();
            cout << "Iteration " << i << ", learning rate: " << c_scheduler.lr()
                 << ", train RMSE: " << sqrt(mse->v()) << ", test RMSE: " << sqrt(test_mse->v())
                 << endl;
        }
    }
    auto end = chrono::high_resolution_clock::now();
    cout << "Total time: " << chrono::duration_cast<chrono::seconds>(end - start).count() << endl;
    gm.save("graph.json");
}