#include "DynAutoDiff/DynAutoDiff.hpp"

using namespace std;
using namespace DynAutoDiff;

int main() {
    // Create variables and expressions.
    auto x = vec<double>({1, 2}), Sigma = pmat<double>({2, 1, 1, 2}, 2, 2),
         y = pvec<double>({2, 3});
    auto z = transpose(x) * inv(Sigma) * y;
    GraphManager gm(z);

    // Run automatic differential.
    gm.run();
    cout << z->v() << endl << Sigma->val() << endl << Sigma->grad() << endl;

    // Save and load.
    gm.save("graph.json");
    auto z1 = gm.load("graph.json");
}