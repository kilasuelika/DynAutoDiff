#include "DynAutoDiff/DynAutoDiff.hpp"

using namespace std;
using namespace DynAutoDiff;

int main()
{
    // Create variables and expressions.
    auto x = vec<double>({1, 2}), Sigma = pmat<double>({2, 1, 1, 2}, 2, 2), y = pvec<double>({2, 3});
    auto z = transpose(x) * inv(Sigma) * y;
    GraphManager gm(z);

    // Run automatic differential.
    gm.run();
    cout << z.v() << endl << Sigma.val() << endl << Sigma.grad() << endl;

    // Save and load.
    gm.save("graph.json");
    auto z1 = gm.load("graph.json");

    // constant
    auto z2 = 3 * x;
    GraphManager gm1(z2);
    gm1.run();

    // subscript
    auto A = pmat<double>({1, 2, 3, 4, 5, 6}, 2, 3);
    auto B = A(0, 1) + A(1, 2);
    GraphManager gm2(B);
    gm2.run();
    cout << B.v() << endl;
    cout << A.grad() << endl;

    return 0;
}
