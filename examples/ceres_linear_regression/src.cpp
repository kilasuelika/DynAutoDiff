#include "../../DynAutoDiff/CeresOptimizer.hpp"

using namespace std;
using namespace DynAutoDiff;

int main() {
    auto X = std::make_shared<Var<>>("../../test/X.txt");
    auto y = std::make_shared<Var<>>("../../test/y.txt");

	//Set parameters.
    auto theta_ols=pvec(X->cols());
	auto theta_mle=pvec(X->cols());
    auto c_ols = psca();
	auto c_mle=psca();

	// Losses.
    auto mse=mse_loss(X*theta_ols+c_ols, y);
	auto ll=-ln_normal_den(y-X*theta_mle-c_mle, csca(0.0), psca(1.0));

    CeresOptimizer ols_opt(mse);
    ols_opt.run();
	CeresOptimizer mle_opt(ll);
	mle_opt.run();

    cout << "OLS theta: "<<endl << theta_ols->val() << endl;
    cout << "OLS c: "<<endl << c_ols->val() << endl;

	cout << "MLE theta: "<<endl << theta_mle->val() << endl;
    cout << "MLE c: "<<endl << c_mle->val() << endl;
}