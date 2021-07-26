#include "../../DynAutoDiff/NloptOptimizer.hpp"
#include "../../DynAutoDiff/Var.hpp"

#include <algorithm>
#include <boost/test/tools/old/interface.hpp>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <vector>

using namespace Eigen;
using namespace std;
using namespace DynAutoDiff;

int main() {
    auto X = std::make_shared<Var<>>("../../test/gmm_datat5000_10.txt");
    assert(X->cols() == 10);
    auto I = cmat(10, 10);
    *I = TMat<>::Identity(10, 10);
    auto x = vec<double>(10);
    auto mu1 = prowvec<double>({10, 10, 10, 10, 10, 10, 10, 10, 10, 10});
    auto sigma1 = pvec<double>(
        {6.74714605673973,   -4.78846285151625,   9.81986317681912,   -0.946778596884535,
         -2.56090828960826,  14.1770454462236,    -2.34872949704331,  5.41836610159671,
         -0.254326999386627, 11.1713144090129,    5.58678489223679,   -9.96410391329875,
         1.93144035168834,   -7.08526287510196,   14.4856678906208,   -0.877117327445445,
         1.47118749856504,   -1.76534777518132,   0.391603651196246,  -0.502081717050957,
         8.55537582476351,   0.434493402355983,   -0.779816583548181, 2.70227465614796,
         0.467426217141603,  1.0717211560211,     -3.18255417776125,  8.5847705536712,
         -1.38950794230753,  -4.27292225169075,   4.94585016403557,   -5.6158163844145,
         7.12046814995529,   -1.38750604593847,   1.50618376866173,   17.2940764425205,
         0.982568550511888,  3.77851328637645,    1.9611207228883,    1.61784487025558,
         -1.7359302585308,   3.26854322440445,    -1.43794578783318,  -4.46217311982041,
         8.89969442980783,   -0.735245799904036,  -0.197278972322572, -2.40540421662934,
         4.70769173394015,   0.00565941495598965, 1.89675367633351,   -1.33799950821767,
         -1.55871056229486,  -1.30374847986576,   4.90519810018584});
    auto mu2 = prowvec<double>({-10, -10, -10, -10, -10, -10, -10, -10, -10, -10});
    auto sigma2 = pvec<double>(
        {4.42148511309383,   3.61827721740259,   26.4204244054304,   -0.0936185215441301,
         5.64396657007986,   8.2835183852921,    -2.08351679398624,  0.590447490269971,
         -3.41078309376636,  6.48119671977636,   -1.61712584196582,  -2.85948127496202,
         -0.902993784872609, 4.45189016484981,   12.6844246779533,   3.16290774167664,
         -2.16737911729119,  3.40644249355353,   -3.50390343314936,  3.53079273683045,
         15.8357509201652,   -0.633351135497599, 8.94088791986241,   4.40985391966033,
         -1.10305925289209,  -1.78695177608141,  -0.975893700774255, 7.47266661628199,
         0.286289972290326,  6.95107119272155,   0.47450832618569,   1.21252576117289,
         -2.15940317200453,  0.387506927188112,  2.10700648377827,   7.2589520774548,
         0.619777849529969,  -2.6084703316509,   -0.124053090455333, -0.772020512153856,
         -1.5472321765894,   2.36949425094364,   -2.28648226136606,  0.822485464677878,
         6.80340398108253,   -1.74077429597856,  -1.2619333236744,   4.42760485330411,
         0.92965155861724,   2.44123132119213,   3.12849197146664,   1.15516181016746,
         -0.656554393730062, 2.40824914931121,   6.94733250995075});
    auto w1 = psca(0.5);

    auto S1 = rsdot(ivecl(sigma1)) + I;
    auto S2 = rsdot(ivecl(sigma2)) + I;

    auto Xmu1 = offset(X, -mu1), Xmu2 = offset(X, -mu2);
    auto p = sigmoid(w1);
    auto p1v = (1.0 / sqrt(pow(2 * numbers::pi, 10))) / sqrt(det(S1)) *
               exp(-0.5 * diag(Xmu1 * inv(S1), transpose(Xmu1)));
    auto p2v = (1.0 / sqrt(pow(2 * numbers::pi, 10))) / sqrt(det(S2)) *
               exp(-0.5 * diag(Xmu2 * inv(S2), transpose(Xmu2)));
    auto ll = -sum(ln(p * p1v + (1 - p) * p2v));
    GraphManager<> gm(ll);

    cout << "mu1 begin: " << mu1->val() << endl;
    cout << "mu2 begin: " << mu2->val() << endl;
    NloptOptimizer opt(ll);
    opt.run();
    std::cout << "mu1 end: " << mu1->val() << endl;
    cout << "mu2 end: " << mu2->val() << endl;
}