#ifndef __DYNAUTODIFF_NLOPTOPTIMIZER__
#define __DYNAUTODIFF_NLOPTOPTIMIZER__
#include "DynAutoDiff.hpp"
#include <nlopt.hpp>

#include <initializer_list>
#include <memory>

namespace DynAutoDiff {
// Due to the limitation of ceres, after computation. Gradient is lost and parmeter nodes now
// doesn't have grad.
template <typename T>
double NloptFunction(const std::vector<double> &x, std::vector<double> &grad, void *data) {
    auto obj = static_cast<T *>(data);
    obj->gm.bind(const_cast<double *>(x.data()), grad.data());
    obj->gm.zero_all();
    // obj->gm.run();
    if (!(grad.size() > 0)) {
        // std::cout << "nullptr" << std::endl;
        obj->root->eval();
    } else {
        obj->gm.run();
    }
    std::cout << obj->root->v() << std::endl;
    return obj->root->v();
}
template <typename T = double> requires std::same_as<T, double> struct NloptOptimizer {
    GraphManager<T> gm;
    std::shared_ptr<Var<T>> root;
    std::vector<T> parm_data;
    std::vector<T> parm_nodes;
    int nparm = 0;

    NloptOptimizer(std::shared_ptr<Var<T>> root,
                   std::initializer_list<std::shared_ptr<Var<T>>> parm_nodes)
        : root(root), gm(root), parm_nodes(parm_nodes) {
        // Automatically bind memory.
        for (int i = 0; i < this->parm_nodes.size(); ++i) {
            nparm += this->parm_nodes.size();
        }
        parm_data.resize(nparm);
        set_options();
    };
    // Automatically decide leaf nodes that need optimization.
    NloptOptimizer(std::shared_ptr<Var<T>> root) : root(root), gm(root) {
        parm_data.resize(gm.nparm());
        gm.copy_parm_val_to(parm_data.data());
        set_options();
    };
    void set_options() {}

    void run() {

        nlopt::opt prob(nlopt::LD_LBFGS, parm_data.size());
        prob.set_min_objective(&NloptFunction<struct NloptOptimizer>, this);
        prob.set_maxeval(10000);
        prob.set_xtol_rel(1e-8);
        double fv;
        nlopt::result res = prob.optimize(parm_data, fv);

        gm.bind(parm_data.data(), nullptr);
    };
};
}; // namespace DynAutoDiff
#endif