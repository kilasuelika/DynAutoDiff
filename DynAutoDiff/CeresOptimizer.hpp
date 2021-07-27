#ifndef __DYNAUTODIFF_CERESOPTIMIZER__
#define __DYNAUTODIFF_CERESOPTIMIZER__
#include "DynAutoDiff.hpp"
#include "ceres/ceres.h"
#include <ceres/types.h>
#include <glog/logging.h>
#include <initializer_list>
#include <memory>

namespace DynAutoDiff {
// Due to the limitation of ceres, after computation. Gradient is lost and parmeter nodes now
// doesn't have grad.
template <typename T = double> requires std::same_as<T, double> struct CeresOptimizer {
    GraphManager<T> gm;
    std::shared_ptr<Var<T>> root;
    std::vector<T> parm_data;
    std::vector<T> parm_nodes;
    int nparm = 0;

    struct ProblemFunctor : public ceres::FirstOrderFunction {
        GraphManager<T> &gm;
        ProblemFunctor(GraphManager<T> &gm) : gm(gm) {}

        bool Evaluate(const double *parameters, double *cost, double *gradient) const override {
            gm.copy_parm_val_from(parameters);
            gm.zero_all();
            gm.run();
            gm.copy_parm_grad_to(gradient);
            *cost = gm.root()->val().coeff(0, 0);
            return true;
        }
        int NumParameters() const override { return gm.nparm(); }
    };

    CeresOptimizer(std::shared_ptr<Var<T>> root,
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
    CeresOptimizer(std::shared_ptr<Var<T>> root) : root(root), gm(root) {
        parm_data.resize(gm.nparm());
        gm.copy_parm_val_to(parm_data.data());
        set_options();

        gm.auto_bind_parm();
    };
    void set_options() {
        // Options.
        options.minimizer_progress_to_stdout = true;
        options.line_search_direction_type = ceres::LineSearchDirectionType::LBFGS;
        options.max_num_iterations = 1000;
        options.max_num_line_search_direction_restarts = 50;
        options.max_num_line_search_step_size_iterations = 50;
        // Allow: QUADRATIC, CUBIC
        options.line_search_interpolation_type = ceres::CUBIC;
    }
    ceres::GradientProblemSolver::Options options;
    ceres::GradientProblemSolver::Summary summary;

    void run() {
        google::InitGoogleLogging("DynAutoDiff with ceres");
        ceres::GradientProblem problem(new ProblemFunctor(gm));
        ceres::Solve(options, problem, parm_data.data(), &summary);
        gm.bind(parm_data.data(), nullptr);
        std::cout << summary.FullReport() << std::endl;
        google::ShutdownGoogleLogging();
    };
};
}; // namespace DynAutoDiff
#endif