#ifndef __DYNAUTODIFF_LOSSESS__
#define __DYNAUTODIFF_LOSSESS__
#include "Common.hpp"
#include "Var.hpp"
#include <numbers>

#define UNWRAP(...) __VA_ARGS__
namespace DynAutoDiff {

template <typename T = double> struct BCELossEvalGrad : EvalGradFunctionBase<T> {
    int N = 0, R;
    BCELossEvalGrad(int Reduction) : R(Reduction){};
    // BCELossEvalGrad(int N) : N(N){};
    std::string get_name() const override { return "MSELossEvalGrad"; };
    boost::json::object to_json() const override {
        boost::json::object res;
        res["name"] = "cross_entropy_loss";
        res["Recution"] = R;
        return res;
    };
    void eval(TMap<T> &dest, const std::vector<TMap<T>> &inputs) override {
        const auto &input = inputs[0];
        const auto &target = inputs[1];
        N = input.rows();
        if (R == Reduction::None) {
            for (int i = 0; i < N; ++i) {
                if (target.coeff(i, 0) == 1) {
                    dest.coeffRef(i, 0) = target.coeff(i, 0) * std::log(input.coeff(i, 0));
                } else {
                    dest.coeffRef(i, 0) =
                        (1 - target.coeff(i, 0)) * std::log(1 - input.coeff(i, 0));
                }
            }
            dest = -dest;
        } else if (R == Reduction::Sum) {
            dest.coeffRef(0, 0) = 0;
            for (int i = 0; i < N; ++i) {
                if (target.coeff(i, 0) == 1) {
                    dest.coeffRef(0, 0) =
                        dest.coeff(0, 0) + target.coeff(i, 0) * std::log(input.coeff(i, 0));
                } else {
                    dest.coeffRef(0, 0) = dest.coeff(0, 0) + (1 - target.coeff(i, 0)) *
                                                                 std::log(1 - input.coeff(i, 0));
                }
            }
            dest = -dest;
            if (R == Reduction::Mean) {
                dest = dest / N;
            }
        }
    };
    std::vector<TMat<T>> grad(const std::shared_ptr<Var<T>> &current) override {
        std::vector<TMat<T>> res(2);
        const auto &input = current->input_node(0)->val();
        const auto &target = current->input_node(1)->val();

        [[likely]] if (current->input_node(0)->requires_grad()) {
            TMat<T> input_g(N, 1);

            for (int i = 0; i < N; ++i) {
                if (target.coeff(i, 0) == 1)
                    input_g.coeffRef(i, 0) = -1.0 / input.coeff(i, 0);
                else
                    input_g.coeffRef(i, 0) = 1.0 / (1 - input.coeff(i, 0));
            }
            if (R == Reduction::None) {
                res[0] = input_g.array() * current->grad().array();
            } else {
                res[0] = input_g * current->grad().coeff(0, 0);
                if (R == Reduction::Mean) {
                    res[0] = res[0] / N;
                }
            }
        }
        [[unlikely]] if (current->input_node(1)->requires_grad()) {
            TMat<T> target_g(N, 1);

            for (int i = 0; i < N; ++i) {
                if (target.coeff(i, 0) == 1)
                    target_g.coeffRef(i, 0) = -std::log(input.coeff(i, 0));
                else
                    target_g.coeffRef(i, 0) = std::log(1 - input.coeff(i, 0));
            }
            if (R == Reduction::None) {
                res[1] = target_g.array() * current->grad().array();
            } else {
                res[1] = target_g * current->grad().coeff(0, 0);
                if (R == Reduction::Mean) {
                    res[1] = res[1] / N;
                }
            }
        }
        return res;
    };
};

template <typename T = double> struct MSELossEvalGrad : EvalGradFunctionBase<T> {
    int N = 0, R;
    MSELossEvalGrad(int Reduction) : R(Reduction){};
    // BCELossEvalGrad(int N) : N(N){};
    std::string get_name() const override { return "MSELossEvalGrad"; };
    boost::json::object to_json() const override {
        boost::json::object res;
        res["name"] = "mse_loss";
        res["Recution"] = R;
        return res;
    };
    void eval(TMap<T> &dest, const std::vector<TMap<T>> &inputs) override {
        const auto &input = inputs[0];
        const auto &target = inputs[1];
        N = input.rows();
        if (R == Reduction::None) {
            dest = (input.array() - target.array()).pow(2);
        } else if (R == Reduction::Sum) {
            dest.coeffRef(0, 0) = (input.array() - target.array()).pow(2).sum();
            if (R == Reduction::Mean) {
                dest = dest / N;
            }
        }
    };
    std::vector<TMat<T>> grad(const std::shared_ptr<Var<T>> &current) override {
        std::vector<TMat<T>> res(2);
        const auto &input = current->input_node(0)->val();
        const auto &target = current->input_node(1)->val();

        [[likely]] if (current->input_node(0)->requires_grad()) {
            if (R == Reduction::None) {
                res[0] = 2 * (input.array() - target.array()) * current->grad().array();
            } else {
                res[0] = 2 * (input.array() - target.array()) * current->grad().coeff(0, 0);
                if (R == Reduction::Mean) {
                    res[0] = res[0] / N;
                }
            }
        }
        [[unlikely]] if (current->input_node(1)->requires_grad()) {
            if (current->input_node(0)->requires_grad()) {
                res[1] = -res[0];
            } else if (R == Reduction::None) {
                res[1] = -2 * (input.array() - target.array()) * current->grad().array();
            } else {
                res[1] = -2 * (input.array() - target.array()) * current->grad().coeff(0, 0);
                if (R == Reduction::Mean) {
                    res[1] = res[1] / N;
                }
            }
        }
        return res;
    };
};

#define LOSSFUNCTIONTEMPLATE(functionname, structname, assertstmt)                                 \
    template <typename T = double, Reduction R = Sum>                                              \
    std::shared_ptr<Var<T>> functionname(std::shared_ptr<Var<T>> input,                            \
                                         std::shared_ptr<Var<T>> target) {                         \
        assertstmt std::vector<std::shared_ptr<Var<T>>> input_nodes{input, target};                \
        if constexpr (R == Reduction::None) {                                                      \
            return std::make_shared<Var<T>>(                                                       \
                input->rows(), 1, (input->requires_grad() || target->requires_grad()),             \
                input_nodes, std::make_unique<structname##EvalGrad<T>>(R));                        \
        } else {                                                                                   \
            return std::make_shared<Var<T>>(                                                       \
                1, 1, (input->requires_grad() || target->requires_grad()), input_nodes,            \
                std::make_unique<structname##EvalGrad<T>>(R));                                     \
        }                                                                                          \
    };

LOSSFUNCTIONTEMPLATE(
    binary_cross_entropy, BCELoss,
    UNWRAP(assert(((input->rows() == target->rows()) && (input->cols() == 1) &&
                   (target->cols() == 1)) &&
                  "input and target must be two column vectors with the same rows.");))
// Allow input and target both matrices.
LOSSFUNCTIONTEMPLATE(mse_loss, MSELoss,
                     UNWRAP(assert(((input->rows() == target->rows()) &&
                                    (input->cols() == target->cols())) &&
                                   "input and target must have the same shape.");))
#undef LOSSFUNCTIONTEMPLATE
#undef UNWRAP
}; // namespace DynAutoDiff
#endif