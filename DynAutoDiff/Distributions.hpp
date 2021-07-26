#ifndef __DYNAUTODIFF_DISTRIBUTIONS__
#define __DYNAUTODIFF_DISTRIBUTIONS__
#include "Common.hpp"
#include "Var.hpp"
#include "boost/math/special_functions/gamma.hpp"
#include <boost/math/special_functions/math_fwd.hpp>
#include <numbers>

#define UNWRAP(...) __VA_ARGS__

namespace DynAutoDiff {
namespace bm = boost::math;
#define DISTFUNCTIONTEMPLATE(functionname, structname, functionargs, funinput, assertstmt,         \
                             requires_grad_stmt)                                                   \
    template <Reduction R = Sum, typename T = double>                                              \
    std::shared_ptr<Var<T>> functionname(std::shared_ptr<Var<T>> X functionargs) {                 \
        assertstmt std::vector<std::shared_ptr<Var<T>>> input_nodes{funinput};                     \
        if constexpr (R == Reduction::None) {                                                      \
            return std::make_shared<Var<T>>(X->rows(), 1, requires_grad_stmt, input_nodes,         \
                                            std::make_unique<structname##EvalGrad<T>>(R));         \
        } else {                                                                                   \
            return std::make_shared<Var<T>>(1, 1, requires_grad_stmt, input_nodes,                 \
                                            std::make_unique<structname##EvalGrad<T>>(R));         \
        }                                                                                          \
    };

template <typename T = double> struct LnMVNormalDenEvalGrad : EvalGradFunctionBase<T> {
    TMat<T> iS;
    TMat<T> Xmu;
    int N, d;
    int R;
    LnMVNormalDenEvalGrad(int R) : R(R){};
    double Sd; // Sigma determinant
    std::string get_name() const override { return "LnMVNormalDenEvalGrad"; };
    boost::json::object to_json() const override {
        boost::json::object res;
        res["name"] = "ln_mvnormal_den";
        res["Reduction"] = R;
        return res;
    };
    void eval(TMap<T> &dest, const std::vector<TMap<T>> &inputs) override {
        N = inputs[0].rows();
        d = inputs[0].cols();
        iS = inputs[2].inverse();
        Sd = inputs[2].determinant();

        Xmu.resize(N, d);
        TMat<T> res(N, 1);

        for (int i = 0; i < N; ++i) {
            Xmu.row(i) = (inputs[0].row(i).transpose() - inputs[1]).transpose();
            res.coeffRef(i, 0) = -Xmu.row(i) * iS * Xmu.row(i).transpose();
        };

        if (R == Reduction::None) {
            dest = 0.5 * (res.array() - std::log(Sd) - d * log(2 * std::numbers::pi));
        } else {
            dest.coeffRef(0, 0) =
                0.5 * (res.sum() - N * std::log(Sd) - N * d * log(2 * std::numbers::pi));

            if (R == Reduction::Mean) {
                dest = dest.array() / N;
            }
        }
    };
    std::vector<TMat<T>> grad(const std::shared_ptr<Var<T>> &current) override {
        const auto &X = current->input_node(0)->val();
        const auto &mu = current->input_node(1)->val();
        const auto &Sigma = current->input_node(2)->val();

        std::vector<TMat<T>> res;

        // X
        if (current->input_node(0)->requires_grad()) {
            TMat<T> Xg(N, d);
            if (R == Reduction::None) {
                for (int i = 0; i < N; ++i) {
                    Xg.row(i) =
                        -(iS * Xmu.row(i).transpose()).transpose() * current->grad().coeff(i, 0);
                }
            } else {
                Xg = -(iS * Xmu.transpose()).transpose() * current->grad().coeff(0, 0);
            }
            // std::cout << Xg << std::endl;
            if (R == Reduction::Mean) {
                Xg = Xg / N;
            }
            res.emplace_back(Xg);
        } else {
            res.emplace_back(TMat<T>());
        };
        // mu
        if (current->input_node(1)->requires_grad()) {
            TMat<T> mug(d, 1);
            // mu's gradient is negation of X's.
            if (current->input_node(0)->requires_grad()) {
                mug = -res[0].colwise().sum().transpose();
            } else {
                if (R == Reduction::None) {
                    TMat<T> Xg(N, d);
                    for (int i = 0; i < N; ++i) {
                        Xg.row(i) =
                            (iS * Xmu.row(i).transpose()).transpose() * current->grad().coeff(i, 0);
                    }
                    mug = Xg.colwise().sum().transpose();
                } else {
                    mug = (iS * Xmu.transpose()).rowwise().sum() * current->g();
                }
            }
            if (R == Reduction::Mean) {
                mug = mug / N;
            }
            res.emplace_back(mug);

        } else {
            res.emplace_back(TMat<T>());
        }
        // Sigma
        if (current->input_node(2)->requires_grad()) {
            TMat<T> Sg(d, d);

            if (R == Reduction::None) {
                Sg.setZero();
                for (int i = 0; i < N; ++i) {
                    Sg = Sg + iS.transpose() * Xmu.row(i).transpose() * Xmu.row(i) *
                                  iS.transpose() * current->grad().coeff(i, 0);
                }
                Sg = Sg - iS.transpose() * current->grad().sum();
            } else {
                Sg = iS.transpose() * Xmu.transpose() * Xmu * iS.transpose() *
                     current->grad().coeff(0, 0);
                Sg = Sg - iS.transpose() * N;
            }

            Sg = Sg * 0.5;

            if (R == Reduction::Mean) {
                Sg = Sg / N;
            }
            res.emplace_back(Sg);
        } else {
            res.emplace_back(TMat<T>());
        }

        return res;
    }
};

template <typename T = double> struct LnNormalDenEvalGrad : EvalGradFunctionBase<T> {
    TMat<T> Xmu;
    int N;
    int R;
    LnNormalDenEvalGrad(int R) : R(R){};
    std::string get_name() const override { return "LnNormalDenEvalGrad"; };
    boost::json::object to_json() const override {
        boost::json::object res;
        res["name"] = "ln_normal_den";
        res["Reduction"] = R;
        return res;
    };
    void eval(TMap<T> &dest, const std::vector<TMap<T>> &inputs) override {
        N = inputs[0].rows();
        T mu = inputs[1].coeff(0, 0);
        T sigma = inputs[2].coeff(0, 0);
        const auto &X = inputs[0];

        TMat<T> res(N, 1);

        res = (X.array() - mu).pow(2);

        if (R == Reduction::None) {
            dest = -0.5 * std::log(2 * std::numbers::pi) - std::log(sigma) -
                   0.5 * res.array() / (sigma * sigma);
        } else {
            dest.coeffRef(0, 0) = -0.5 * N * std::log(2 * std::numbers::pi) - N * std::log(sigma) -
                                  0.5 * res.sum() / (sigma * sigma);
            if (R == Reduction::Mean) {
                dest = dest.array() / N;
            }
        }
    };
    std::vector<TMat<T>> grad(const std::shared_ptr<Var<T>> &current) override {
        const auto &X = current->input_node(0)->val();
        T mu = current->input_node(1)->val().coeff(0, 0);
        T sigma = current->input_node(2)->val().coeff(0, 0);

        std::vector<TMat<T>> res;

        // X
        if (current->input_node(0)->requires_grad()) {
            TMat<T> Xg(N, 1);
            if (R == Reduction::None) {
                Xg = -(X.array() - mu) / (sigma * sigma) * current->grad().array();
            } else {
                Xg = -(X.array() - mu) / (sigma * sigma) * current->grad().coeff(0, 0);

                if (R == Reduction::Mean) {
                    Xg = Xg / N;
                }
            }
            // std::cout << Xg << std::endl;
            res.emplace_back(Xg);
        } else {
            res.emplace_back(TMat<T>());
        };
        // mu
        if (current->input_node(1)->requires_grad()) {
            TMat<T> mug(1, 1);
            // mu's gradient is negation of X's.
            if (current->input_node(0)->requires_grad()) {
                mug.coeffRef(0, 0) = -res[0].sum();
            } else {
                if (R == Reduction::None) {
                    mug.coeffRef(0, 0) =
                        ((X.array() - mu) * current->grad().array()).sum() / (sigma * sigma);
                } else {
                    mug.coeffRef(0, 0) =
                        ((X.array() - mu)).sum() / (sigma * sigma) * current->grad().coeff(0, 0);
                    if (R == Reduction::Mean) {
                        mug = mug / N;
                    }
                }
            }
            res.emplace_back(mug);
        } else {
            res.emplace_back(TMat<T>());
        }
        // Sigma
        if (current->input_node(2)->requires_grad()) {
            TMat<T> sg(1, 1);

            if (R == Reduction::None) {
                sg.coeffRef(0, 0) =
                    ((-1.0 / sigma + (X.array() - mu).pow(2) / (std::pow(sigma, 3))) *
                     current->grad().array())
                        .sum();
            } else {
                sg.coeffRef(0, 0) =
                    (-N / sigma + ((X.array() - mu).pow(2).sum()) / (std::pow(sigma, 3))) *
                    current->grad().coeff(0, 0);
                if (R == Reduction::Mean) {
                    sg = sg / N;
                }
            }
            res.emplace_back(sg);
        } else {
            res.emplace_back(TMat<T>());
        }

        return res;
    }
};

template <typename T = double> struct LnTDenEvalGrad : EvalGradFunctionBase<T> {
    TMat<T> Xmu;
    int N;
    int R;
    TMat<T> xm, xm2, fx, lfx;
    LnTDenEvalGrad(int R) : R(R){};
    std::string get_name() const override { return "LnTDenEvalGrad"; };
    boost::json::object to_json() const override {
        boost::json::object res;
        res["name"] = "ln_t_den";
        res["Reduction"] = R;
        return res;
    };
    void eval(TMap<T> &dest, const std::vector<TMap<T>> &inputs) override {
        N = inputs[0].rows();
        fx.resize(N, 1);
        xm.resize(N, 1);
        xm2.resize(N, 1);
        lfx.resize(N, 1);
        T mu = inputs[1].coeff(0, 0);
        T sigma = inputs[2].coeff(0, 0);
        T nu = inputs[3].coeff(0, 0);
        const auto &X = inputs[0];

        xm = X.array() - mu;
        xm2 = xm.array().pow(2);
        fx = xm2.array() / (sigma * sigma) / nu + 1;
        lfx = fx.array().log();

        if (R == Reduction::None) {
            dest = bm::lgamma(nu / 2 + 0.5) - bm::lgamma(nu / 2) -
                   0.5 * std::log(nu * std::numbers::pi) - std::log(sigma) -
                   (nu + 1) / 2 * lfx.array();
        } else {
            dest.coeffRef(0, 0) = N * (bm::lgamma(nu / 2 + 0.5) - bm::lgamma(nu / 2) -
                                       0.5 * std::log(nu * std::numbers::pi) - std::log(sigma)) -
                                  (nu + 1) / 2 * lfx.array().sum();
            if (R == Reduction::Mean) {
                dest = dest.array() / N;
            }
        }
    };
    std::vector<TMat<T>> grad(const std::shared_ptr<Var<T>> &current) override {
        const auto &X = current->input_node(0)->val();
        T mu = current->input_node(1)->val().coeff(0, 0);
        T sigma = current->input_node(2)->val().coeff(0, 0);
        T nu = current->input_node(3)->val().coeff(0, 0);

        std::vector<TMat<T>> res;

        // X
        if (current->input_node(0)->requires_grad()) {
            TMat<T> Xg(N, 1);
            if (R == Reduction::None) {
                Xg = -(1 + 1.0 / nu) / (sigma * sigma) * xm.array() / fx.array() *
                     current->grad().array();
            } else {
                Xg = -(1 + 1.0 / nu) / (sigma * sigma) * (xm.array() / fx.array()) *
                     current->grad().coeff(0, 0);

                if (R == Reduction::Mean) {
                    Xg = Xg / N;
                }
            }
            // std::cout << Xg << std::endl;
            res.emplace_back(Xg);
        } else {
            res.emplace_back(TMat<T>());
        };
        // mu
        if (current->input_node(1)->requires_grad()) {
            TMat<T> mug(1, 1);
            // mu's gradient is negation of X's.
            if (current->input_node(0)->requires_grad()) {
                mug.coeffRef(0, 0) = -res[0].sum();
            } else {
                if (R == Reduction::None) {
                    mug.coeffRef(0, 0) = (1 + 1.0 / nu) / (sigma * sigma) *
                                         (xm.array() / fx.array() * current->grad().array()).sum();
                } else {
                    mug.coeffRef(0, 0) = (1 + 1.0 / nu) / (sigma * sigma) *
                                         (xm.array() / fx.array()).sum() *
                                         current->grad().coeff(0, 0);
                    if (R == Reduction::Mean) {
                        mug = mug / N;
                    }
                }
            }
            res.emplace_back(mug);
        } else {
            res.emplace_back(TMat<T>());
        }
        // sigma
        if (current->input_node(2)->requires_grad()) {
            TMat<T> sg(1, 1);

            if (R == Reduction::None) {
                sg.coeffRef(0, 0) = ((-1.0 / sigma + (1 + 1.0 / nu) / (sigma * sigma * sigma) *
                                                         xm2.array() / fx.array()) *
                                     current->grad().array())
                                        .sum();
            } else {
                sg.coeffRef(0, 0) = (-N / sigma + (1 + 1.0 / nu) / (sigma * sigma * sigma) *
                                                      (xm2.array() / fx.array()).sum()) *
                                    current->grad().coeff(0, 0);
                if (R == Reduction::Mean) {
                    sg = sg / N;
                }
            }
            res.emplace_back(sg);
        } else {
            res.emplace_back(TMat<T>());
        }
        // nu
        if (current->input_node(3)->requires_grad()) {
            TMat<T> nug(1, 1);

            if (R == Reduction::None) {
                nug.coeffRef(0, 0) =
                    0.5 * ((bm::digamma((nu + 1) / 2) - bm::digamma(nu / 2) - 1.0 / nu -
                            (lfx.array() -
                             (nu + 1) / (nu * nu) / (sigma * sigma) * xm2.array() / fx.array())) *
                           current->grad().array())
                              .sum();
            } else {
                nug.coeffRef(0, 0) =
                    0.5 *
                    (N * (bm::digamma((nu + 1) / 2) - bm::digamma(nu / 2) - 1.0 / nu) -
                     (lfx.array() -
                      (nu + 1) / (nu * nu) / (sigma * sigma) * xm2.array() / fx.array())
                         .sum()) *
                    current->grad().coeff(0, 0);
                if (R == Reduction::Mean) {
                    nug = nug / N;
                }
            }
            res.emplace_back(nug);
        } else {
            res.emplace_back(TMat<T>());
        }
        return res;
    }
};

DISTFUNCTIONTEMPLATE(
    ln_mvnormal_den, LnMVNormalDen,
    UNWRAP(, std::shared_ptr<Var<T>> mu, std::shared_ptr<Var<T>> Sigma), UNWRAP(X, mu, Sigma),
    UNWRAP(assert(((mu->cols() == 1) && (Sigma->rows() == Sigma->cols())) &&
                  "mu should be a column vector. Sigma should be a square matrix.");),
    X->requires_grad() || mu->requires_grad() || Sigma->requires_grad())
DISTFUNCTIONTEMPLATE(
    ln_normal_den, LnNormalDen, UNWRAP(, std::shared_ptr<Var<T>> mu, std::shared_ptr<Var<T>> sigma),
    UNWRAP(X, mu, sigma),
    UNWRAP(assert(((X->cols() == 1) && (mu->size() == 1) && (sigma->size() == 1)) &&
                  "X must be a column vector. mu and sigma must be a scalar.");),
    X->requires_grad() || mu->requires_grad() || sigma->requires_grad())
DISTFUNCTIONTEMPLATE(
    ln_t_den, LnTDen,
    UNWRAP(, std::shared_ptr<Var<T>> mu, std::shared_ptr<Var<T>> sigma, std::shared_ptr<Var<T>> nu),
    UNWRAP(X, mu, sigma, nu),
    UNWRAP(assert(((X->cols() == 1) && (mu->size() == 1) && (sigma->size() == 1) &&
                   (nu->size() == 1)) &&
                  "X must be a column vector. mu, sigma and nu must be scalars.");),
    X->requires_grad() || mu->requires_grad() || sigma->requires_grad() || nu->requires_grad())
#undef DISTFUNCTIONTEMPLATE
#undef UNWRAP
};     // namespace DynAutoDiff
#endif // !