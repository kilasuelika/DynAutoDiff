#ifndef __DYNAUTODIFF__
#define __DYNAUTODIFF__
#include "EigenHelper.hpp"
#include <boost/json/src.hpp>
#include <cmath>
#include <concepts>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <stack>
#include <stdexcept>
#include <utility>
#include <vector>

#define UNWRAP(...) __VA_ARGS__
#define STRING(s) #s

namespace DynAutoDiff {

template <typename T> class VarImpl;
template<typename T> class Var;
template <typename T> using SVarImpl = std::shared_ptr<VarImpl<T>>;


using SVarImpld = std::shared_ptr<VarImpl<double>>;

template <typename T> struct EvalGradFunctionBase {
    virtual std::string get_name() const = 0;
    virtual boost::json::object to_json() const = 0;
    virtual void eval(TMap<T> &dest, const std::vector<TMap<T>> &inputs) = 0;
    virtual std::vector<TMat<T>> grad(const Var<T> &current) = 0;
};

template <typename T1, typename T2> static inline bool shape_match(const T1 &l, const T2 &r) {
    return (l.rows() == r.rows()) && (l.cols() == r.cols());
};
static std::string _shape_str(int r1, int c1, int r2, int c2) {
    return "(" + std::to_string(r1) + ", " + std::to_string(c1) + ") and (" + std::to_string(r2) +
           ", " + std::to_string(c2) + ")";
};
static inline int _matmul_result_rows(int r1, int c1, int r2, int c2) {
    if (r1 * c1 == 1) {
        return r2;
    } else {
        return r1;
    }
};
static inline int _matmul_result_cols(int r1, int c1, int r2, int c2) {
    if (r2 * c2 == 1) {
        return c1;
    } else {
        return c2;
    }
};

template <typename T = double> class VarImpl : public std::enable_shared_from_this<struct VarImpl<T>> {

  private:
    template <typename T1> friend class GraphManager;
    template <typename T1> friend struct NaiveGD;

    TMap<T> _val, _grad;
    int _rows = 0;
    int _cols = 0;
    int _size = 0;
    int _id = 0;                         // For saving and loading graph.
    bool _own_v = false, _own_g = false; // If true, then data is managed by outside.
    bool _requires_grad = false;
    bool _evaluated = false;
    bool _visited = false;
    bool _leaf = true;

    // bool _is_leaf;

    // std::vector<std::shared_ptr<struct VarImpl<T>>> _input_nodes;
    std::vector<Var<T>> _input_nodes;

    std::allocator<T> _alloc;

    std::unique_ptr<EvalGradFunctionBase<T>> fn_ = nullptr;

    void _bind(T *val, T *grad, int rows, int cols, bool own_v, bool own_g) {
        if (val != _val.data()) {
            new (&_val) TMap<T>(val, rows, cols);
            _own_v = own_v;

            // Set values to zero.
            if (own_v) {
                _val.setZero();
            }

            _rows = rows;
            _cols = cols;
            _size = rows * cols;
        };
        if (grad != nullptr) {
            if (grad != _grad.data()) {
                new (&_grad) TMap<T>(grad, rows, cols);
                _own_g = own_g;
                _requires_grad = true;

                if (own_g) {
                    _grad.setZero();
                }
            }
        } else {
            _own_g = false;
            _requires_grad = false;
        };
    };

  public:
    using Scalar = T;

    VarImpl(){};
    VarImpl(int rows, int cols, bool requires_grad = false, bool allocate = true)
        : _rows(rows), _cols(cols), _size(rows * cols), _requires_grad(requires_grad),
          _val(nullptr, 0, 0), _grad(nullptr, 0, 0) {
        if (allocate) {
            T *v = _alloc.allocate(rows * cols);
            T *g = nullptr;
            if (requires_grad) {
                // std::cout << "Allocating gradient memory " << rows << "x" << cols << std::endl;
                g = _alloc.allocate(rows * cols);
            };
            _bind(v, g, rows, cols, true, requires_grad);
        }
    };
    VarImpl(int rows, int cols, bool requires_grad,
        const std::vector<Var<T>> &input_nodes,
        std::unique_ptr<EvalGradFunctionBase<T>> fn)
        : _rows(rows), _cols(cols), _size(rows * cols), _requires_grad(requires_grad),
          _val(nullptr, 0, 0), _grad(nullptr, 0, 0), _input_nodes(input_nodes), fn_(std::move(fn)) {
        // Allocate memory for values.
        T *v = _alloc.allocate(rows * cols);
        T *g = nullptr;
        if (requires_grad) {
            // std::cout << "Allocating gradient memory " << rows << "x" << cols << std::endl;
            g = _alloc.allocate(rows * cols);
        };
        _bind(v, g, rows, cols, true, requires_grad);
    };
    VarImpl(T *val, T *grad, int rows, int cols)
        : _rows(rows), _cols(cols), _size(rows * cols), _requires_grad(requires_grad),
          _val(nullptr, 0, 0), _grad(nullptr, 0, 0) {
        _bind(val, grad, rows, cols, false, false);
    };
    // Read data from file.
    VarImpl(const std::string &file, bool requires_grad = false)
        : _val(nullptr, 0, 0), _grad(nullptr, 0, 0) {
        auto [rows, cols] = decide_shape<T>(file);
        _rows = rows;
        _cols = cols;
        T *v = _alloc.allocate(_rows * _cols);
        T *g = nullptr;
        if (requires_grad) {
            // std::cout << "Allocating gradient memory " << rows << "x" << cols << std::endl;
            g = _alloc.allocate(rows * cols);
        };
        _bind(v, g, rows, cols, true, requires_grad);
        // Read data
        std::ifstream f(file);
        for (int i = 0; i < _rows; ++i) {
            for (int j = 0; j < _cols; ++j) {
                f >> _val.coeffRef(i, j);
            }
        }
    };
    /*VarImpl(const TMat &val){

    };*/
    void release_memory() {
        // Remember to deallocate memory.
        if (_own_v) {
            // std::cout << "Deallocating value memory. " << std::endl;
            _alloc.deallocate(_val.data(), _size);
        }
        if (_requires_grad) {
            if (_own_g) {
                _alloc.deallocate(_grad.data(), _size);
            };
        };
    };
    ~VarImpl() { release_memory(); };

    void setRandom() { _val.setRandom(); };
    void setRandomNormal(int seed = -1) {
        if (seed == -1) {
            std::random_device rd;
            seed = rd();
        };
        std::default_random_engine eng(seed);
        std::normal_distribution dist(T(0.0), T(1.0));
        std::for_each(this->vbegin(), this->vend(), [&eng, &dist](T &x) mutable { x = dist(eng); });
    };

    void set_fn(std::unique_ptr<EvalGradFunctionBase<T>>&& v) {
        fn_=std::move(v);
    }
    std::unique_ptr<EvalGradFunctionBase<T>>& fn() {return fn_;}

    void set_id(int i) {_id=i;}
    const int& id() const {return _id;}
    bool is_leaf() const { return _leaf; };
    void set_leaf(bool v) {_leaf=v;}
    int rows() const { return _rows; };
    int cols() const { return _cols; };
    int size() const { return _size; };
    int input_size(int k) const { return _input_nodes[k].size(); };
    // constant value begin.
    const T *cvbegin() const { return _val.data(); };
    const T *cvend() const { return _val.data() + _size; };
    T *vbegin() { return _val.data(); };
    T *vend() { return _val.data() + _size; };
    const T *cgbegin() const { return _grad.data(); };
    const T *cgend() const { return _grad.data() + _size; };
    T *gbegin() { return _grad.data(); };
    T *gend() { return _grad.data() + _size; };
    bool requires_grad() const { return _requires_grad; };

    void bind(T *val, T *grad, int rows, int cols) {
        release_memory();

        _bind(val, grad, rows, cols, false, false);
    };
    void bind(T *v, T *g) {
        release_memory();

        _bind(v, g, _rows, _cols, false, (g == nullptr));
    };
    void resize(int rows, int cols, bool requires_grad = false) {
        release_memory();

        T *v = _alloc.allocate(rows * cols);
        T *g = nullptr;
        if (requires_grad) {
            g = _alloc.allocate(rows * cols);
        };
        _bind(v, g, rows, cols, true, requires_grad);
    };
    bool visited() const {return _visited;}
    void set_visited(bool v) { _visited=v; }
    bool evaluated() const {return _evaluated;}
    void set_evaluated(bool v) {
        _evaluated=v;
    }
    // auto evaluated() const { return _evaluated; };

    const auto &val() const { return _val; };
    void set_val(const TMat<T>& m) {_val=m;}
    // const auto& va()const {return _val.array();};
    T v(int i = -1, int j = -1) const {
        if (i == -1 && j == -1) {
            return _val.coeff(0, 0);
        } else if (j == -1) {
            if (i >= _size)
                throw std::range_error("Subscript for v() is out-of-bound.");
            if (_rows == 1) {
                return _val.coeff(0, i);
            } else if (_cols == 1) {
                return _val.coeff(i, 0);
            } else {
                throw std::range_error("Invalid subscript. This is a matrix");
            }
        } else {
            return _val.coeff(i, j);
        }
    }
    void set_grad(const TMat<T>& m) {_grad=m;}
    const auto &grad() const {
        if (!_requires_grad)
            throw(std::range_error("This node has no gradient."));
        else
            return _grad;
    };
    T g(int i = -1, int j = -1) const {
        if (requires_grad()) {
            if (i == -1 && j == -1) {
                return _grad.coeff(0, 0);
            } else if (j == -1) {
                if (i >= _size)
                    throw std::range_error("Subscript for g() is out-of-bound.");
                if (_rows == 1) {
                    return _grad.coeff(0, i);
                } else if (_cols == 1) {
                    return _grad.coeff(i, 0);
                } else {
                    throw std::range_error("Invalid subscript. This is a matrix");
                }
            } else {
                if (i * _cols + j + 1 > _size) {
                    throw std::range_error("Subscript for g() is out-of-bound.");
                } else
                    return _grad.coeff(i, j);
            }
        } else {
            throw std::range_error("This node has no gradient.");
        }
    };
    // const auto& ga()const {return _grad.array();};
    const auto &input_nodes() const { return _input_nodes; };
    void add_input_node(const Var<T>& v){_input_nodes.emplace_back(v);}
    auto input_node(int k) const { return _input_nodes[k]; };

    const TMap<T> &eval() {
        if (_evaluated) {
            return _val;
        }
        if (_input_nodes.size() > 0) {
            std::vector<TMap<T>> inputs;
            for (size_t i = 0; i < _input_nodes.size(); ++i) {
                inputs.emplace_back(_input_nodes[i].eval());
            };
            fn_->eval(_val, inputs);
        }
        _evaluated = true;
        return _val;
    };
    void backward(const TMat<T> &seed) {
        if (_requires_grad) {
            if (_leaf)
                _grad = _grad + seed;
            else
                _grad = seed;
            if (_input_nodes.size() > 0) {
                std::vector<TMat<T>> grads = fn_->grad(this->shared_from_this());
                for (int i = 0; i < _input_nodes.size(); ++i) {
                    _input_nodes[i].backward(grads[i]);
                }
            }
        }
    }
    // eval() and backward()
    const TMap<T> &evalb() {
        eval();
        backward();
        return _val;
    }
    void backward() {
        TMat<T> seed = TMat<T>::Ones(_rows, _cols);
        backward(seed);
    }
    // Assign values.
    void operator=(std::initializer_list<T> v) {
        assert(
            (v.size() == _size) &&
            "Input size should be equal to the size of VarImpliable for operator = initializer_list.");
        // std::copy(v.begin(), v.end(), _val.data());
        std::copy(v.begin(), v.end(), _val.data());
    };

    void zero_grad() {
        _grad.setZero();
    }
    // Inplacement assign;
    void operator=(const std::shared_ptr<struct VarImpl<T>> &VarImpl){

    };
    // Assign Eigen data.
    template <typename TD> void operator=(const TD &data) { _val = data; }
};



template<typename T>
class Var {
public:
    Var(std::shared_ptr<VarImpl<T>> ptr):
        var_(std::move(ptr)) {

    }

    /**
     * Operators
     *
     *
     **/

    Var<T> inv() const {
        return inv(this);
    }

/**
 *info
 **/
    void set_id(int i)const {var_->set_id(i);}
    int id() const {return var_->id();}
    int rows() const { return var_->rows(); }
    int cols() const { return var_->cols(); }
    int size() const { return var_->size(); }
    int input_size(int k) const { return var_->input_size(k); }
    const auto &input_nodes() const { return var_->input_nodes(); };
    auto input_node(int k) const { return var_->input_node(k); }
    bool requires_grad() const { return var_->requires_grad(); }
    void set_visited(bool v) const { var_->set_visited(v); }
    bool visited() const {return var_->visited(); }
    void set_evaluated(bool v)  { var_->set_evaluated( v); }
    bool evaluated() const {return var_->evaluated(); }
    bool is_leaf() const {return var_->is_leaf(); }
    void set_leaf(bool v) const {var_->set_leaf(v);}
    // Member access
    void add_input_node(const Var<T>& v){var_->add_input_node(v);}
    // Data access
    T v(int i = -1, int j = -1) const {
        return var_->v(i, j);
    }
    const auto &grad() const {return var_->grad();}
    void set_grad(const TMat<T>& m) const {var_->set_grad(m);}
    const auto &val() const {return var_->val();}
    void set_val(const TMat<T>& m) const {var_->set_val(m);}
    T g(int i = -1, int j = -1) const {return var_->g(i,j);}
    void set_fn(std::unique_ptr<EvalGradFunctionBase<T>>&& v) const {var_->set_fn(std::move(v));}
    std::unique_ptr<EvalGradFunctionBase<T>>& fn() const {return var_->fn();}
    //grad
    const TMap<T> & eval() const {return var_->eval();}
    void backward() const {var_->backward();}
    void backward(const TMat<T> &seed) const {var_->backward(seed);}
    const TMap<T> &evalb() const {return var_->evalb();}
    void zero_grad()const {var_->zero_grad();}

    // Iterator
    const T *cvbegin() const { return var_->cvbegin(); }
    const T *cvend() const { return var_->cvend(); };
    T *vbegin() { return var_->vbegin(); };
    T *vend() { return var_->vend(); };
    const T *cgbegin() const { return var_->cgbegin(); };
    const T *cgend() const { return var_->cgend(); };
    T *gbegin() { return var_->gbegin(); };
    T *gend() { return var_->gend(); };
protected:
    std::shared_ptr<VarImpl<T>> var_;
};

//----------------------------------------------------------------------------------------------------------
// Convinient function for initializing VarImpliables.
template <typename T = double>
inline Var<T> sca(const T &v, bool requires_grad = false) {
    auto res = std::make_shared<VarImpl<T>>(1, 1, requires_grad);
    *res = {v};
    return res;
};
// p for parameter.
template <typename T = double> inline Var<T> psca(T v = 0.0) {
    return sca<T>(v, true);
};
// c for constant
template <typename T = double> inline Var<T> csca(T v = 0.0) {
    return sca<T>(v, false);
};
template <std::floating_point T = double>
Var<T> mat(std::initializer_list<T> v, int rows, int cols,
                            bool requires_grad = false) {
    auto res = std::make_shared<VarImpl<T>>(rows, cols, requires_grad);
    *res = v;
    return res;
};
template <std::floating_point T = double>
Var<T> mat(const TMat<T> &m, bool requires_grad = false) {
    auto res = std::make_shared<VarImpl<T>>(m.rows(), m.cols(), requires_grad);
    *res = m;
    return res;
};
// p for parameter.
template <typename T = double> inline Var<T> pmat(int rows, int cols) {
    return std::make_shared<VarImpl<T>>(rows, cols, true);
};
// p for parameter.
template <typename T = double>
inline Var<T> pmat(std::initializer_list<T> v, int rows, int cols) {
    assert((v.size() == rows * cols) && "Data size is not equal to rows*cols");
    auto res = std::make_shared<VarImpl<T>>(rows, cols, true);
    *res = v;
    return res;
};
// c for constant
template <typename T = double> inline Var<T> cmat(int rows, int cols) {
    return std::make_shared<VarImpl<T>>(rows, cols, false);
};
template <typename T = double>
Var<T> mat(int rows, int cols, bool requires_grad = false) {
    auto res = std::make_shared<VarImpl<T>>(rows, cols, requires_grad);
    return res;
};
template <typename T = double>
Var<T> rowvec(std::initializer_list<T> v, bool requires_grad = false) {
    auto res = std::make_shared<VarImpl<T>>(1, v.size(), requires_grad);
    *res = v;
    return res;
};
template <typename T = double> Var<T> prowvec(std::initializer_list<T> v) {
    auto res = std::make_shared<VarImpl<T>>(1, v.size(), true);
    *res = v;
    return res;
};
template <std::floating_point T = double>
inline Var<T> vec(std::initializer_list<T> v, bool requires_grad = false) {
    auto res = std::make_shared<VarImpl<T>>(v.size(), 1, requires_grad);
    *res = v;
    return res;
};
template <std::floating_point T = double>
inline Var<T> vec(int n, bool requires_grad = false) {
    auto res = std::make_shared<VarImpl<T>>(n, 1, requires_grad);
    return res;
}
/**
 * pvec: parameter vector
 * cvec: constant vector
 **/
template <typename T = double> inline Var<T> pvec(int n) {
    auto res = std::make_shared<VarImpl<T>>(n, 1, true);
    return res;
};
template <typename T = double> inline Var<T> pvec(std::initializer_list<T> v) {
    auto res = std::make_shared<VarImpl<T>>(v.size(), 1, true);
    *res = v;
    return res;
};
template <typename T = double> inline Var<T> cvec(std::initializer_list<T> v) {
    auto res = std::make_shared<VarImpl<T>>(v.size(), 1, false);
    *res = v;
    return res;
};
template <typename T = double>
void bind_seq(T *v, T *g, std::initializer_list<Var<T>> VarImpls) {
    int vi = 0, gi = 0;
    for (auto &VarImpl : VarImpls) {
        VarImpl->bind(v + vi, g + gi);
        vi += VarImpl.size();
        gi += (VarImpl.requires_grad()) * VarImpl.size();
    };
};
template <typename T = double> auto load(const std::string &file, bool requires_grad = false) {
    return std::make_shared<VarImpl<T>>(file, requires_grad);
}
//-------------------------------------------------------------------------------------------------------------
// Arithmetic
// template <typename T = double> class VarImpl { Var<T> _impl; };
// negation
template <typename T> struct NegationEvalGrad : EvalGradFunctionBase<T> {
    std::string get_name() const override { return "NegationEvalGrad"; };
    boost::json::object to_json() const override {
        boost::json::object res;
        res["name"] = "negation";
        return res;
    };
    void eval(TMap<T> &dest, const std::vector<TMap<T>> &inputs) override { dest = -inputs[0]; };
    std::vector<TMat<T>> grad(const Var<T> &current) override {
        return {-current.grad().array()};
    };
};

template <typename T> Var<T> operator-(const Var<T> &operand) {
    std::vector<Var<T>> input_nodes{operand};
    return std::make_shared<VarImpl<T>>(operand.rows(), operand.cols(), (operand.requires_grad()),
                                    input_nodes, std::make_unique<NegationEvalGrad<T>>());
};
// + - * /
#define BINARYARITHOP(sym, name, eval_op, result_rows, result_cols, backward_operation_lhs1_0,     \
                      backward_operation_lhs1_1, backward_operation_rhs1_0,                        \
                      backward_operation_rhs1_1, backward_operation_0, backward_operation_1,       \
                      assert_cond, assert_message)                                                 \
    template <typename T> struct name##EvalGrad : EvalGradFunctionBase<T> {                        \
        std::string get_name() const override { return STRING(name##EvalGrad); };                  \
        boost::json::object to_json() const override {                                             \
            boost::json::object res;                                                               \
            std::cout << this->get_name() << std::endl;                                            \
            res["name"] = STRING(name##EvalGrad);                                                  \
            return res;                                                                            \
        };                                                                                         \
        void eval(TMap<T> &dest, const std::vector<TMap<T>> &inputs) override {                    \
            if (inputs[0].size() == 1) {                                                           \
                dest = inputs[0].coeff(0, 0) sym inputs[1].array();                                \
            } else if (inputs[1].size() == 1) {                                                    \
                dest = inputs[0].array() sym inputs[1].coeff(0, 0);                                \
            } else {                                                                               \
                dest = eval_op;                                                                    \
            }                                                                                      \
        };                                                                                         \
        std::vector<TMat<T>> grad(const Var<T> &current) override {               \
            std::vector<TMat<T>> res(2);                                                           \
            const auto &G = current.grad();                                                       \
            const auto &L = current.input_node(0).val();                                         \
            const auto &R = current.input_node(1).val();                                         \
            if (current.input_size(0) == 1) {                                                     \
                if (current.input_node(0).requires_grad())                                       \
                    res[0] = backward_operation_lhs1_0;                                            \
                if (current.input_node(1).requires_grad())                                       \
                    res[1] = backward_operation_lhs1_1;                                            \
            } else if (current.input_size(1) == 1) {                                              \
                if (current.input_node(0).requires_grad())                                       \
                    res[0] = backward_operation_rhs1_0;                                            \
                if (current.input_node(1).requires_grad())                                       \
                    res[1] = backward_operation_rhs1_1;                                            \
            } else {                                                                               \
                if (current.input_node(0).requires_grad())                                       \
                    res[0] = backward_operation_0;                                                 \
                if (current.input_node(1).requires_grad())                                       \
                    res[1] = backward_operation_1;                                                 \
            };                                                                                     \
            return res;                                                                            \
        };                                                                                         \
    };                                                                                             \
    template <typename T>                                                                          \
    Var<T> operator sym(const Var<T> &lhs,                       \
                                         const Var<T> &rhs) {                     \
        assert((assert_cond) && assert_message);                                                   \
        std::vector<Var<T>> input_nodes{lhs, rhs};                                \
        auto res = std::make_shared<VarImpl<T>>(result_rows, result_cols,                              \
                                            (lhs.requires_grad() || rhs.requires_grad()),        \
                                            input_nodes, std::make_unique<name##EvalGrad<T>>());   \
        return res;                                                                                \
    };                                                                                             \
    template <typename T, typename ST>                                                             \
    Var<T> operator sym(const Var<T> &lhs, const ST &scav) {     \
        auto rhs = sca(static_cast<T>(scav));                                                      \
        return lhs sym rhs;                                                                        \
    };                                                                                             \
    template <typename T, typename ST>                                                             \
    Var<T> operator sym(const ST &scav, const Var<T> &rhs) {     \
        auto lhs = sca(static_cast<T>(scav));                                                      \
        return lhs sym rhs;                                                                        \
    };

// Plus
BINARYARITHOP(
    +, Plus, inputs[0].array() + inputs[1].array(), UNWRAP(std::max(lhs.rows(), rhs.rows())),
    UNWRAP(std::max(lhs.cols(), rhs.cols())), UNWRAP(eigen_scalar_mat(T(current.grad().sum()))),
    UNWRAP(current.grad()), UNWRAP(current.grad()),
    UNWRAP(eigen_scalar_mat(T(current.grad().sum()))), UNWRAP(current.grad()),
    UNWRAP(current.grad()), shape_match(lhs, rhs) || (lhs.size() == 1) || (rhs.size() == 1),
    "Mat shape must be equal for + operator or at least one of them is actually a scalar.")
// minus
BINARYARITHOP(
    -, Minus, inputs[0].array() - inputs[1].array(), UNWRAP(std::max(lhs.rows(), rhs.rows())),
    UNWRAP(std::max(lhs.cols(), rhs.cols())), UNWRAP(eigen_scalar_mat(T(current.grad().sum()))),
    UNWRAP(-current.grad()), UNWRAP(current.grad()),
    UNWRAP(eigen_scalar_mat(-T(current.grad().sum()))), UNWRAP(current.grad()),
    UNWRAP(-current.grad()), shape_match(lhs, rhs) || (lhs.size() == 1) || (rhs.size() == 1),
    "Mat shape must be equal for - operator or at least one of them is actually a scalar.")
// times: matrix product.
BINARYARITHOP(
        *, Times, inputs[0] * inputs[1],
        UNWRAP(_matmul_result_rows(lhs.rows(), lhs.cols(), rhs.rows(), rhs.cols())),
        UNWRAP(_matmul_result_cols(lhs.rows(), lhs.cols(), rhs.rows(), rhs.cols())),
        UNWRAP(eigen_scalar_mat(T((G.array() * R.array()).sum()))), UNWRAP(G *L.coeff(0, 0)),
        UNWRAP(G *R.coeff(0, 0)), UNWRAP(eigen_scalar_mat(T((G.array() * L.array()).sum()))),
        UNWRAP(G *R.transpose()), UNWRAP(L.transpose() * G),
        (lhs.cols() == rhs.rows()) || (lhs.size() == 1) || (rhs.size() == 1),
        UNWRAP(
            "Cols of left matrix should be equal to rows of the right matrix for matrix "
            "multiplication operator * or at least one of them is actually a scalar. Shapes are: "))
// division
BINARYARITHOP(
    /, Division, inputs[0].array() + inputs[1].array(), UNWRAP(std::max(lhs.rows(), rhs.rows())),
    UNWRAP(std::max(lhs.cols(), rhs.cols())),
    UNWRAP(eigen_scalar_mat(T((G.array() / R.array()).sum()))),
    UNWRAP(-L.coeff(0, 0) * G.array() / R.array().pow(2)), UNWRAP(G.array() * R.coeff(0, 0)),
    UNWRAP(eigen_scalar_mat(-T((G.array() * L.array()).sum()) / std::pow(R.coeff(0, 0), 2))),
    UNWRAP(G.array() / R.array()), UNWRAP(-G.array() * L.array() / R.array().pow(2)),
    shape_match(lhs, rhs) || (lhs.size() == 1) || (rhs.size() == 1),
    "Mat shape must be equal for elementwise division operator / or at least one of them "
    "is actually a scalar.")

// Element-wise product
template <typename T = double> struct Eprod2EvalGrad : EvalGradFunctionBase<T> {
    std::string get_name() const override { return "Eprod2EvalGrad"; };
    boost::json::object to_json() const override {
        boost::json::object res;
        res["name"] = "Eprod2EvalGrad";
        return res;
    };
    void eval(TMap<T> &dest, const std::vector<TMap<T>> &inputs) override {
        const auto &X = inputs[0];
        const auto &Y = inputs[1];
        dest = X.array() * Y.array();
    };
    std::vector<TMat<T>> grad(const Var<T> &current) {
        const auto &X = current.input_node(0).val();
        const auto &Y = current.input_node(1).val();
        const auto &G = current.grad();
        return {G.array() * Y.array(), G.array() * X.array()};
    };
};
template <typename T = double>
Var<T> eprod(const Var<T> &X, const Var<T> &Y) {
    assert(((X.rows() == Y.rows()) && (X.cols() && Y.cols())) &&
           "A and B must have the same shape for element-wise product eprod(X, Y) operator.");
    std::vector<Var<T>> input_nodes{X, Y};
    return std::make_shared<VarImpl<T>>(1, 1, (X.requires_grad() || Y.requires_grad()), input_nodes,
                                    std::make_unique<Eprod2EvalGrad<T>>());
};

// Offset.
template <typename T = double> struct OffsetEvalGrad : EvalGradFunctionBase<T> {
    int D;
    OffsetEvalGrad(int D) : D(D){};
    std::string get_name() const override { return "OffsetEvalGrad"; };
    boost::json::object to_json() const override {
        boost::json::object res;
        res["name"] = "OffsetEvalGrad";
        res["D"] = D;
        return res;
    };
    void eval(TMap<T> &dest, const std::vector<TMap<T>> &inputs) override {
        const auto &X = inputs[0];
        const auto &x = inputs[1];
        if (D == 0) {
            dest = X;
            for (auto row : dest.rowwise()) {
                row = row + x;
            }
        } else if (D == 1) {
            dest = X;
            for (auto col : dest.colwise()) {
                col = col + x;
            }
        } else {
            throw std::range_error("Shape mismatch for operator offset(X, x).");
        }
    };
    std::vector<TMat<T>> grad(const Var<T> &current) {
        const auto &X = current.input_node(0).val();
        const auto &x = current.input_node(1).val();
        const auto &Ga = current.grad().array();
        if (D == 0) {
            // x is a row-vector
            return {Ga, Ga.colwise().sum()};
        } else {
            return {Ga, Ga.rowwise().sum()};
        }
    };
};
template <typename T = double>
Var<T> offset(const Var<T> &X, const Var<T> &x) {
    assert((((x.cols() == 1) && (X.rows() == x.rows())) ||
            ((x.rows() == 1) && (X.cols() == x.cols()))) &&
           "Please check shape of input to offset(X, x), X must be a matrix. x is a row of column "
           "vector.");
    if ((x.cols() == 1) && (X.rows() == x.rows())) {
        std::vector<Var<T>> input_nodes{X, x};
        return std::make_shared<VarImpl<T>>(X.rows(), X.cols(),
                                        (X.requires_grad() || x.requires_grad()), input_nodes,
                                        std::make_unique<OffsetEvalGrad<T>>(1));
    } else if ((x.rows() == 1) && (X.cols() == x.cols())) {
        std::vector<Var<T>> input_nodes{X, x};
        return std::make_shared<VarImpl<T>>(X.rows(), X.cols(),
                                        (X.requires_grad() || x.requires_grad()), input_nodes,
                                        std::make_unique<OffsetEvalGrad<T>>(0));
    } else {
        throw std::range_error("Shape mismatch for offsex(X, x).");
    }
};

//-------------------------------------------------------------------------------------------------------------
// Unary functions
#define UNARYFUNCTION(functionname, structname, rows, cols, evalop, gradop, datamember, funargs,   \
                      funVarImpl, evalextra, gradextra, assertstmt, constructorarg, constructorinit,   \
                      to_json_exp)                                                                 \
    template <typename T = double> struct structname##EvalGrad : EvalGradFunctionBase<T> {         \
        std::string get_name() const override { return STRING(structname##EvalGrad); };            \
        boost::json::object to_json() const override {                                             \
            boost::json::object res;                                                               \
            res["name"] = STRING(structname##EvalGrad);                                            \
            to_json_exp return res;                                                                \
        };                                                                                         \
        datamember structname##EvalGrad(constructorarg) constructorinit{};                         \
        void eval(TMap<T> &dest, const std::vector<TMap<T>> &inputs) override {                    \
            const auto &A = inputs[0];                                                             \
            evalop                                                                                 \
        };                                                                                         \
        std::vector<TMat<T>> grad(const Var<T> &current) override {               \
            const auto &A = current.input_node(0).val();                                         \
            const auto &G = current.grad();                                                       \
            const auto &Ga = current.grad().array();                                              \
            gradop                                                                                 \
        };                                                                                         \
    };                                                                                             \
    template <typename T>                                                                          \
    Var<T> functionname(const Var<T> &operand funargs) {         \
        assertstmt std::vector<Var<T>> input_nodes{operand};                      \
        return {std::make_shared<VarImpl<T>>(rows, cols, (operand.requires_grad()), input_nodes,       \
                                        std::make_unique<structname##EvalGrad<T>>(funVarImpl))};        \
    };
UNARYFUNCTION(exp, Exp, operand.rows(), operand.cols(), UNWRAP(dest = A.array().exp();),
              UNWRAP(return {Ga * current.val().array()};), , , , , , , , , )
UNARYFUNCTION(ln, Ln, operand.rows(), operand.cols(), UNWRAP(dest = A.array().log();),
              UNWRAP(return {Ga / A.array()};), , , , , , , , , )
UNARYFUNCTION(sin, Sin, operand.rows(), operand.cols(), UNWRAP(dest = A.array().sin();),
              UNWRAP(return {Ga * A.array().cos()};), , , , , , , , , )
UNARYFUNCTION(cos, Cos, operand.rows(), operand.cols(), UNWRAP(dest = A.array().cos();),
              UNWRAP(return {-Ga * A.array().sin()};), , , , , , , , , )
UNARYFUNCTION(tan, Tan, operand.rows(), operand.cols(), UNWRAP(dest = A.array().tan();),
              UNWRAP(return {Ga / A.array().cos().pow(2)};), , , , , , , , , )
UNARYFUNCTION(sinh, Sinh, operand.rows(), operand.cols(), UNWRAP(dest = A.array().sinh();),
              UNWRAP(return {Ga * A.array().cosh()};), , , , , , , , , )
UNARYFUNCTION(cosh, Cosh, operand.rows(), operand.cols(), UNWRAP(dest = A.array().cosh();),
              UNWRAP(return {Ga * A.array().sinh()};), , , , , , , , , )
UNARYFUNCTION(tanh, Tanh, operand.rows(), operand.cols(), UNWRAP(dest = A.array().tanh();),
              UNWRAP(return {Ga / A.array().cosh().pow(2)};), , , , , , , , , )
UNARYFUNCTION(sigmoid, Sigmoid, operand.rows(), operand.cols(),
              UNWRAP(dest = 1.0 / (1 + (-A.array()).exp());),
              UNWRAP(auto maexp = (-A.array()).exp(); return {Ga * maexp / (1 + maexp).pow(2)};), ,
              , , , , , , , )
UNARYFUNCTION(relu, ReLU, operand.rows(), operand.cols(),
              UNWRAP(dest = A.unaryExpr([](T x) -> T { return x > 0 ? x : 0; });),
              UNWRAP(return {Ga *
                             A.unaryExpr([](T x) -> T { return x > 0 ? 1 : 0; }).eval().array()};),
              , , , , , , , , )
UNARYFUNCTION(sqrt, Sqrt, operand.rows(), operand.cols(), UNWRAP(dest = A.array().sqrt();),
              UNWRAP(return {0.5 * Ga * current.val().array() / A.array()};), , , , , , , , , )
UNARYFUNCTION(pow, Pow, operand.rows(), operand.cols(), UNWRAP(dest = A.array().pow(n);),
              UNWRAP(return {Ga * n * current.val().array() / A.array()};), UNWRAP(int n;),
              UNWRAP(, int n), n, , , , int n,
              UNWRAP(
                  : n(n)),
              UNWRAP(res["n"] = n;))

// Matrix unary function
UNARYFUNCTION(transpose, Transpose, operand.cols(), operand.rows(), UNWRAP(dest = A.transpose();),
              UNWRAP(return {G.transpose()};), , , , , , , , , )
// Self multiplication: A*A^T
UNARYFUNCTION(lsdot, LSdot, operand.rows(), operand.rows(), UNWRAP(dest = A * A.transpose();),
              UNWRAP(return {2 * G * A};), , , , , , , , , )
// Self multiplication: A^T*A
UNARYFUNCTION(rsdot, RSdot, operand.cols(), operand.cols(), UNWRAP(dest = A.transpose() * A;),
              UNWRAP(return {2 * A * G};), , , , , , , , , )
// det
UNARYFUNCTION(det, Det, 1, 1, UNWRAP(dest.coeffRef(0, 0) = A.determinant();),
              UNWRAP(return {G.coeff(0, 0) * current.val().coeff(0, 0) *
                             A.inverse().transpose()};),
              , , , , ,
              UNWRAP(assert((operand.rows() == operand.cols()) &&
                            "Input should be a square matrix for det.");),
              , , )
// logdet
UNARYFUNCTION(logdet, LogDet, 1, 1, UNWRAP(dest.coeffRef(0, 0) = std::log(T(A.determinant()));),
              UNWRAP(return {G.coeff(0, 0) * A.inverse().transpose()};), , , , , ,
              UNWRAP(assert((operand.rows() == operand.cols()) &&
                            "Input should be a square matrix for logdet.");),
              , , )
// inv
UNARYFUNCTION(inv, Inv, operand.rows(), operand.rows(), UNWRAP(dest = A.inverse();),
              UNWRAP(return {-(current.val() * G * current.val())};), , , , , ,
              UNWRAP(assert((operand.rows() == operand.cols()) &&
                            "Input should be a square matrix for logdet.");),
              , , )
// ivech
static inline auto _s_size(int n) { return (-1 + std::sqrt(1 + 8 * n)) / 2; };
UNARYFUNCTION(ivech, IVech, _s_size(operand.size()), _s_size(operand.size()),
              UNWRAP(
                  int n = dest.rows();
                  auto f = [n](int i, int j) -> int { return (2 * n - i + 1) * i / 2 + j - i; };
                  for (int i = 0; i < n; ++i) {
                      dest.coeffRef(i, i) = *(inputs[0].data() + f(i, i));
                      for (int j = i + 1; j < n; ++j) {
                          dest.coeffRef(i, j) = *(inputs[0].data() + f(i, j));
                          dest.coeffRef(j, i) = dest.coeffRef(i, j);
                      }
                  }),
              UNWRAP(
                  TMat<T> res = A; int n = G.rows();
                  auto f = [n](int i, int j) -> int { return (2 * n - i + 1) * i / 2 + j - i; };
                  for (int i = 0; i < n; ++i) {
                      *(res.data() + f(i, i)) = G.coeff(i, i);
                      for (int j = i + 1; j < n; ++j) {
                          *(res.data() + f(i, j)) = G.coeff(i, j) + G.coeff(j, i);
                      }
                  } return {res};),
              , , , , ,
              UNWRAP(assert((((operand.rows() == 1) || (operand.cols() == 1)) &&
                             _s_size(operand.size()) == std::trunc(_s_size(operand.size()))) &&
                            "Input should be a vector, and the size must be appropriate for lower "
                            "part of a square matrix.");),
              , , )
UNARYFUNCTION(ivecl, IVecl, _s_size(operand.size()), _s_size(operand.size()),
              UNWRAP(
                  int n = dest.rows();
                  auto f = [n](int i, int j) -> int { return (2 * n - i + 1) * i / 2 + j - i; };
                  for (int i = 0; i < n; ++i) {
                      dest.coeffRef(i, i) = *(inputs[0].data() + f(i, i));
                      for (int j = i + 1; j < n; ++j) {
                          // dest.coeffRef(i, j) = 0;
                          dest.coeffRef(j, i) = *(inputs[0].data() + f(i, j));
                      }
                  }),
              UNWRAP(
                  TMat<T> res(A.rows(), A.cols()); res.setZero(); int n = G.rows();
                  auto f = [n](int i, int j) -> int { return (2 * n - i + 1) * i / 2 + j - i; };
                  for (int i = 0; i < n; ++i) {
                      *(res.data() + f(i, i)) = G.coeff(i, i);
                      for (int j = i + 1; j < n; ++j) {
                          *(res.data() + f(i, j)) = G.coeff(j, i);
                      }
                  } return {res};),
              , , , , ,
              UNWRAP(assert((((operand.rows() == 1) || (operand.cols() == 1)) &&
                             _s_size(operand.size()) == std::trunc(_s_size(operand.size()))) &&
                            "Input should be a vector, and the size must be appropriate for lower "
                            "part of a square matrix.");),
              , , )
UNARYFUNCTION(ivecu, IVecu, _s_size(operand.size()), _s_size(operand.size()),
              UNWRAP(
                  int n = dest.rows();
                  auto f = [n](int i, int j) -> int { return (2 * n - i + 1) * i / 2 + j - i; };
                  for (int i = 0; i < n; ++i) {
                      dest.coeffRef(i, i) = *(inputs[0].data() + f(i, i));
                      for (int j = i + 1; j < n; ++j) {
                          dest.coeffRef(i, j) = *(inputs[0].data() + f(i, j));
                          dest.coeffRef(j, i) = 0;
                      }
                  }),
              UNWRAP(
                  TMat<T> res(A.rows(), A.cols()); res.setZero(); int n = G.rows();
                  auto f = [n](int i, int j) -> int { return (2 * n - i + 1) * i / 2 + j - i; };
                  for (int i = 0; i < n; ++i) {
                      *(res.data() + f(i, i)) = G.coeff(i, i);
                      for (int j = i + 1; j < n; ++j) {
                          *(res.data() + f(i, j)) = G.coeff(i, j);
                      }
                  } return {res};),
              , , , , ,
              UNWRAP(assert((((operand.rows() == 1) || (operand.cols() == 1)) &&
                             _s_size(operand.size()) == std::trunc(_s_size(operand.size()))) &&
                            "Input should be a vector, and the size must be appropriate for lower "
                            "part of a square matrix.");),
              , , )

//-----------------------------------------------------------------------------
// Cat
template <typename T = double> struct CatEvalGrad : EvalGradFunctionBase<T> {
    int D = 0, n;
    std::vector<std::pair<int, int>> pos;
    CatEvalGrad(int D) : D(D){};
    std::string get_name() const override { return "CatEvalGrad"; };
    boost::json::object to_json() const override {
        boost::json::object res;
        res["name"] = "CatEvalGrad";
        res["D"] = D;
        return res;
    };
    void eval(TMap<T> &dest, const std::vector<TMap<T>> &inputs) override {
        n = inputs.size();
        pos.resize(n);
        if (D == 0) {
            int k = 0;
            for (int i = 0; i < n; ++i) {
                auto &mat = inputs[i];
                pos[i] = std::pair<int, int>(k, mat.rows());
                dest(Eigen::seqN(k, mat.rows()), Eigen::indexing::all) = mat;
                k += mat.rows();
            }
        } else if (D == 1) {
            int k = 0;
            for (int i = 0; i < n; ++i) {
                auto &mat = inputs[i];
                pos[i] = std::pair<int, int>(k, mat.cols());
                dest(Eigen::indexing::all, Eigen::seqN(k, mat.cols())) = mat;
                k += mat.cols();
            }
        } else {
            throw std::invalid_argument("Invalid dimension for CatEvalGrad. Allowed values are "
                                        "0(row), 1(col). But given value is " +
                                        std::to_string(D));
        }
    };
    std::vector<TMat<T>> grad(const Var<T> &current) {
        const auto &G = current.grad();
        std::vector<TMat<T>> res(n);

        if (D == 0) {
            for (int i = 0; i < n; ++i) {
                auto &p = pos[i];
                res[i] = G(Eigen::seqN(p.first, p.second), Eigen::indexing::all);
            }
        } else if (D == 1) {
            for (int i = 0; i < n; ++i) {
                auto &p = pos[i];
                res[i] = G(Eigen::indexing::all, Eigen::seqN(p.first, p.second));
            }
        }

        return res;
    };
};

// D=0: row, D=1: col.
template <int D = 0, typename T = double>
Var<T> cat(const std::vector<Var<T>> &x) {

    int N = x.size();
    // requires_grad
    bool requires_grad = false;
    for (auto &node : x) {
        requires_grad = requires_grad || node.requires_grad();
    }

    int rows, cols;
    // check size
    if constexpr (D == 0) {
        rows = 0;
        cols = x[0].cols();
        for (int i = 0; i < N; ++i) {
            if (x[i].cols() != cols)
                throw std::runtime_error("Columns of each element should be equal in cat<0>(x).");
            else {
                rows += x[i].rows();
            }
        }
    } else if constexpr (D == 1) {
        rows = x[0].rows();
        cols = 0;
        for (int i = 0; i < N; ++i) {
            if (x[i].rows() != rows)
                throw std::runtime_error("Rows of each element should be equal in cat<1>(x).");
            else {
                cols += x[i].cols();
            }
        }
    } else {
        throw std::invalid_argument("Invalid dimension for cat<dimension>(x). Allowed values are "
                                    "0(row), 1(col). But given value is " +
                                    std::to_string(D));
    }

    return std::make_shared<VarImpl<T>>(rows, cols, requires_grad, x,
                                    std::make_unique<CatEvalGrad<T>>(D));
};

template <int D = 0, typename T = double>
Var<T> cat(std::initializer_list<Var<T>> &x) {
    std::vector<Var<T>> xv = x;
    return cat(xv);
};
// norm. D: dim, 0(all), 1(row, norm of each row, get a row vector ), 2(col)
template <typename T = double> struct LpNormEvalGrad : EvalGradFunctionBase<T> {
    int n; // Power
    int D; // MatReduction
    TArr<T> vabs, vabsn;
    LpNormEvalGrad(int n, int D) : n(n), D(D){};
    std::string get_name() const override { return "LpNormEvalGrad"; };
    boost::json::object to_json() const override {
        boost::json::object res;
        res["name"] = "LpNormEvalGrad";
        res["n"] = n;
        res["D"] = D;
        return res;
    };
    void eval(TMap<T> &dest, const std::vector<TMap<T>> &inputs) override {
        const auto &X = inputs[0];
        vabs = X.array().abs();
        if (n < 50) [[likely]] {
            vabsn = vabs.pow(n);
            if (D == 0) {
                // Row
                dest = vabsn.rowwise().sum().pow(1.0 / n);
            } else if (D == 1) {
                dest = vabsn.colwise().sum().pow(1.0 / n);
                // Col.
            } else {
                // All
                dest.coeffRef(0, 0) = std::pow(vabsn.sum(), 1.0 / n);
            }
        } else {

            // Infinity norm
            if (D == 0) {
                // Row
                dest = vabs.rowwise().maxCoeff();
            } else if (D == 1) {
                dest = vabs.rowwise().maxCoeff();
                // Col.
            } else {
                // All
                dest.coeffRef(0, 0) = vabs.maxCoeff();
            }
        }
    };
    std::vector<TMat<T>> grad(const Var<T> &current) {
        const auto &arr = current.input_node(0).val().array();
        const auto &G = current.grad();
        if (D == 0) {
            if (n == 1) {
                return {_broadcasting_mul(G, arr.unaryExpr(&sign<T>))};
            } else if (n < 50) {
                TMat<T> res(arr.rows(), arr.cols());
                for (int i = 0; i < arr.rows(); ++i) {
                    T tmp = current.val().coeff(i, 0) / vabsn.row(i).sum();
                    res.row(i) = G.coeff(i, 0) * tmp * vabsn.row(i) / vabs.row(i) *
                                 arr.row(i).unaryExpr(&sign<T>);
                }
                return {res};
            } else {
                TMat<T> res(arr.rows(), arr.cols());
                res.setZero();
                for (int i = 0; i < arr.rows(); ++i) {
                    int k;
                    vabs.row(i).maxCoeff(&k);
                    res.coeffRef(i, k) = G.coeff(i, 0) * sign(arr.coeff(i, k));
                }
                return {res};
            };
        } else if (D == 1) {
            if (n == 1) {
                return {_broadcasting_mul(G, arr.unaryExpr(&sign<T>))};
            } else if (n < 50) {
                TMat<T> res(arr.rows(), arr.cols());
                for (int j = 0; j < arr.cols(); ++j) {
                    T tmp = current.val().coeff(0, j) / vabsn.col(j).sum();
                    res.col(j) = G.coeff(0, j) * tmp * vabsn.col(j) / vabs.col(j) *
                                 arr.col(j).unaryExpr(&sign<T>);
                }
                return {res};
            } else {
                TMat<T> res(arr.rows(), arr.cols());
                res.setZero();
                for (int j = 0; j < arr.cols(); ++j) {
                    int k;
                    vabs.col(j).maxCoeff(&k);
                    res.coeffRef(k, j) = G.coeff(0, j) * sign(arr.coeff(k, j));
                }
                return {res};
            };
        } else {
            // Reduction all.
            if (n == 1) {
                return {G.coeff(0, 0) * arr.unaryExpr(&sign<T>)};
            } else if (n < 50) {
                TArr<T> res = G.coeff(0, 0) * vabsn / vabs * arr.unaryExpr(&sign<T>) *
                              (current.val().coeff(0, 0) / vabsn.sum());
                return {res};
            } else {
                TMat<T> res(arr.rows(), arr.cols());
                res.setZero();
                int i, j;
                vabs.maxCoeff(&i, &j);
                res.coeffRef(i, j) = G.coeff(0, 0) * sign(arr.coeff(i, j));
                return {res};
            };
        }
    };
};

// D=2 for all.
template <int n = 2, int D = -1, typename T = double> requires requires() { n > 0; }
Var<T> lpnorm(const Var<T> &operand) {
    std::vector<Var<T>> input_nodes{operand};
    if constexpr (D == 0) {
        return std::make_shared<VarImpl<T>>(operand.rows(), 1, (operand.requires_grad()), input_nodes,
                                        std::make_unique<LpNormEvalGrad<T>>(n, D));
    } else if constexpr (D == 1) {
        return std::make_shared<VarImpl<T>>(1, operand.cols(), (operand.requires_grad()), input_nodes,
                                        std::make_unique<LpNormEvalGrad<T>>(n, D));
    } else {
        return std::make_shared<VarImpl<T>>(1, 1, (operand.requires_grad()), input_nodes,
                                        std::make_unique<LpNormEvalGrad<T>>(n, D));
    };
};

// trace
template <typename T = double> struct TraceEvalGrad : EvalGradFunctionBase<T> {
    std::string get_name() const override { return "TraceEvalGrad"; };
    boost::json::object to_json() const override {
        boost::json::object res;
        res["name"] = "TraceEvalGrad";
        return res;
    };
    void eval(TMap<T> &dest, const std::vector<TMap<T>> &inputs) override {
        const auto &X = inputs[0];
        dest.coeffRef(0, 0) = X.trace();
    };
    std::vector<TMat<T>> grad(const Var<T> &current) {
        const auto &X = current.input_node(0).val();
        const auto &G = current.grad();
        return {G.coeff(0, 0) * TMat<T>::Identity(X.rows(), X.cols())};
    };
};
template <typename T = double> Var<T> trace(const Var<T> &X) {
    assert((X.cols() == X.rows()) && "A must be a square matrix for trace(X) operator.");
    std::vector<Var<T>> input_nodes{X};
    return std::make_shared<VarImpl<T>>(1, 1, (X.requires_grad()), input_nodes,
                                    std::make_unique<TraceEvalGrad<T>>());
};

template <typename T = double> struct Trace2EvalGrad : EvalGradFunctionBase<T> {
    std::string get_name() const override { return "Trace2EvalGrad"; };
    boost::json::object to_json() const override {
        boost::json::object res;
        res["name"] = "Trace2EvalGrad";
        return res;
    };
    void eval(TMap<T> &dest, const std::vector<TMap<T>> &inputs) override {
        const auto &X = inputs[0];
        const auto &Y = inputs[1];
        dest.coeffRef(0, 0) = 0;
        for (int i = 0; i < X.rows(); ++i) {
            dest.coeffRef(0, 0) += X.row(i).dot(Y.col(i));
        }
    };
    std::vector<TMat<T>> grad(const Var<T> &current) {
        const auto &X = current.input_node(0).val();
        const auto &Y = current.input_node(1).val();
        const auto &G = current.grad();
        return {G.coeff(0, 0) * Y.transpose(), G.coeff(0, 0) * X.transpose()};
    };
};
template <typename T = double>
Var<T> trace(const Var<T> &X, const Var<T> &Y) {
    assert(((X.cols() == Y.rows()) && (X.rows() == Y.cols())) &&
           "X*Y must be a valid matrix product for trace(X, Y) operator.");
    std::vector<Var<T>> input_nodes{X, Y};
    return std::make_shared<VarImpl<T>>(1, 1, (X.requires_grad() || Y.requires_grad()), input_nodes,
                                    std::make_unique<Trace2EvalGrad<T>>());
};

// diag. Always return a column vector.
template <typename T = double> struct DiagEvalGrad : EvalGradFunctionBase<T> {
    std::string get_name() const override { return "DiagEvalGrad"; };
    boost::json::object to_json() const override {
        boost::json::object res;
        res["name"] = "DiagEvalGrad";
        return res;
    };
    void eval(TMap<T> &dest, const std::vector<TMap<T>> &inputs) override {
        const auto &X = inputs[0];
        dest = X.diagonal();
    };
    std::vector<TMat<T>> grad(const Var<T> &current) {
        const auto &X = current.input_node(0).val();
        const auto &G = current.grad();
        return {G.array() * TMat<T>::Identity(X.rows(), X.cols()).array()};
    };
};
template <typename T = double> Var<T> diag(const Var<T> &X) {
    assert((X.cols() == X.rows()) && "X must be a square matrix for trace(X) operator.");
    std::vector<Var<T>> input_nodes{X};
    return std::make_shared<VarImpl<T>>(X.rows(), 1, (X.requires_grad()), input_nodes,
                                    std::make_unique<DiagEvalGrad<T>>());
};
// diag(A, B): diag(A*B)
template <typename T = double> struct Diag2EvalGrad : EvalGradFunctionBase<T> {
    std::string get_name() const override { return "Diag2EvalGrad"; };
    boost::json::object to_json() const override {
        boost::json::object res;
        res["name"] = "Diag2EvalGrad";
        return res;
    };
    void eval(TMap<T> &dest, const std::vector<TMap<T>> &inputs) override {
        const auto &X = inputs[0];
        const auto &Y = inputs[1];
        for (int i = 0; i < X.rows(); ++i) {
            dest.coeffRef(i, 0) = X.row(i).dot(Y.col(i));
        }
    };
    std::vector<TMat<T>> grad(const Var<T> &current) {
        const auto &X = current.input_node(0).val();
        const auto &Y = current.input_node(1).val();
        const auto &G = current.grad();
        return {G.asDiagonal() * Y.transpose(), X.transpose() * G.asDiagonal()};
    };
};
template <typename T = double>
Var<T> diag(const Var<T> &X, const Var<T> &Y) {
    assert(((X.cols() == Y.rows()) && (X.rows() == Y.cols())) &&
           "X*Y must be a valid matrix product for diag(X, Y) operator.");
    std::vector<Var<T>> input_nodes{X, Y};
    return std::make_shared<VarImpl<T>>(X.rows(), 1, (X.requires_grad() || Y.requires_grad()),
                                    input_nodes, std::make_unique<Diag2EvalGrad<T>>());
};

template <typename T = double> struct SumEvalGrad : EvalGradFunctionBase<T> {
    int D; // MatReduction
    SumEvalGrad(int D) : D(D){};
    std::string get_name() const override { return "SumEvalGrad"; };
    boost::json::object to_json() const override {
        boost::json::object res;
        res["name"] = "SumEvalGrad";
        res["D"] = D;
        return res;
    };
    void eval(TMap<T> &dest, const std::vector<TMap<T>> &inputs) override {
        const auto &X = inputs[0];
        if (D == -1) {
            dest.coeffRef(0, 0) = X.sum();
        } else if (D == 0) {
            dest = X.rowwise().sum();
        } else if (D == 1) {
            dest = X.colwise().sum();
        } else {
            throw std::range_error("Invalid dimension " + std::to_string(D) +
                                   ". Allowed values are -1(all), 0(row-wise sum), 1(col).");
        }
    };
    std::vector<TMat<T>> grad(const Var<T> &current) {
        const auto &arr = current.input_node(0).val().array();
        const auto &G = current.grad();
        if (D == -1) {
            return {G.coeff(0, 0) * TMat<T>::Ones(arr.rows(), arr.cols())};
        } else if (D == 0) {
            return {_broadcasting_mul(G, TMat<T>::Ones(arr.rows(), arr.cols()))};
        } else if (D == 1) {
            return {_broadcasting_mul(G, TMat<T>::Ones(arr.rows(), arr.cols()))};
        } else {
            throw std::range_error("Invalid dimension " + std::to_string(D) +
                                   ". Allowed values are -1(all), 0(row-wise sum), 1(col).");
        }
    };
};

// D=-1: all , D=0: rowwise .
template <int D = -1, typename T = double>
Var<T> sum(const Var<T> &operand) {
    std::vector<Var<T>> input_nodes{operand};
    if constexpr (D == 0) {
        return std::make_shared<VarImpl<T>>(operand.rows(), 1, (operand.requires_grad()), input_nodes,
                                        std::make_unique<SumEvalGrad<T>>(D));
    } else if constexpr (D == 1) {
        return std::make_shared<VarImpl<T>>(1, operand.cols(), (operand.requires_grad()), input_nodes,
                                        std::make_unique<SumEvalGrad<T>>(D));
    } else {
        return std::make_shared<VarImpl<T>>(1, 1, (operand.requires_grad()), input_nodes,
                                        std::make_unique<SumEvalGrad<T>>(D));
    };
};

//-----------------------------------------------------------------------------
template <typename T = double> struct WeightedSumVVEvalGrad : EvalGradFunctionBase<T> {
    int n = 0;
    std::string get_name() const override { return "WeightedSumVVEvalGrad"; };
    boost::json::object to_json() const override {
        boost::json::object res;
        res["name"] = "WeightedSumVVEvalGrad";
        return res;
    };
    void eval(TMap<T> &dest, const std::vector<TMap<T>> &inputs) override {
        n = inputs.size() / 2;
        dest.setZero();
        for (int i = 0; i < n; ++i) {
            dest += inputs[i * 2] * inputs[i * 2 + 1].coeff(0, 0);
        };
    };
    std::vector<TMat<T>> grad(const Var<T> &current) {
        const auto &arr = current.input_node(0).val().array();
        const auto &G = current.grad();
        std::vector<TMat<T>> res;
        for (int i = 0; i < n; ++i) {
            res.emplace_back(G * current.input_node(2 * i + 1).val().coeff(0, 0));
            res.emplace_back(
                eigen_scalar_mat(T((G.array() * current.input_node(2 * i).val().array()).sum())));
        }
        return res;
    };
};

template <int D = -1, typename T = double>
Var<T> sum(const std::vector<Var<T>> &x,
                            const std::vector<Var<T>> &weights) {
    std::vector<Var<T>> input_nodes;
    // Check size of x and weights to be equal.
    if (x.size() != weights.size()) {
        throw(std::runtime_error(
            "Size of x and weights must be equal in sum(x, weights). However they are " +
            std::to_string(x.size()) + " and " + std::to_string(weights.size()) + "."));
    }
    // Add nodes.
    for (int i = 0; i < x.size(); ++i) {
        input_nodes.emplace_back(x[i]);
        input_nodes.emplace_back(weights[i]);
    }
    // Check equal size of elements of x.
    int rows = x[0].rows(), cols = x[0].cols();
    for (int i = 1; i < x.size(); ++i) {
        auto node = x[i];
        if ((node.rows() != rows) || (node.cols() != cols)) {
            throw(std::runtime_error(
                "Element size of x in sum(x, weights) must be equal. However size of element " +
                std::to_string(i) + " and previous element are " +
                _shape_str(node.rows(), node.cols(), x[i - 1].rows(), x[i - 1].cols()) + "."));
        }
    }
    // Check weights to be scalar.
    for (int i = 0; i < weights.size(); ++i) {
        auto node = weights[i];
        if (node.size() != 1) {
            throw(std::runtime_error(
                "Element size of weights in sum(x, weights) must be 1. However it's " +
                std::to_string(node.size()) + " for element " + std::to_string(i) + "."));
        }
    }
    // requires_grad
    bool requires_grad = false;
    for (auto &node : x) {
        requires_grad = requires_grad || node.requires_grad();
    }
    return std::make_shared<VarImpl<T>>(rows, cols, requires_grad, input_nodes,
                                    std::make_unique<WeightedSumVVEvalGrad<T>>());
};
//-----------------------------------------------------------------------------
template <typename T = double> struct MeanEvalGrad : EvalGradFunctionBase<T> {
    int D; // MatReduction
    MeanEvalGrad(int D) : D(D){};
    std::string get_name() const override { return "MeanEvalGrad"; };
    boost::json::object to_json() const override {
        boost::json::object res;
        res["name"] = "MeanEvalGrad";
        res["D"] = D;
        return res;
    };
    void eval(TMap<T> &dest, const std::vector<TMap<T>> &inputs) override {
        const auto &X = inputs[0];
        if (D == -1) {
            dest.coeffRef(0, 0) = X.mean();
        } else if (D == 0) {
            dest = X.rowwise().mean();
        } else if (D == 1) {
            dest = X.colwise().mean();
        } else {
            throw std::range_error("Invalid dimension " + std::to_string(D) +
                                   ". Allowed values are -1(all), 0(row-wise sum), 1(col).");
        }
    };
    std::vector<TMat<T>> grad(const Var<T> &current) {
        const auto &arr = current.input_node(0).val().array();
        const auto &G = current.grad();
        if (D == -1) {
            return {G.coeff(0, 0) * TMat<T>::Constant(arr.rows(), arr.cols(), 1.0 / arr.size())};
        } else if (D == 0) {
            return {
                _broadcasting_mul(G, TMat<T>::Constant(arr.rows(), arr.cols(), 1.0 / arr.cols()))};
        } else if (D == 1) {
            return {
                _broadcasting_mul(G, TMat<T>::Constant(arr.rows(), arr.cols(), 1.0 / arr.rows()))};
        } else {
            throw std::range_error("Invalid dimension " + std::to_string(D) +
                                   ". Allowed values are -1(all), 0(row-wise sum), 1(col).");
        }
    };
};

// D=-1: all , D=0: rowwise .
template <int D = -1, typename T = double>
Var<T> mean(const Var<T> &operand) {
    std::vector<Var<T>> input_nodes{operand};
    if constexpr (D == 0) {
        return std::make_shared<VarImpl<T>>(operand.rows(), 1, (operand.requires_grad()), input_nodes,
                                        std::make_unique<MeanEvalGrad<T>>(D));
    } else if constexpr (D == 1) {
        return std::make_shared<VarImpl<T>>(1, operand.cols(), (operand.requires_grad()), input_nodes,
                                        std::make_unique<MeanEvalGrad<T>>(D));
    } else {
        return std::make_shared<VarImpl<T>>(1, 1, (operand.requires_grad()), input_nodes,
                                        std::make_unique<MeanEvalGrad<T>>(D));
    };
};

template <typename T = double> struct VarianceEvalGrad : EvalGradFunctionBase<T> {
    int D; // Mat Dim
    TMat<T> x_d;
    VarianceEvalGrad(int D) : D(D){};
    std::string get_name() const override { return "VarianceEvalGrad"; };
    boost::json::object to_json() const override {
        boost::json::object res;
        res["name"] = "VarianceEvalGrad";
        res["D"] = D;
        return res;
    };
    void eval(TMap<T> &dest, const std::vector<TMap<T>> &inputs) override {
        const auto &X = inputs[0];
        if (D == -1) {
            x_d = X.array() - X.mean();
            dest.coeffRef(0, 0) = x_d.array().pow(2).mean();
        } else if (D == 0) {
            x_d = X.colwise() - X.rowwise().mean();
            dest = x_d.array().pow(2).rowwise().mean();
        } else if (D == 1) {
            x_d = X.rowwise() - X.colwise().mean();
            dest = x_d.array().pow(2).colwise().mean();
        } else {
            throw std::range_error("Invalid dimension " + std::to_string(D) +
                                   ". Allowed values are -1(all), 0(row-wise sum), 1(col).");
        }
    };
    std::vector<TMat<T>> grad(const Var<T> &current) {
        const auto &arr = current.input_node(0).val().array();
        const auto &G = current.grad();

        if (D == -1) {
            return {G.coeff(0, 0) * 2.0 / arr.size() * x_d};
        } else if (D == 0) {
            return {G.coeff(0, 0) * 2.0 / arr.cols() * x_d};
        } else if (D == 1) {
            return {G.coeff(0, 0) * 2.0 / arr.rows() * x_d};
        } else {
            throw std::range_error("Invalid dimension " + std::to_string(D) +
                                   ". Allowed values are -1(all), 0(row-wise sum), 1(col).");
        }
    };
};

// D=-1: all , D=0: rowwise .
template <int D = -1, typename T = double>
Var<T> variance(const Var<T> &operand) {
    std::vector<Var<T>> input_nodes{operand};
    if constexpr (D == 0) {
        return std::make_shared<VarImpl<T>>(operand.rows(), 1, (operand.requires_grad()), input_nodes,
                                        std::make_unique<VarianceEvalGrad<T>>(D));
    } else if constexpr (D == 1) {
        return std::make_shared<VarImpl<T>>(1, operand.cols(), (operand.requires_grad()), input_nodes,
                                        std::make_unique<VarianceEvalGrad<T>>(D));
    } else {
        return std::make_shared<VarImpl<T>>(1, 1, (operand.requires_grad()), input_nodes,
                                        std::make_unique<VarianceEvalGrad<T>>(D));
    };
};

template <typename T = double> struct LinearEvalGrad : EvalGradFunctionBase<T> {
    int type; // 0: b is a scalar, 1: b is a row vector, 2: b is a column vector.
    std::string get_name() const override { return "LinearEvalGrad"; };
    boost::json::object to_json() const override {
        boost::json::object res;
        res["name"] = "LinearEvalGrad";
        return res;
    };
    void eval(TMap<T> &dest, const std::vector<TMap<T>> &inputs) override {
        const auto &A = inputs[0];
        const auto &x = inputs[1];
        const auto &b = inputs[2];
        if (b.size() == 1) [[unlikely]] {
            type = 0;
            dest = (A * x).array() + b.coeff(0, 0);
        } else if (b.rows() == 1) {
            type = 1;
            dest = A * x;
            for (auto row : dest.rowwise()) {
                row += b;
            }
        } else if (b.cols() == 1) {
            type = 2;
            dest = A * x;
            for (auto col : dest.colwise()) {
                col += b;
            }
        } else {
            throw std::range_error("b is not a scalar or a vector for linear(A,x,b).");
        }
    };
    std::vector<TMat<T>> grad(const Var<T> &current) {
        const auto &A = current.input_node(0).val();
        const auto &x = current.input_node(1).val();
        const auto &b = current.input_node(2).val();
        const auto &G = current.grad();
        std::vector<TMat<T>> res(3);
        if (current.input_node(0).requires_grad()) {
            res[0] = G * x.transpose();
        } else {
            res[0] = TMat<T>();
        };
        if (current.input_node(1).requires_grad()) {
            res[1] = A.transpose() * G;
        } else {
            res[1] = TMat<T>();
        };
        if (current.input_node(2).requires_grad()) {
            if (type == 0) {
                res[2] = eigen_scalar_mat(T(G.sum()));
            } else if (type == 1) {
                res[2] = G.colwise().sum();
            } else { // type==2
                res[2] = G.rowwise().sum();
            }
        } else {
            res[2] = TMat<T>();
        };
        return res;
    };
};

// A*x+b
template <int D = -1, typename T = double>
Var<T> linear(const Var<T> &A, const Var<T> &x,
                               const Var<T> &b) {
    assert(((A.cols() == x.rows()) &&
            ((b.size() == 1) || ((b.rows() == 1) && (b.cols() == x.cols())) ||
             ((b.cols() == 1) && (A.rows() == b.rows())))) &&
           "A*x must be a valid matrix product for linear(). And b should be a scalar or vector.");
    std::vector<Var<T>> input_nodes{A, x, b};
    return std::make_shared<VarImpl<T>>(
        A.rows(), x.cols(), (A.requires_grad() || x.requires_grad() || b.requires_grad()),
        input_nodes, std::make_unique<LinearEvalGrad<T>>());
};

template <typename T = double> struct SoftmaxVEvalGrad : EvalGradFunctionBase<T> {
    int k;
    std::string get_name() const override { return "SumEvalGrad"; };
    boost::json::object to_json() const override {
        boost::json::object res;
        res["name"] = "SoftmaxVEvalGrad";
        return res;
    };
    void eval(TMap<T> &dest, const std::vector<TMap<T>> &inputs) override {
        k = inputs.size();
        for (int i = 0; i < k; ++i) {
            dest.coeffRef(i, 0) = inputs[i].coeff(0, 0);
        };
        dest = dest.array().exp();
        dest = dest / dest.sum();
    };
    std::vector<TMat<T>> grad(const Var<T> &current) {
        std::vector<TMat<T>> res(k);
        T s = (current.val().array() * current.grad().array()).sum();
        for (int i = 0; i < k; ++i) {
            T tmp = current.val().coeff(i, 0) * (current.grad().coeff(i, 0) - s);
            res[i] = eigen_scalar_mat(tmp);
        }
        return res;
    };
};

template <int D = -1, typename T = double>
Var<T> softmax(const std::vector<Var<T>> &input_nodes) {
    bool requires_grad = false;
    for (int i = 0; i < input_nodes.size(); ++i) {
        assert((input_nodes[i].size() == 1) && "Size of each node in softmax() should be 1.");
        requires_grad = (requires_grad || input_nodes[i].requires_grad());
    };
    return std::make_shared<VarImpl<T>>(input_nodes.size(), 1, requires_grad, input_nodes,
                                    std::make_unique<SoftmaxVEvalGrad<T>>());
}

}// namespace DynAutoDiff

#undef UNWRAP
#undef BINARYARITHOP
#undef UNARYFUNCTION
#endif
