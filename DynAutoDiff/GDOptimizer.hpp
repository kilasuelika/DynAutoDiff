#ifndef __DYNAUTODIFF__GDOPTIMIZER__
#define __DYNAUTODIFF__GDOPTIMIZER__

#include "DynAutoDiff.hpp"
#include <stdexcept>

#define CHECK_NULLPTR                                                                              \
    assert((gm != nullptr) && "GraphManager is nullptr. Please set it to use step().");            \
    assert((scheduler != nullptr) && "GraphManager is nullptr. Please set it to use step().");

namespace DynAutoDiff {

template <typename T> struct LRScheduler {
  public:
    virtual void step() = 0;
    virtual T lr() const = 0;
};

template <typename T = double> inline void check_graphmanager(GraphManager<T> *const gm) {
    if (gm->nparm() == 0) {
        throw std::invalid_argument(
            "The parameter number of the GraphManager is 0. Please check the model.");
    };
    if ((gm->nparm() != gm->val().size()) || (gm->nparm() != gm->grad().size())) {
        throw std::invalid_argument(
            "Number of parameter of GraphManager is not same as size of _val and _grad buffer. "
            "Please ensure you have called gm.auto_bind_parm() so that storage of parameters "
            "is contiguous.");
    };
};
template <typename T = double> struct ConstLR : LRScheduler<T> {
    T _lr = 1e-6;
    ConstLR(T lr) : _lr(lr){};
    void step() override{};
    T lr() const override { return _lr; };
};
template <typename T = double> struct StepLR : LRScheduler<T> {
    int k = 0;
    int step_size = 50;
    T gamma = 0.8, _lr = 0.1;
    StepLR(T init_lr = 0.1, int step_size = 50, T gamma = 0.8)
        : k(step_size), _lr(init_lr), step_size(step_size), gamma(gamma){};
    void step() override {
        if (k == 0) {
            _lr *= gamma;
            k = step_size;
        } else {
            --k;
        };
    };
    T lr() const override { return _lr; };
};
template <typename T = double> struct InterleaveLR : LRScheduler<T> {
    int k = 0;
    int step_size = 2000;
    T gamma = 0.8, _lr, _lr_s = 1e-6, _lr_b = 1e-3;
    InterleaveLR(T init_lr = 1e-6, int step_size = 500, T gamma = 0.8)
        : k(step_size), _lr(init_lr), _lr_s(init_lr), step_size(step_size), gamma(gamma){};
    void step() override {
        if (k == -1) {
            _lr_s = std::min(_lr_s / gamma, 1e-5);
            _lr_b = std::min(_lr_b / gamma, 8e-3);
            _lr = _lr_s;
            k = step_size;
        } else if (k > 0) {
            //_lr*=gamma;
            --k;
        } else if (k == 0) {
            _lr = _lr_b;
            --k;
        } else {
        };
    };
    T lr() const override { return _lr; };
};

template <typename T = double> struct NaiveGD {
    GraphManager<T> *gm;
    LRScheduler<T> *scheduler;

    NaiveGD(GraphManager<T> *gm = nullptr, LRScheduler<T> *scheduler = nullptr)
        : gm(gm), scheduler(scheduler) {
        check_graphmanager<T>(gm);
    };

    // Update n steps.
    void step(int n = 1) {
        CHECK_NULLPTR
        for (int i = 0; i < n; ++i) {
            gm->run();
            gm->_val -= scheduler->lr() * gm->_grad;
            scheduler->step();
        };
    };
    void set_scheduler(LRScheduler<T> *scheduler) { this->scheduler = scheduler; }
};

template <typename T = double> struct SGD {
    GraphManager<T> *gm;
    LRScheduler<T> *scheduler;

    SGD(GraphManager<T> *gm = nullptr, LRScheduler<T> *scheduler = nullptr)
        : gm(gm), scheduler(scheduler) {
        check_graphmanager(gm);
    };

    // Update n steps.
    void step(int n = 1) {
        CHECK_NULLPTR
        for (int i = 0; i < n; ++i) {
            gm->run();
            gm->_val -= scheduler->lr() * gm->_grad;
            scheduler->step();
        };
    };
    void set_scheduler(LRScheduler<T> *scheduler) { this->scheduler = scheduler; }
};

template <typename T = double> struct Adam {
    GraphManager<T> *gm;
    LRScheduler<T> *scheduler;

    T beta1, beta2, epsilon = 1e-8;
    TVecA<T> m, v;
    int n = 1;

    Adam(GraphManager<T> *gm = nullptr, LRScheduler<T> *scheduler = nullptr, T beta1 = 0.9,
         T beta2 = 0.999)
        : gm(gm), scheduler(scheduler), beta1(beta1), beta2(beta2) {
        check_graphmanager(gm);
        m.resize(gm->_nparm);
        m.setZero();
        v.resize(gm->_nparm);
        v.setZero();
    };

    // Update n steps.
    void step(int k = 1) {
        for (int i = 0; i < k; ++i) {
            gm->run();
            // std::cout << "val: " << gm->_val.coeff(0) << ", grad: " << gm->_grad.coeff(0);
            // std::cout<<gm->_grad<<std::endl;
            m = beta1 * m + (1 - beta1) * gm->_grad;
            v = beta1 * v + (1 - beta1) * gm->_grad.pow(2);
            // TVecA<T> delta = scheduler->lr() * m / (v.sqrt() + epsilon);
            // std::cout << ", delta: " << delta.coeff(0) << std::endl;
            gm->_val -= scheduler->lr() * (m / (1 - std::pow(beta1, n))) /
                        ((v / (1 - std::pow(beta2, n))).sqrt() + epsilon);

            scheduler->step();
            ++n;
        };
    };
    void set_scheduler(LRScheduler<T> *scheduler) { this->scheduler = scheduler; }
};

}; // namespace DynAutoDiff
#endif