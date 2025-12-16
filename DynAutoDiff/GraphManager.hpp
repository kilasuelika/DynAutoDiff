#ifndef __DYNAUTODIFF_GRAPHMANAGER__
#define __DYNAUTODIFF_GRAPHMANAGER__
#include "Distributions.hpp"
#include "EigenHelper.hpp"
#include "Losses.hpp"
#include "Var.hpp"

#include <functional>
#include <map>
#include <stack>

namespace DynAutoDiff {
// Track all nodes in a graph using DFS.
template <typename T = double> class GraphManager {

    public:
    GraphManager(){};
    GraphManager(Var<T> root) : _root(root) { track_nodes(root); };
    // The first node is implicitly to be root.
    void track_nodes(Var<T> top) {
        std::stack<Var<T>> s;
        s.push(top);
        while (!s.empty()) {
            auto cur = s.top();
            //cur->_visited = true;
            cur.set_visited(true);
            _all_nodes.push_back(cur);
            s.pop();
            for (auto &node : cur.input_nodes()) {
                if (!node.visited()) {
                    s.push(node);
                }
            }
        }
        _reset_visit_status();
        _gen_leaf_tags();

        for (auto &cur : _all_nodes) {
            if (cur.requires_grad()) {
                if (cur.is_leaf())
                    _parm_nodes.push_back(cur);
                else
                    _intermediate_grad_nodes.push_back(cur);
            }
        }

        _update_nparm();
    }
    const auto &intermediate_grad_nodes() const { return _intermediate_grad_nodes; };
    const auto &parm_nodes() const { return _parm_nodes; };
    int nparm() const { return _nparm; };
    const auto &all_nodes() const { return _all_nodes; };
    auto &root() const { return _root; };
    const auto &val() const { return _val; };
    const auto &grad() const { return _grad; };

    int total_nodes() const { return _all_nodes.size(); };
    const TMap<T> &run(bool _zero_grad = true) {
        zero_all(_zero_grad);
        _root.eval();
        _root.backward();
        return _root.val();
    };
    void set_root(std::shared_ptr<Var<T>> root) {
        track_nodes(root);
        this->_root = root;
    };
    // If clear_leaf==false, then only clear gradients of non-leaf nodes(which represents
    // intermediate results).
    void zero_grad() {
        // Always clear intermediate nodes.
        // for (auto &node : _intermediate_grad_nodes)
        //    node->_grad.setZero();
        if (_g_allocated) {
            _grad.setZero();
        } else {
            for (auto &node : _parm_nodes) {
                //node->_grad.setZero();
                node.zero_grad();
            }
        }
    }
    void zero_eval_flag() {
        for (auto &node : _all_nodes) {
            node.set_evaluated(false);
        }
    }
    // Bind data. A Convinent wraper for ceres optimizer.
    void bind(T *val, T *grad) {
        int k = 0;
        for (auto &node : _parm_nodes) {
            node->bind(val + k, grad == nullptr ? nullptr : grad + k, node->rows(), node->cols());
            k += node->size();
        };
    };
    // Allocate val and grad in continguous memory.
    std::pair<TMap<T>, TMap<T>> auto_bind_parm() {
        if (_v_allocated && _g_allocated) {

        } else {
            _val.resize(_nparm);
            _grad.resize(_nparm);
            // Copy data.
            copy_parm_val_to(_val.data());

            bind(_val.data(), _grad.data());
            _v_allocated = true;
            _g_allocated = true;
        };
        return std::make_pair(TMap<T>(_val.data(), _nparm, 1), TMap<T>(_grad.data(), _nparm, 1));
    };
    //
    void zero_all(bool _zero_grad = true) {
        zero_eval_flag();
        if (_zero_grad)
            zero_grad();
    };
    // Write a json file. Root is the first node.
    void save(const std::string &filename) {
        // Add id to nodes.
        _set_id();
        // Generate json.
        boost::json::array res;
        std::cout << _all_nodes.size() << std::endl;
        int i = 0;
        for (const auto &node : _all_nodes) {
            std::cout << "node: " << i++ << std::endl;
            res.emplace_back(_node_to_json(node));
        };
        // Write file.
        std::ofstream file(filename);
        file << res;
    }
    void clear() {
        // _root = nullptr;
        _all_nodes.clear();
        _parm_nodes.clear();
        _intermediate_grad_nodes.clear();
    };
    // Copy values of parm nodes to begin.
    void copy_parm_val_to(T *begin) {
        int k = 0;
        for (int i = 0; i < _parm_nodes.size(); ++i) {
            auto &node = _parm_nodes[i];
            std::copy(node->cvbegin(), node->cvend(), begin + k);
            k += node->size();
        }
    };
    void copy_parm_val_from(const T *begin) {
        if (_v_allocated) {
            // std::cout << "v allocated" << std::endl;
            std::copy(begin, begin + _nparm, _val.data());
        } else {
            int k = 0;
            for (int i = 0; i < _parm_nodes.size(); ++i) {
                auto &node = _parm_nodes[i];
                std::copy(begin + k, begin + k + _nparm, node->vbegin());
                k += node->size();
            }
        }
    }
    void copy_parm_grad_to(T *begin) {
        if (_g_allocated) {
            std::copy(_grad.begin(), _grad.end(), begin);
        } else {
            int k = 0;
            for (int i = 0; i < _parm_nodes.size(); ++i) {
                auto &node = _parm_nodes[i];
                std::copy(node.cgbegin(), node.cgend(), begin + k);
                k += node.size();
            }
        };
    };
    auto load(const std::string &filename) {
        clear();

        std::ifstream file(filename);
        std::string json_string((std::istreambuf_iterator<char>(file)),
                                (std::istreambuf_iterator<char>()));

        boost::json::array node_arr = boost::json::parse(json_string).as_array();
        for (int i = 0; i < node_arr.size(); ++i) {
            auto &j_node = node_arr[i].as_object();
            auto node = _node_from_json(j_node);
            _all_nodes.push_back(node);
            if (node.requires_grad()) {
                _parm_nodes.push_back(node);
            }
        }
        // Create input_nodes;
        for (int i = 0; i < node_arr.size(); ++i) {
            if (!_all_nodes[i].is_leaf()) {
                auto j_arr = node_arr[i].as_object()["input_nodes"].as_array();
                for (int j = 0; j < j_arr.size(); ++j) {
                    // _all_nodes[i]->_input_nodes.push_back(_all_nodes[j_arr[j].as_int64()]);
                    _all_nodes[i].add_input_node(_all_nodes[j_arr[j].as_int64()]);
                }
            }
        }
        _root = _all_nodes[0];
        return _root;
    }

  private:


    template <typename T1> friend struct NaiveGD;
    template <typename T1> friend struct Adam;

    Var<T> _root;
    //_parm_nodes: leaf with gradient.
    std::vector<Var<T>> _all_nodes, _parm_nodes, _intermediate_grad_nodes;
    TVecA<T> _val, _grad;
    bool _v_allocated = false, _g_allocated = false;
    int _nparm = 0;

    void _reset_visit_status() {
        for (auto &node : _all_nodes) {
            node.set_visited(false);
        }
    };
    void _gen_leaf_tags() {
        for (auto &node : _all_nodes) {
            if (node.input_nodes().size() == 0) {
                node.set_leaf(true);
            } else {
                node.set_leaf(false);
            }
        }
    };
    void _set_id() {
        for (int i = 0; i < _all_nodes.size(); ++i) {
            _all_nodes[i].set_id(i);
        };
    }
    auto _node_to_json(const Var<T> &node) {
        boost::json::object obj;
        obj["id"] = node.id();
        obj["rows"] = node.rows();
        obj["cols"] = node.cols();
        obj["requires_grad"] = node.requires_grad();
        obj["value"] = to_json_array(node.val());
        if (node.requires_grad()) {
            obj["grad"] = to_json_array(node.grad());
        };
        obj["leaf"] = node.is_leaf();
        if (!node.is_leaf()) {
            boost::json::array input_nodes;
            for (const auto &inode : node.input_nodes()) {
                input_nodes.emplace_back(inode.id());
            }
            obj["input_ndoes"] = input_nodes;
            obj["fn"] = node.fn()->to_json();
        }

        return obj;
    }
    Var<T> _node_from_json(boost::json::object &obj) const {
        Var<T> res =Var<T>{std::make_shared<VarImpl<T>>(
            obj["rows"].as_int64(), obj["cols"].as_int64(), obj["requires_grad"].as_bool())};
        // Set values.
        TMat<T> value = from_json_array<T>(obj["value"].as_array());
        res.set_val(value);
        // Set grad
        if (obj["requires_grad"].as_bool()) {
            TMat<T> grad = from_json_array<T>(obj["grad"].as_array());
            // res->_grad = grad;
            res.set_grad(grad);
        }
        // Deal with input: creating, fn.
        if (!obj["leaf"].as_bool()) {
            res.set_fn(
                _fn_factory[boost::json::value_to<std::string>(obj["fn"].get_object()["name"])](
                    obj["fn"]));
        }
        return res;
    }

    // We must initialize _input_node after all node have been read.
    void _create_input_node(){};
    void _update_nparm() {
        _nparm = 0;
        for (const auto &node : _parm_nodes) {
            _nparm += node.size();
        }
    };
    inline static std::map<
        std::string, std::function<std::unique_ptr<EvalGradFunctionBase<T>>(boost::json::value &v)>>
        _fn_factory{{"NegationEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<NegationEvalGrad<T>>();
                     }},
                    {"PlusEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<PlusEvalGrad<T>>();
                     }},
                    {"MinusEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<MinusEvalGrad<T>>();
                     }},
                    {"TimesEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<TimesEvalGrad<T>>();
                     }},
                    {"DivisionEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<DivisionEvalGrad<T>>();
                     }},
                    {"ExpEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<ExpEvalGrad<T>>();
                     }},
                    {"LnEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<LnEvalGrad<T>>();
                     }},
                    {"SinEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<SinEvalGrad<T>>();
                     }},
                    {"CosEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<CosEvalGrad<T>>();
                     }},
                    {"TanEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<TanEvalGrad<T>>();
                     }},
                    {"SinhEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<SinhEvalGrad<T>>();
                     }},
                    {"CoshEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<CoshEvalGrad<T>>();
                     }},
                    {"TanhEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<TanhEvalGrad<T>>();
                     }},
                    {"SigmoidEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<SigmoidEvalGrad<T>>();
                     }},
                    {"ReLUEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<ReLUEvalGrad<T>>();
                     }},
                    {"PowEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<PowEvalGrad<T>>(v.get_object()["n"].get_int64());
                     }},
                    {"TransposeEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<TransposeEvalGrad<T>>();
                     }},
                    {"LSdotEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<LSdotEvalGrad<T>>();
                     }},
                    {"RSdotEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<RSdotEvalGrad<T>>();
                     }},
                    {"DiagEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<DiagEvalGrad<T>>();
                     }},
                    {"Diag2EvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<Diag2EvalGrad<T>>();
                     }},
                    {"TraceEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<TraceEvalGrad<T>>();
                     }},
                    {"Trace2EvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<Trace2EvalGrad<T>>();
                     }},
                    {"DetEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<DetEvalGrad<T>>();
                     }},
                    {"LogDetEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<LogDetEvalGrad<T>>();
                     }},
                    {"InvEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<InvEvalGrad<T>>();
                     }},
                    {"IVechEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<IVechEvalGrad<T>>();
                     }},
                    {"IVeclEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<IVeclEvalGrad<T>>();
                     }},
                    {"IVecuEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<IVecuEvalGrad<T>>();
                     }},
                    {"LinearEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<LinearEvalGrad<T>>();
                     }},
                    {"SumEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<SumEvalGrad<T>>(v.get_object()["D"].get_int64());
                     }},
                    {"MeanEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<MeanEvalGrad<T>>(v.get_object()["D"].get_int64());
                     }},
                    {"VarianceEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<VarianceEvalGrad<T>>(v.get_object()["D"].get_int64());
                     }},
                    {"LnMVNormalDenEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<LnMVNormalDenEvalGrad<T>>(
                             v.get_object()["R"].get_int64());
                     }},
                    {"LnNormalDenEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<LnNormalDenEvalGrad<T>>(
                             v.get_object()["R"].get_int64());
                     }},
                    {"BCELossEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<BCELossEvalGrad<T>>(
                             v.get_object()["Reduction"].as_int64());
                     }},
                    {"MSELossEvalGrad",
                     [](boost::json::value &v) -> std::unique_ptr<EvalGradFunctionBase<T>> {
                         return std::make_unique<MSELossEvalGrad<T>>(
                             v.get_object()["Reduction"].as_int64());
                     }}};

};

}; // namespace DynAutoDiff

#endif