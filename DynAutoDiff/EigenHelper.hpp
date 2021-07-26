#ifndef __DYNAUTODIFF_EIGENHELPER__
#define __DYNAUTODIFF_EIGENHELPER__
#include <boost/json.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace DynAutoDiff {
// Type definition
template <typename T = double>
using TMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <typename T = double>
using TArr = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <typename T = double> using TVec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T = double> using TRVec = Eigen::Matrix<T, 1, Eigen::Dynamic>;
using TVecd = Eigen::Matrix<double, Eigen::Dynamic, 1>;
template <typename T = double> using TMap = Eigen::Map<TMat<T>>;

// Scalar to a Eigen::Matrix
template <typename T> inline TMat<T> eigen_scalar_mat(const T &v) {
    TMat<T> res(1, 1);
    res << v;
    return res;
};

// Broadcasting mul.
// mxn * 1xn: return mxn, each column of mxn is scaled by each col of 1xn.
// mxn * mx1: likewise.
// Or switching order.
#define BROADCASTINGMULERROR                                                                       \
    throw std::range_error("Invalid broadcasting mul. Shapes are (" + std::to_string(l.rows()) +   \
                           ", " + std::to_string(l.cols()) + "), (" + std::to_string(r.rows()) +   \
                           ", " + std::to_string(r.cols()) + ").");
template <typename T1, typename T2>
TMat<typename T1::Scalar> _broadcasting_mul(const T1 &l, const T2 &r) {
    using T = typename T1::Scalar;
    // mxn, mx1
    if (l.rows() == r.rows()) {
        if (r.cols() == 1) {
            TMat<T> res = l;
            for (int j = 0; j < l.rows(); ++j) {
                res.row(j) = res.row(j) * r.coeff(j, 0);
            }
            return res;
        } else if (l.cols() == 1) {
            return _broadcasting_mul(r, l);
        } else {
            BROADCASTINGMULERROR
        }
    } else if (l.cols() == r.cols()) {
        if (l.rows() == 1) {
            TMat<T> res = r;
            for (int j = 0; j < r.cols(); ++j) {
                res.col(j) = res.col(j) * l.coeff(0, j);
            }
            return res;
        } else if (r.rows() == 1) {
            return _broadcasting_mul(r, l);
        } else {
            BROADCASTINGMULERROR
        }
    } else {
        BROADCASTINGMULERROR
    }
};

// negation operator to a Map<>
/*template<typename T>
TMat<T> operator -(const TMap<T>& op){
        TMat<T> res=op;
        res=-res;
        return res;
};*/

// decide shape of file.
template <typename T = double> auto decide_shape(const std::string &filename) {
    std::ifstream file(filename);
    int rows = 0, cols = 0;
    std::string str;
    std::getline(file, str);
    std::stringstream ss(str);
    double tmp;
    while (ss >> tmp) {
        ++cols;
    };
    ++rows;
    while (std::getline(file, str)) {
        if (str.size() > 0)
            ++rows;
    }

    return std::make_pair(rows, cols);
};
// Load tab delimited file. Shape is automatically detected.
template <typename T = double> TMat<T> load_mat(const std::string &filename) {
    if (!std::filesystem::exists(std::filesystem::path(filename))) {
        throw std::range_error("File " + filename + " not exists.");
    }
    auto [rows, cols] = decide_shape(filename);
    std::ifstream file(filename);
    TMat<T> res(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> res.coeffRef(i, j);
        }
    }
    return res;
};

// Convert a matrix to a json array.
template <typename T> boost::json::array to_json_array(const T &data) {
    boost::json::array res;
    for (int i = 0; i < data.rows(); ++i) {
        boost::json::array arr;
        for (int j = 0; j < data.cols(); ++j) {
            arr.emplace_back(data.coeff(i, j));
        }
        res.emplace_back(arr);
    }
    return res;
};
template <std::floating_point T = double> TMat<T> from_json_array(const boost::json::array &data) {
    int rows = data.size(), cols = data[0].as_array().size();
    TMat<T> res(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            res.coeffRef(i, j) = data[i].as_array()[j].get_double();
        }
    }
    return res;
};
template <typename T = double> inline T sign(T x) {
    if (x > 0)
        return 1;
    else if (x == 0)
        return 0;
    else
        return -1;
};

}; // namespace DynAutoDiff
#endif