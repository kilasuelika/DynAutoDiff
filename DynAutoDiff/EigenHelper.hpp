#ifndef __DYNAUTODIFF_EIGENHELPER__
#define __DYNAUTODIFF_EIGENHELPER__
#include "boost/lexical_cast.hpp"
// #include "strtk/strtk.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include "3rd_party/csv-parser/csv.hpp"
#include <boost/json.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/util/Constants.h>

namespace DynAutoDiff {
// Type definition
template <typename T = double>
using TMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using TMatd=TMat<double>;

template <typename T = double>
using TArr = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T = double> using TVec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T = double> using TVecA = Eigen::Array<T, Eigen::Dynamic, 1>;
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
template <typename T = double> auto decide_tab_file_shape(const std::string &filename) {
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
template <typename T = double> TMat<T> load_tab_file(const std::string &filename) {
    if (!std::filesystem::exists(std::filesystem::path(filename))) {
        throw std::range_error("File " + filename + " not exists.");
    }
    auto [rows, cols] = decide_tab_file_shape<T>(filename);
    std::ifstream file(filename);
    TMat<T> res(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> res.coeffRef(i, j);
        }
    }
    return res;
};

// decide shape of csv file.
template <typename T = double> auto decide_csv_file_shape(const std::string &filename) {
    csv::CSVReader reader(filename);
    int rows = 0, cols = 0;

    csv::CSVRow row;
    const std::string csv_delimiter = ",";
    //strtk::parse(line, csv_delimiter, line_tokens);


    ++rows;
    while (reader.read_row(row)) {
        if (row.size() > 0)
            ++rows;

        cols = row.size();
    }

    return std::make_pair(rows, cols);
}

// Load csv file. Shape is automatically detected.
template <typename T = double>
TMat<T> load_csv_file(const std::string &filename, int skip_row = 0, int skip_column = 0) {
    auto [rows, cols] = decide_csv_file_shape<T>(filename);
    //std::ifstream file(filename);
    csv::CSVReader reader(filename);
    TMat<T> res(rows - skip_row, cols - skip_column);

    const std::string csv_delimiter = ",";

    std::string line;

    int k = 0, km = 0;
    for (csv::CSVRow& row: reader) {
        ++k;
        if (k <= skip_row) {
            continue;
        } else {
            int i=0;
            for (csv::CSVField& field: row){
                if (i<skip_column) continue;
            //for (int i = skip_column; i < line_tokens.size(); ++i) {
                T val;
                try {
                    val = std::stod(field.get<>());
                } catch (...) {
                    throw std::runtime_error("Failed to parse number at Row " +
                                             std::to_string(k) + " Col " + std::to_string(i) +
                                             ", string is: " + field.get<>());
                }
                res.coeffRef(km, i - skip_column) = val;
            };
            ++km;
        };
    }

    return res;
};

template <typename T = double>
TMat<T> load_mat(const std::string &filename, int skip_row = 0, int skip_column = 0) {
    auto p = std::filesystem::path(filename);
    if (!std::filesystem::exists(p)) {
        throw std::range_error("File " + filename + " not exists.");
    };
    // Load data.
    if (p.extension() == std::filesystem::path(".txt")) {
        return load_tab_file<T>(filename);
    } else if (p.extension() == std::filesystem::path(".csv")) {
        return load_csv_file<T>(filename, skip_row, skip_column);
    } else {
        throw std::runtime_error("Unknown file type " + p.extension().string());
    }
};

template <typename T>
void write_mat(const T &mat, const std::string &filename, const std::string &delimiter = ",") {
    std::ofstream file(filename);
    int m = mat.rows(), n = mat.cols();
    for (int i = 0; i < m - 1; ++i) {
        for (int j = 0; j < n - 1; ++j) {
            file << mat.coeff(i, j) << delimiter;
        }
        file << mat.coeff(i, n - 1) << std::endl;
    }
    for (int j = 0; j < n - 1; ++j) {
        file << mat.coeff(m - 1, j) << delimiter;
    }
    file << mat.coeff(m - 1, n - 1);
}
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
