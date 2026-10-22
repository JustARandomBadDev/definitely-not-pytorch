#ifndef DNP_TENSOR_TPP
#define DNP_TENSOR_TPP

#include "dnp/tensor.hpp"

#include <cstddef>
#include <sstream>
#include <stdexcept>

namespace dnp {

template <typename T>
Tensor<T>::Tensor(
    Shape p_shape
) {
    _shape = p_shape;
    _size = _shape.rows * _shape.cols;
    _data.resize(_size, static_cast<T>(0));
}

template <typename T>
Tensor<T>::Tensor(
    std::initializer_list<std::initializer_list<T>> p_values
) {
    _shape = {p_values.size(), p_values.begin()->size()};

    _size = _shape.rows * _shape.cols;
    _data.resize(_size);

    std::size_t row = 0;
    std::size_t col = 0;

    for (const auto& values_row : p_values) {
        for (const auto& value : values_row) {
            _data[row * _shape.cols + col] = value;
            col++;
        }

        row++;
        col = 0;
    }
}

template <typename T>
template <typename Generator>
void Tensor<T>::generate(
    Generator p_generator
) {
    for (T& v : _data) {
        v = p_generator();
    }
}

template <typename T>
template <typename F>
void Tensor<T>::map(
    F p_function
) {
    for (T& v : _data) {
        v = p_function(v);
    }
}

// =======================================================
//                         Setters
// =======================================================

template <typename T>
void Tensor<T>::set(
    std::size_t row,
    std::size_t col,
    const T v
) {
    if (row >= _shape.rows || col >= _shape.cols) {
        throw std::invalid_argument("Set outside tensor shape");
    }

    _data[row * _shape.cols + col] = v;
}

template <typename T>
void Tensor<T>::set(
    std::size_t i,
    const T v
) {
    if (i >= _size) {
        throw std::invalid_argument("Set outside tensor shape");
    }

    _data[i] = v;
}

// =======================================================
//                         Getters
// =======================================================

template <typename T>
T Tensor<T>::get(
    std::size_t row,
    std::size_t col
) const {
    if (row >= _shape.rows || col >= _shape.cols) {
        throw std::invalid_argument("Get outside tensor shape");
    }

    return _data[row * _shape.cols + col];
}

template <typename T> Shape Tensor<T>::shape() const {
    return _shape;
}

template <typename T> std::size_t Tensor<T>::size() const {
    return _size;
}

template <typename T> std::string Tensor<T>::toString() const {
    std::ostringstream stream;

    stream << "Tensor(shape=" << _shape.rows << 'x' << _shape.cols
           << ", values=[";

    if (_shape.rows > 0) stream << '\n';

    for (std::size_t row = 0; row < _shape.rows; row++) {
        stream << "    [";

        for (std::size_t col = 0; col < _shape.cols; col++) {
            if (col > 0) stream << ", ";

            stream << get(row, col);
        }

        stream << ']';

        if (row + 1 < _shape.rows) stream << ',';

        stream << '\n';
    }

    stream << "])";

    return stream.str();
}

// ==============================================================
//                         Copy operators
// ==============================================================

template <typename T>
Tensor<T> Tensor<T>::matmul(
    const Tensor<T>& p_other,
    MatrixOp p_this_op,
    MatrixOp p_other_op
) const {
    const Shape other_shape = p_other.shape();

    const bool transpose_this = p_this_op == MatrixOp::Transpose;
    const bool transpose_other = p_other_op == MatrixOp::Transpose;

    const std::size_t this_rows = transpose_this ? _shape.cols : _shape.rows;

    const std::size_t this_cols = transpose_this ? _shape.rows : _shape.cols;

    const std::size_t other_rows =
        transpose_other ? other_shape.cols : other_shape.rows;

    const std::size_t other_cols =
        transpose_other ? other_shape.rows : other_shape.cols;

    if (this_cols != other_rows) {
        throw std::invalid_argument("matmul: Tensor shapes don't match");
    }

    Tensor<T> result({this_rows, other_cols});

    for (std::size_t row = 0; row < this_rows; row++) {
        for (std::size_t col = 0; col < other_cols; col++) {
            T value = static_cast<T>(0);

            for (std::size_t i = 0; i < this_cols; i++) {
                const T a = transpose_this ? get(i, row) : get(row, i);

                const T b =
                    transpose_other ? p_other.get(col, i) : p_other.get(i, col);

                value += a * b;
            }

            result.set(row, col, value);
        }
    }

    return result;
}

template <typename T>
Tensor<T> Tensor<T>::elementwise_mul(
    const Tensor<T>& p_other
) {
    if (_shape.rows != p_other._shape.rows ||
        _shape.cols != p_other._shape.cols) {
        throw std::invalid_argument("Tensor shapes don't match");
    }

    Tensor<float> result(_shape);

    for (std::size_t i = 0; i < _size; i++) {
        result.set(i, _data[i] * p_other._data[i]);
    }

    return result;
}

template <typename T>
Tensor<T> Tensor<T>::addRowVector(
    const Tensor<T>& p_row
) const {
    if (p_row.shape().rows != 1 || p_row.shape().cols != _shape.cols) {
        throw std::invalid_argument("Row tensor shape must be [1, cols]");
    }

    Tensor<T> result(_shape);

    for (std::size_t row = 0; row < _shape.rows; row++) {
        for (std::size_t col = 0; col < _shape.cols; col++) {
            result.set(row, col, get(row, col) + p_row.get(0, col));
        }
    }

    return result;
}

template <typename T> Tensor<T> Tensor<T>::sumRows() const {
    Tensor<T> result(Shape{1, _shape.cols});

    for (std::size_t col = 0; col < _shape.cols; col++) {
        T sum = static_cast<T>(0);

        for (std::size_t row = 0; row < _shape.rows; row++) {
            sum += get(row, col);
        }

        result.set(0, col, sum);
    }

    return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator+(
    const Tensor<T>& p_other
) const {
    if (_shape.rows != p_other._shape.rows ||
        _shape.cols != p_other._shape.cols) {
        throw std::invalid_argument("Tensor shapes don't match");
    }

    Tensor<T> result = *this;

    result += p_other;

    return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator-(
    const Tensor<T>& p_other
) const {
    if (_shape.rows != p_other._shape.rows ||
        _shape.cols != p_other._shape.cols) {
        throw std::invalid_argument("Tensor shapes don't match");
    }

    Tensor<T> result = *this;

    result -= p_other;

    return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator*(
    T p_scalar
) const {
    Tensor<T> result = *this;

    for (T& v : result._data) {
        v *= p_scalar;
    }

    return result;
}

// ========================================================
//                         Operators
// ========================================================

template <typename T>
Tensor<T>& Tensor<T>::operator+=(
    const Tensor<T>& p_other
) {
    if (_shape.rows != p_other._shape.rows ||
        _shape.cols != p_other._shape.cols) {
        throw std::invalid_argument("Tensor shapes don't match");
    }

    for (std::size_t i = 0; i < _size; i++) {
        _data[i] += p_other._data[i];
    }

    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator-=(
    const Tensor<T>& p_other
) {
    if (_shape.rows != p_other._shape.rows ||
        _shape.cols != p_other._shape.cols) {
        throw std::invalid_argument("Tensor shapes don't match");
    }

    for (std::size_t i = 0; i < _size; i++) {
        _data[i] -= p_other._data[i];
    }

    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator*=(
    T p_scalar
) {
    for (T& v : _data) {
        v *= p_scalar;
    }

    return *this;
}

} // namespace dnp

#endif
