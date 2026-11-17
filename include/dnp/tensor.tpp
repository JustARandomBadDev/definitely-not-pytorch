#ifndef DNP_TENSOR_TPP
#define DNP_TENSOR_TPP

#include "dnp/tensor.hpp"

#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace dnp {

template <typename T>
Tensor<T>::Tensor(
    const Shapes& p_shape
) {
    _shapes = p_shape;
    _size = computeSize(_shapes);
    _data.resize(_size, static_cast<T>(0));
}

template <typename T>
Tensor<T>::Tensor(
    std::initializer_list<std::uint32_t> p_shapes
)
    : Tensor(Shapes(p_shapes)) {}

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

template <typename T>
void Tensor<T>::reshape(const Shapes p_shapes) {
    const auto new_size = computeSize(p_shapes);

    if (new_size != _size) throw std::runtime_error("Invalid size of new shape");

    _shapes = p_shapes;
    _size = new_size;
}

template<typename T>
Tensor<T> Tensor<T>::transpose() const {
    if (_shapes.size() < 2)
        throw std::runtime_error("Cannot transpose 1D tensor");

    const Shapes matrix_shape(_shapes.end() - 2, _shapes.end());
    const auto matrix_size = computeSize(matrix_shape);

    Shapes new_shapes;
    new_shapes.reserve(_shapes.size());

    new_shapes.insert(new_shapes.end(), _shapes.begin(), _shapes.end() - 2);
    new_shapes.push_back(matrix_shape[1]);
    new_shapes.push_back(matrix_shape[0]);

    Tensor<T> new_tensor(new_shapes);

    for (std::size_t i = 0; i < _size; i += matrix_size) {
        for (std::size_t y = 0; y < matrix_shape[0]; y++) {
            for (std::size_t x = 0; x < matrix_shape[1]; x++) {
                new_tensor.set(
                    i + x * matrix_shape[0] + y,
                    _data[i + y * matrix_shape[1] + x]
                );
            }
        }
    }

    return new_tensor;
}

// =======================================================
//                         Private
// =======================================================

template <typename T> std::size_t Tensor<T>::computeSize(const Shapes p_shapes) {
    if (p_shapes.empty()) {
        throw std::invalid_argument("Invalid tensor shape");
    }

    std::size_t size = 1;

    for (const auto shape : p_shapes) {
        if (shape == 0) {
            throw std::invalid_argument(
                "Tensor dimensions must be greater than zero"
            );
        }
        if (size > std::numeric_limits<std::size_t>::max() / shape) {
            throw std::overflow_error("Tensor size overflow");
        }
        size *= shape;
    }

    return size;
}

// =======================================================
//                         Setters
// =======================================================

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
    std::size_t i
) const {
    if (i >= _size) {
        throw std::invalid_argument("Get outside tensor shape");
    }

    return _data[i];
}

template <typename T> Shapes Tensor<T>::shapes() const {
    return _shapes;
}

template <typename T> std::size_t Tensor<T>::size() const {
    return _size;
}

template <typename T> std::string Tensor<T>::toString() const {
    std::ostringstream stream;

    stream << "Tensor(shape=[";

    for (std::size_t i = 0; i < _shapes.size(); i++) {
        if (i > 0) stream << ", ";
        stream << _shapes[i];
    }

    stream << "], values=[";

    if (_shapes.size() == 2) {
        stream << '\n';

        for (std::size_t row = 0; row < _shapes[0]; row++) {
            stream << "    [";

            for (std::size_t col = 0; col < _shapes[1]; col++) {
                if (col > 0) stream << ", ";
                stream << get(row * _shapes[1] + col);
            }

            stream << ']';
            if (row + 1 < _shapes[0]) stream << ',';
            stream << '\n';
        }
    } else {
        for (std::size_t i = 0; i < _size; i++) {
            if (i > 0) stream << ", ";
            stream << _data[i];
        }
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
    if (_shapes.size() != 2 || p_other._shapes.size() != 2) {
        throw std::invalid_argument("matmul requires 2D tensors");
    }

    const Shapes other_shape = p_other.shapes();

    const bool transpose_this = p_this_op == MatrixOp::Transpose;
    const bool transpose_other = p_other_op == MatrixOp::Transpose;

    const std::uint32_t this_rows = transpose_this ? _shapes[1] : _shapes[0];
    const std::uint32_t this_cols = transpose_this ? _shapes[0] : _shapes[1];

    const std::uint32_t other_rows =
        transpose_other ? other_shape[1] : other_shape[0];

    const std::uint32_t other_cols =
        transpose_other ? other_shape[0] : other_shape[1];

    if (this_cols != other_rows) {
        throw std::invalid_argument("matmul: Tensor shapes don't match");
    }

    Tensor<T> result({this_rows, other_cols});

    for (std::size_t row = 0; row < this_rows; row++) {
        for (std::size_t col = 0; col < other_cols; col++) {
            T value = static_cast<T>(0);

            for (std::size_t i = 0; i < this_cols; i++) {
                const T a = transpose_this ? get(i * _shapes[1] + row)
                                           : get(row * _shapes[1] + i);

                const T b = transpose_other
                                ? p_other.get(col * other_shape[1] + i)
                                : p_other.get(i * other_shape[1] + col);

                value += a * b;
            }

            result.set(row * other_cols + col, value);
        }
    }

    return result;
}

template <typename T>
Tensor<T> Tensor<T>::batched_matmul(
    const Tensor<T>& p_other,
    MatrixOp p_this_op,
    MatrixOp p_other_op
) const {
    if (_shapes.size() < 3 || _shapes.size() != p_other._shapes.size()) {
        throw std::invalid_argument("batched_matmul requires matching N-D tensors");
    }

    const Shapes other_shape = p_other.shapes();
    const std::size_t ndim = _shapes.size();

    std::size_t batch_count = 1;

    for (std::size_t i = 0; i < ndim - 2; i++) {
        if (_shapes[i] != other_shape[i]) {
            throw std::invalid_argument("batched_matmul: Batch shapes don't match");
        }

        batch_count *= _shapes[i];
    }

    const bool transpose_this = p_this_op == MatrixOp::Transpose;
    const bool transpose_other = p_other_op == MatrixOp::Transpose;

    const std::uint32_t this_rows =
        transpose_this ? _shapes[ndim - 1] : _shapes[ndim - 2];
    const std::uint32_t this_cols =
        transpose_this ? _shapes[ndim - 2] : _shapes[ndim - 1];

    const std::uint32_t other_rows =
        transpose_other ? other_shape[ndim - 1] : other_shape[ndim - 2];
    const std::uint32_t other_cols =
        transpose_other ? other_shape[ndim - 2] : other_shape[ndim - 1];

    if (this_cols != other_rows) {
        throw std::invalid_argument("batched_matmul: Tensor shapes don't match");
    }

    Shapes result_shape(_shapes.begin(), _shapes.end() - 2);
    result_shape.push_back(this_rows);
    result_shape.push_back(other_cols);

    Tensor<T> result(result_shape);

    const std::size_t this_matrix_size =
        static_cast<std::size_t>(_shapes[ndim - 2]) * _shapes[ndim - 1];
    const std::size_t other_matrix_size =
        static_cast<std::size_t>(other_shape[ndim - 2]) * other_shape[ndim - 1];
    const std::size_t result_matrix_size =
        static_cast<std::size_t>(this_rows) * other_cols;

    for (std::size_t batch = 0; batch < batch_count; batch++) {
        const std::size_t offset_a = batch * this_matrix_size;
        const std::size_t offset_b = batch * other_matrix_size;
        const std::size_t offset_result = batch * result_matrix_size;

        for (std::size_t row = 0; row < this_rows; row++) {
            for (std::size_t col = 0; col < other_cols; col++) {
                T value = static_cast<T>(0);

                for (std::size_t i = 0; i < this_cols; i++) {
                    const T a = transpose_this
                        ? get(offset_a + i * _shapes[ndim - 1] + row)
                        : get(offset_a + row * _shapes[ndim - 1] + i);

                    const T b = transpose_other
                        ? p_other.get(offset_b + col * other_shape[ndim - 1] + i)
                        : p_other.get(offset_b + i * other_shape[ndim - 1] + col);

                    value += a * b;
                }

                result.set(
                    offset_result + row * other_cols + col,
                    value
                );
            }
        }
    }

    return result;
}

template <typename T>
Tensor<T> Tensor<T>::elementwise_mul(
    const Tensor<T>& p_other
) {
    if (_shapes != p_other._shapes) {
        throw std::invalid_argument("Tensor shapes don't match");
    }

    Tensor<T> result(_shapes);

    for (std::size_t i = 0; i < _size; i++) {
        result.set(i, _data[i] * p_other._data[i]);
    }

    return result;
}

template <typename T>
Tensor<T> Tensor<T>::addRowVector(
    const Tensor<T>& p_row
) const {
    if (_shapes.size() != 2 || p_row._shapes.size() != 2 ||
        p_row._shapes[0] != 1 || p_row._shapes[1] != _shapes[1]) {
        throw std::invalid_argument(
            "Row tensor shape must be [1, cols] (2D only)"
        );
    }

    Tensor<T> result(_shapes);

    for (std::size_t row = 0; row < _shapes[0]; row++) {
        for (std::size_t col = 0; col < _shapes[1]; col++) {
            result.set(row * _shapes[1] + col,
                get(row * _shapes[1] + col) + p_row.get(col));
        }
    }

    return result;
}

template <typename T> Tensor<T> Tensor<T>::sumRows() const {
    if (_shapes.size() != 2) {
        throw std::invalid_argument("sumRows requires a 2D tensor");
    }

    Tensor<T> result({1, _shapes[1]});

    for (std::size_t col = 0; col < _shapes[1]; col++) {
        T sum = static_cast<T>(0);

        for (std::size_t row = 0; row < _shapes[0]; row++) {
            sum += get(row * _shapes[1] + col);
        }

        result.set(col, sum);
    }

    return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator+(
    const Tensor<T>& p_other
) const {
    Tensor<T> result = *this;
    result += p_other;
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator-(
    const Tensor<T>& p_other
) const {
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
    if (_shapes != p_other._shapes) {
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
    if (_shapes != p_other._shapes) {
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
