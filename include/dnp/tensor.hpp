#ifndef DNP_TENSOR_HPP
#define DNP_TENSOR_HPP

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace dnp {

using Shapes = std::vector<std::uint32_t>;

enum class MatrixOp { None, Transpose };

template <typename T> class Tensor {
  public:
    Tensor(const Shapes& p_shape);
    Tensor(std::initializer_list<std::uint32_t> p_shapes);

    template <typename Generator> void generate(Generator p_generator);

    template <typename F> void map(F p_function);

    void set(std::size_t i, T v);
    T get(std::size_t i) const;
    T* data() {
        return _data.data();
    }
    const T* data() const {
        return _data.data();
    }

    Shapes shapes() const;
    std::size_t size() const;
    std::string toString() const;

    void reshape(const Shapes p_shapes);
    Tensor<T> transpose() const;

    Tensor<T> matmul(const Tensor<T>& p_other,
        MatrixOp p_this_op = MatrixOp::None,
        MatrixOp p_other_op = MatrixOp::None) const;
    Tensor<T> batched_matmul(const Tensor<T>& p_other,
    MatrixOp p_this_op,
    MatrixOp p_other_op) const;
    Tensor<T> elementwise_mul(const Tensor<T>& p_other);
    Tensor<T> addRowVector(const Tensor<T>& p_row) const;
    Tensor<T> sumRows() const;

    Tensor<T> operator+(const Tensor<T>& p_other) const;
    Tensor<T> operator-(const Tensor<T>& p_other) const;
    Tensor<T> operator*(T p_scalar) const;

    Tensor<T>& operator+=(const Tensor<T>& p_other);
    Tensor<T>& operator-=(const Tensor<T>& p_other);
    Tensor<T>& operator*=(T p_scalar);

  private:
    std::vector<T> _data;
    Shapes _shapes;
    std::size_t _size;

    static std::size_t computeSize(const Shapes p_shapes);
};

} // namespace dnp

#ifndef DNP_TENSOR_TPP
#include "dnp/tensor.tpp"
#endif

#endif
