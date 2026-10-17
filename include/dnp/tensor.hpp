#ifndef DNP_TENSOR_HPP
#define DNP_TENSOR_HPP

#include <cstddef>
#include <vector>

namespace dnp {

struct Shape {
    std::size_t rows;
    std::size_t cols;
};

enum class MatrixOp { None, Transpose };

template <typename T> class Tensor {
  public:
    Tensor(Shape p_shape);
    Tensor(std::initializer_list<std::initializer_list<T>> p_values);

    template <typename Generator> void generate(Generator p_generator);

    template <typename F> void map(F p_function);

    void set(std::size_t x, std::size_t y, T v);
    void set(std::size_t i, T v);
    T get(std::size_t x, std::size_t y) const;

    Shape shape() const;
    std::size_t size() const;

    Tensor<T> matmul(
        const Tensor<T>& p_other,
        MatrixOp p_this_op = MatrixOp::None,
        MatrixOp p_other_op = MatrixOp::None
    ) const;
    Tensor<T> elementwise_mul(const Tensor<T>& p_other);

    Tensor<T> operator+(const Tensor<T>& p_other) const;
    Tensor<T> operator-(const Tensor<T>& p_other) const;
    Tensor<T> operator*(T p_scalar) const;

    Tensor<T>& operator+=(const Tensor<T>& p_other);
    Tensor<T>& operator-=(const Tensor<T>& p_other);
    Tensor<T>& operator*=(T p_scalar);

  private:
    std::vector<T> _data;
    Shape _shape;
    std::size_t _size;
};

} // namespace dnp

#ifndef DNP_TENSOR_TPP
#include "dnp/tensor.tpp"
#endif

#endif