#include "test.hpp"

#include <stdexcept>
#include <string>

namespace {

using dnp::MatrixOp;
using dnp::Shape;
using dnp::Tensor;
using dnp::test::expectEqual;
using dnp::test::expectNear;
using dnp::test::expectShape;
using dnp::test::expectTensorNear;
using dnp::test::expectThrows;
using dnp::test::expectTrue;

DNP_TEST(
    tensor_constructs_from_shape
) {
    const Tensor<float> tensor(Shape{2, 3});

    expectShape(tensor, 2, 3, "shape constructor");
    expectTensorNear(tensor,
        2,
        3,
        {0, 0, 0, 0, 0, 0},
        1e-6,
        "shape constructor zero initialization");
}

DNP_TEST(
    tensor_constructs_from_initializer_list
) {
    const Tensor<float> tensor{{1, 2, 3}, {4, 5, 6}};

    expectTensorNear(tensor,
        2,
        3,
        {1, 2, 3, 4, 5, 6},
        1e-6,
        "initializer-list constructor");

    expectThrows<std::invalid_argument>(
        []() {
            Tensor<float> empty(
                std::initializer_list<std::initializer_list<float>>{}
            );
            (void)empty;
        },
        "empty initializer-list constructor"
    );
    expectThrows<std::invalid_argument>(
        []() {
            const Tensor<float> ragged{{1}, {2, 3}};
            (void)ragged;
        },
        "ragged initializer-list constructor"
    );
}

DNP_TEST(
    tensor_get_set_and_data_access
) {
    Tensor<double> tensor(Shape{2, 2});
    tensor.set(0, 1, 2.5);
    tensor.set(2, -3.0);
    tensor.data()[3] = 4.5;

    const Tensor<double>& const_tensor = tensor;
    expectNear(const_tensor.get(0, 1), 2.5, 1e-12, "coordinate set/get");
    expectNear(const_tensor.get(1, 0), -3.0, 1e-12, "flat set/get");
    expectNear(const_tensor.data()[3], 4.5, 1e-12, "data pointer access");

    expectThrows<std::invalid_argument>([&]() { tensor.get(2, 0); },
        "out-of-range get");
    expectThrows<std::invalid_argument>([&]() { tensor.get(4); },
        "out-of-range flat get");
    expectThrows<std::invalid_argument>([&]() { tensor.set(4, 1.0); },
        "out-of-range set");
}

DNP_TEST(
    tensor_arithmetic_and_compound_operators
) {
    const Tensor<float> left{{1, 2}, {3, 4}};
    const Tensor<float> right{{5, 6}, {7, 8}};

    expectTensorNear(left + right,
        2,
        2,
        {6, 8, 10, 12},
        1e-6,
        "tensor addition");
    expectTensorNear(right - left,
        2,
        2,
        {4, 4, 4, 4},
        1e-6,
        "tensor subtraction");
    expectTensorNear(left * 2.5f,
        2,
        2,
        {2.5, 5.0, 7.5, 10.0},
        1e-6,
        "scalar multiplication");

    Tensor<float> compound = left;
    compound += right;
    expectTensorNear(compound, 2, 2, {6, 8, 10, 12}, 1e-6, "compound addition");
    compound -= right;
    expectTensorNear(compound,
        2,
        2,
        {1, 2, 3, 4},
        1e-6,
        "compound subtraction");
    compound *= 3.0f;
    expectTensorNear(compound,
        2,
        2,
        {3, 6, 9, 12},
        1e-6,
        "compound scalar multiplication");
}

DNP_TEST(
    tensor_arithmetic_rejects_shape_mismatches
) {
    Tensor<float> matrix{{1, 2}, {3, 4}};
    const Tensor<float> row{{1, 2}};

    expectThrows<std::invalid_argument>([&]() { (void)(matrix + row); },
        "addition shape mismatch");
    expectThrows<std::invalid_argument>([&]() { (void)(matrix - row); },
        "subtraction shape mismatch");
    expectThrows<std::invalid_argument>([&]() { matrix += row; },
        "compound addition shape mismatch");
    expectThrows<std::invalid_argument>([&]() { matrix -= row; },
        "compound subtraction shape mismatch");
}

DNP_TEST(
    tensor_elementwise_multiplication
) {
    Tensor<double> left{{1, 2, 3}, {4, 5, 6}};
    const Tensor<double> right{{2, -1, 0.5}, {0, 3, -2}};

    expectTensorNear(left.elementwise_mul(right),
        2,
        3,
        {2.0, -2.0, 1.5, 0.0, 15.0, -12.0},
        1e-12,
        "elementwise multiplication");

    const Tensor<double> invalid{{1, 2}};
    expectThrows<std::invalid_argument>(
        [&]() { (void)left.elementwise_mul(invalid); },
        "elementwise multiplication shape mismatch"
    );
}

DNP_TEST(
    tensor_map_applies_function_to_all_values
) {
    Tensor<float> tensor{{-2, -1}, {0, 3}};
    tensor.map([](float p_value) { return p_value * p_value + 1.0f; });

    expectTensorNear(tensor, 2, 2, {5, 2, 1, 10}, 1e-6, "tensor map");
}

DNP_TEST(
    tensor_matrix_multiplication
) {
    const Tensor<float> left{{1, 2, 3}, {4, 5, 6}};
    const Tensor<float> right{{7, 8}, {9, 10}, {11, 12}};

    expectTensorNear(left.matmul(right),
        2,
        2,
        {58, 64, 139, 154},
        1e-6,
        "normal matrix multiplication");
}

DNP_TEST(
    tensor_matrix_multiplication_transpose_modes
) {
    const Tensor<float> left{{1, 2, 3}, {4, 5, 6}};
    const Tensor<float> two_by_two{{1, 2}, {3, 4}};
    const Tensor<float> other_transposed{{7, 8, 9}, {10, 11, 12}};
    const Tensor<float> both_transposed{{1, 2}, {3, 4}, {5, 6}, {7, 8}};

    expectTensorNear(left.matmul(two_by_two, MatrixOp::Transpose),
        3,
        2,
        {13, 18, 17, 24, 21, 30},
        1e-6,
        "transpose left operand");
    expectTensorNear(
        left.matmul(other_transposed, MatrixOp::None, MatrixOp::Transpose),
        2,
        2,
        {50, 68, 122, 167},
        1e-6,
        "transpose right operand"
    );
    expectTensorNear(
        left.matmul(both_transposed, MatrixOp::Transpose, MatrixOp::Transpose),
        3,
        4,
        {9, 19, 29, 39, 12, 26, 40, 54, 15, 33, 51, 69},
        1e-6,
        "transpose both operands"
    );
}

DNP_TEST(
    tensor_matrix_multiplication_rejects_invalid_dimensions
) {
    const Tensor<float> left{{1, 2, 3}, {4, 5, 6}};
    const Tensor<float> right{{1, 2}, {3, 4}};

    expectThrows<std::invalid_argument>([&]() { (void)left.matmul(right); },
        "matmul invalid dimensions",
        "shapes don't match");
}

DNP_TEST(
    tensor_add_row_vector
) {
    const Tensor<float> matrix{{1, 2, 3}, {4, 5, 6}};
    const Tensor<float> row{{10, 20, 30}};
    const Tensor<float> result = matrix.addRowVector(row);

    expectTensorNear(result,
        2,
        3,
        {11, 22, 33, 14, 25, 36},
        1e-6,
        "row-vector addition");
    expectTensorNear(matrix,
        2,
        3,
        {1, 2, 3, 4, 5, 6},
        1e-6,
        "row-vector addition leaves source unchanged");
}

DNP_TEST(
    tensor_add_row_vector_rejects_invalid_shapes
) {
    const Tensor<float> matrix{{1, 2, 3}, {4, 5, 6}};
    const Tensor<float> wrong_columns{{1, 2}};
    const Tensor<float> multiple_rows{{1, 2, 3}, {4, 5, 6}};

    expectThrows<std::invalid_argument>(
        [&]() { (void)matrix.addRowVector(wrong_columns); },
        "row vector with wrong column count"
    );
    expectThrows<std::invalid_argument>(
        [&]() { (void)matrix.addRowVector(multiple_rows); },
        "row vector with multiple rows"
    );
}

DNP_TEST(
    tensor_sum_rows
) {
    const Tensor<float> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    const Tensor<float> single_row{{-1, 2, 4}};

    expectTensorNear(matrix.sumRows(),
        1,
        3,
        {12, 15, 18},
        1e-6,
        "sum multiple rows");
    expectTensorNear(single_row.sumRows(),
        1,
        3,
        {-1, 2, 4},
        1e-6,
        "sum single row");
}

DNP_TEST(
    tensor_to_string_contains_shape_and_values
) {
    const Tensor<float> tensor{{1, 2.5f}, {-3, 4}};
    const std::string output = tensor.toString();

    expectTrue(output.find("shape=2x2") != std::string::npos,
        "tensor string contains shape");
    expectTrue(output.find("[1, 2.5]") != std::string::npos,
        "tensor string contains first row");
    expectTrue(output.find("[-3, 4]") != std::string::npos,
        "tensor string contains second row");
}

} // namespace
