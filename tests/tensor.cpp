#include "test.hpp"

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace {

using dnp::MatrixOp;
using dnp::Shapes;
using dnp::Tensor;
using dnp::test::expectEqual;
using dnp::test::expectNear;
using dnp::test::expectThrows;
using dnp::test::expectTrue;

template <typename T>
Tensor<T> makeTensor(
    std::initializer_list<std::initializer_list<T>> p_rows
) {
    if (p_rows.size() == 0 || p_rows.begin()->size() == 0) {
        throw std::invalid_argument("Tensor must have non-empty rows");
    }

    const std::size_t cols = p_rows.begin()->size();
    for (const auto& row : p_rows) {
        if (row.size() != cols) {
            throw std::invalid_argument("Tensor rows must have the same size");
        }
    }

    Tensor<T> tensor(
        {static_cast<std::uint32_t>(p_rows.size()),
            static_cast<std::uint32_t>(cols)}
    );

    std::size_t i = 0;
    for (const auto& row : p_rows) {
        for (const T value : row) {
            tensor.set(i++, value);
        }
    }

    return tensor;
}

template <typename T>
T get2D(
    const Tensor<T>& p_tensor,
    std::size_t p_row,
    std::size_t p_col
) {
    const auto shape = p_tensor.shapes();
    if (shape.size() != 2 || p_row >= shape[0] || p_col >= shape[1]) {
        throw std::invalid_argument("Get outside tensor shape");
    }
    return p_tensor.get(p_row * shape[1] + p_col);
}

template <typename T>
void set2D(
    Tensor<T>& p_tensor,
    std::size_t p_row,
    std::size_t p_col,
    T p_value
) {
    const auto shape = p_tensor.shapes();
    if (shape.size() != 2 || p_row >= shape[0] || p_col >= shape[1]) {
        throw std::invalid_argument("Set outside tensor shape");
    }
    p_tensor.set(p_row * shape[1] + p_col, p_value);
}

template <typename T>
void expectShape(
    const Tensor<T>& p_tensor,
    std::size_t p_rows,
    std::size_t p_cols,
    const std::string& p_context
) {
    const auto shape = p_tensor.shapes();
    expectEqual(shape.size(), std::size_t{2}, p_context + " dimension count");
    if (shape.size() != 2) return;
    expectEqual(shape[0],
        static_cast<std::uint32_t>(p_rows),
        p_context + " rows");
    expectEqual(shape[1],
        static_cast<std::uint32_t>(p_cols),
        p_context + " columns");
}

template <typename T>
void expectTensorNear(
    const Tensor<T>& p_tensor,
    std::size_t p_rows,
    std::size_t p_cols,
    std::initializer_list<double> p_expected,
    double p_tolerance,
    const std::string& p_context
) {
    expectShape(p_tensor, p_rows, p_cols, p_context);
    expectEqual(p_tensor.size(),
        p_expected.size(),
        p_context + " element count");

    std::size_t i = 0;
    for (const double value : p_expected) {
        if (i >= p_tensor.size()) break;
        expectNear(static_cast<double>(p_tensor.get(i)),
            value,
            p_tolerance,
            p_context + " element " + std::to_string(i));
        i++;
    }
}

DNP_TEST(
    tensor_constructs_from_shape
) {
    const Tensor<float> tensor(Shapes{2, 3});

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
    const Tensor<float> tensor({2, 3});
    expectShape(tensor, 2, 3, "initializer-list shape constructor");
    expectEqual(tensor.size(),
        std::size_t{6},
        "initializer-list element count");

    expectThrows<std::invalid_argument>(
        []() {
            const Tensor<float> empty(std::initializer_list<std::uint32_t>{});
            (void)empty;
        },
        "empty tensor shape"
    );
    expectThrows<std::invalid_argument>(
        []() {
            const Tensor<float> invalid({2, 0});
            (void)invalid;
        },
        "zero tensor dimension"
    );
}

DNP_TEST(
    tensor_supports_nd_shapes_and_linear_access
) {
    Tensor<float> tensor({2, 3, 4});
    const auto shape = tensor.shapes();

    expectEqual(shape.size(), std::size_t{3}, "3D dimension count");
    expectEqual(shape[0], std::uint32_t{2}, "3D first dimension");
    expectEqual(shape[1], std::uint32_t{3}, "3D second dimension");
    expectEqual(shape[2], std::uint32_t{4}, "3D third dimension");
    expectEqual(tensor.size(), std::size_t{24}, "3D element count");

    tensor.set(23, 12.5f);
    expectNear(tensor.get(23), 12.5f, 1e-6, "3D linear access");
    expectNear((tensor + tensor).get(23),
        25.0f,
        1e-6,
        "3D elementwise addition");
}

DNP_TEST(
    tensor_get_set_and_data_access
) {
    Tensor<double> tensor(Shapes{2, 2});
    set2D(tensor, 0, 1, 2.5);
    tensor.set(2, -3.0);
    tensor.data()[3] = 4.5;

    const Tensor<double>& const_tensor = tensor;
    expectNear(get2D(const_tensor, 0, 1), 2.5, 1e-12, "coordinate set/get");
    expectNear(get2D(const_tensor, 1, 0), -3.0, 1e-12, "flat set/get");
    expectNear(const_tensor.data()[3], 4.5, 1e-12, "data pointer access");

    expectThrows<std::invalid_argument>([&]() { get2D(tensor, 2, 0); },
        "out-of-range get");
    expectThrows<std::invalid_argument>([&]() { tensor.get(4); },
        "out-of-range flat get");
    expectThrows<std::invalid_argument>([&]() { tensor.set(4, 1.0); },
        "out-of-range set");
}

DNP_TEST(
    tensor_arithmetic_and_compound_operators
) {
    const Tensor<float> left = makeTensor<float>({{1, 2}, {3, 4}});
    const Tensor<float> right = makeTensor<float>({{5, 6}, {7, 8}});

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
    Tensor<float> matrix = makeTensor<float>({{1, 2}, {3, 4}});
    const Tensor<float> row = makeTensor<float>({{1, 2}});

    expectThrows<std::invalid_argument>([&]() { (void)(matrix + row); },
        "addition shape mismatch");
    expectThrows<std::invalid_argument>([&]() { (void)(matrix - row); },
        "subtraction shape mismatch");
    expectThrows<std::invalid_argument>([&]() { matrix += row; },
        "compound addition shape mismatch");
    expectThrows<std::invalid_argument>([&]() { matrix -= row; },
        "compound subtraction shape mismatch");

    const Tensor<float> reshaped({1, 4});
    expectThrows<std::invalid_argument>([&]() { (void)(matrix + reshaped); },
        "equal element count with different shapes");
}

DNP_TEST(
    tensor_elementwise_multiplication
) {
    Tensor<double> left = makeTensor<double>({{1, 2, 3}, {4, 5, 6}});
    const Tensor<double> right = makeTensor<double>({{2, -1, 0.5}, {0, 3, -2}});

    expectTensorNear(left.elementwise_mul(right),
        2,
        3,
        {2.0, -2.0, 1.5, 0.0, 15.0, -12.0},
        1e-12,
        "elementwise multiplication");

    const Tensor<double> invalid = makeTensor<double>({{1, 2}});
    expectThrows<std::invalid_argument>(
        [&]() { (void)left.elementwise_mul(invalid); },
        "elementwise multiplication shape mismatch"
    );
}

DNP_TEST(
    tensor_map_applies_function_to_all_values
) {
    Tensor<float> tensor = makeTensor<float>({{-2, -1}, {0, 3}});
    tensor.map([](float p_value) { return p_value * p_value + 1.0f; });

    expectTensorNear(tensor, 2, 2, {5, 2, 1, 10}, 1e-6, "tensor map");
}

DNP_TEST(
    tensor_matrix_multiplication
) {
    const Tensor<float> left = makeTensor<float>({{1, 2, 3}, {4, 5, 6}});
    const Tensor<float> right = makeTensor<float>({{7, 8}, {9, 10}, {11, 12}});

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
    const Tensor<float> left = makeTensor<float>({{1, 2, 3}, {4, 5, 6}});
    const Tensor<float> two_by_two = makeTensor<float>({{1, 2}, {3, 4}});
    const Tensor<float> other_transposed =
        makeTensor<float>({{7, 8, 9}, {10, 11, 12}});
    const Tensor<float> both_transposed =
        makeTensor<float>({{1, 2}, {3, 4}, {5, 6}, {7, 8}});

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
    const Tensor<float> left = makeTensor<float>({{1, 2, 3}, {4, 5, 6}});
    const Tensor<float> right = makeTensor<float>({{1, 2}, {3, 4}});

    expectThrows<std::invalid_argument>([&]() { (void)left.matmul(right); },
        "matmul invalid dimensions",
        "shapes don't match");

    const Tensor<float> nd({2, 3, 4});
    expectThrows<std::invalid_argument>([&]() { (void)nd.matmul(nd); },
        "matmul rejects non-2D input",
        "requires 2D");
}

DNP_TEST(
    tensor_add_row_vector
) {
    const Tensor<float> matrix = makeTensor<float>({{1, 2, 3}, {4, 5, 6}});
    const Tensor<float> row = makeTensor<float>({{10, 20, 30}});
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
    const Tensor<float> matrix = makeTensor<float>({{1, 2, 3}, {4, 5, 6}});
    const Tensor<float> wrong_columns = makeTensor<float>({{1, 2}});
    const Tensor<float> multiple_rows =
        makeTensor<float>({{1, 2, 3}, {4, 5, 6}});

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
    const Tensor<float> matrix =
        makeTensor<float>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    const Tensor<float> single_row = makeTensor<float>({{-1, 2, 4}});

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
    const Tensor<float> tensor = makeTensor<float>({{1, 2.5f}, {-3, 4}});
    const std::string output = tensor.toString();

    expectTrue(output.find("shape=[2, 2]") != std::string::npos,
        "tensor string contains shape");
    expectTrue(output.find("[1, 2.5]") != std::string::npos,
        "tensor string contains first row");
    expectTrue(output.find("[-3, 4]") != std::string::npos,
        "tensor string contains second row");
}

DNP_TEST(
    tensor_reshape_preserves_values_in_2d_3d_and_4d
) {
    Tensor<double> tensor({4, 6});
    for (std::size_t i = 0; i < tensor.size(); i++) {
        tensor.set(i, static_cast<double>(i) - 7.5);
    }
    const Tensor<double> original = tensor;
    for (const Shapes shape :
        {Shapes{3, 8}, Shapes{2, 3, 4}, Shapes{2, 2, 2, 3}}) {
        tensor.reshape(shape);
        expectTrue(tensor.shapes() == shape, "reshape dimensions");
        expectEqual(tensor.size(), original.size(), "reshape size");
        for (std::size_t i = 0; i < tensor.size(); i++) {
            expectNear(tensor.get(i), original.get(i), 1e-12, "reshape values");
        }
    }
    expectTrue(original.shapes() == Shapes{4, 6}, "reshape copy independence");
}

DNP_TEST(
    tensor_reshape_rejects_invalid_shapes_without_mutation
) {
    Tensor<float> tensor = makeTensor<float>({{1, 2, 3}, {4, 5, 6}});
    expectThrows<std::runtime_error>([&]() { tensor.reshape({2, 4}); },
        "reshape incompatible size");
    expectThrows<std::invalid_argument>([&]() { tensor.reshape({}); },
        "reshape empty shape");
    expectThrows<std::invalid_argument>([&]() { tensor.reshape({2, 0, 3}); },
        "reshape zero dimension");
    expectThrows<std::overflow_error>(
        [&]() { tensor.reshape({0xffffffffu, 0xffffffffu, 0xffffffffu}); },
        "reshape dimension product overflow"
    );
    expectTensorNear(tensor,
        2,
        3,
        {1, 2, 3, 4, 5, 6},
        1e-6,
        "failed reshape leaves tensor unchanged");
}

DNP_TEST(
    tensor_transpose_swaps_last_axes_in_2d_3d_and_4d
) {
    for (const Shapes shape :
        {Shapes{2, 3}, Shapes{2, 2, 3}, Shapes{2, 3, 2, 3}}) {
        Tensor<double> input(shape);
        for (std::size_t i = 0; i < input.size(); i++) {
            input.set(i, static_cast<double>(i) + 0.5);
        }
        const auto result = input.transpose();
        Shapes expected_shape = shape;
        expected_shape[shape.size() - 2] = 3;
        expected_shape.back() = 2;
        expectTrue(result.shapes() == expected_shape, "transpose dimensions");
        expectEqual(result.size(), input.size(), "transpose size");
        const double expected[] = {0.5, 3.5, 1.5, 4.5, 2.5, 5.5};
        for (std::size_t i = 0; i < result.size(); i++) {
            expectNear(result.get(i),
                expected[i % 6] + static_cast<double>(i / 6) * 6,
                1e-12,
                "transposed batch values");
            expectNear(input.get(i),
                static_cast<double>(i) + 0.5,
                1e-12,
                "transpose preserves source values");
        }
        expectTrue(input.shapes() == shape, "transpose preserves source shape");
        const auto restored = result.transpose();
        expectTrue(restored.shapes() == shape, "double transpose shape");
        for (std::size_t i = 0; i < input.size(); i++) {
            expectNear(restored.get(i),
                input.get(i),
                1e-12,
                "double transpose values");
        }
    }
    const Tensor<double> vector({3});
    expectThrows<std::runtime_error>([&]() { (void)vector.transpose(); },
        "transpose rejects 1D tensor");
}

DNP_TEST(
    tensor_batched_matmul_matches_2d_for_all_transpose_modes
) {
    for (const Shapes batch_shape : {Shapes{2}, Shapes{2, 3}}) {
        for (const auto left_op : {MatrixOp::None, MatrixOp::Transpose}) {
            for (const auto right_op : {MatrixOp::None, MatrixOp::Transpose}) {
                Shapes left_shape = batch_shape;
                Shapes right_shape = batch_shape;
                const bool left_transposed = left_op == MatrixOp::Transpose;
                const bool right_transposed = right_op == MatrixOp::Transpose;
                left_shape.push_back(left_transposed ? 3 : 2);
                left_shape.push_back(left_transposed ? 2 : 3);
                right_shape.push_back(right_transposed ? 4 : 3);
                right_shape.push_back(right_transposed ? 3 : 4);
                Tensor<double> left(left_shape);
                Tensor<double> right(right_shape);
                for (std::size_t i = 0; i < left.size(); i++) {
                    left.set(i, static_cast<double>(i) * 0.25 - 2.0);
                }
                for (std::size_t i = 0; i < right.size(); i++) {
                    right.set(i, 3.0 - static_cast<double>(i) * 0.125);
                }
                const Tensor<double> original_left = left;
                const Tensor<double> original_right = right;
                const auto result =
                    left.batched_matmul(right, left_op, right_op);
                Shapes expected_shape = batch_shape;
                expected_shape.push_back(2);
                expected_shape.push_back(4);
                expectTrue(result.shapes() == expected_shape,
                    "batched product shape");
                expectEqual(result.size(),
                    left.size() / 6 * 8,
                    "batched product size");
                for (std::size_t batch = 0; batch < left.size() / 6; batch++) {
                    Tensor<double> a(
                        {left_shape[left_shape.size() - 2], left_shape.back()}
                    );
                    Tensor<double> b(
                        {right_shape[right_shape.size() - 2],
                            right_shape.back()}
                    );
                    for (std::size_t i = 0; i < a.size(); i++)
                        a.set(i, left.get(batch * 6 + i));
                    for (std::size_t i = 0; i < b.size(); i++)
                        b.set(i, right.get(batch * 12 + i));
                    const auto expected = a.matmul(b, left_op, right_op);
                    for (std::size_t i = 0; i < expected.size(); i++) {
                        expectNear(result.get(batch * 8 + i),
                            expected.get(i),
                            1e-12,
                            "batched product equals individual matmul");
                    }
                }
                expectTrue(left.shapes() == original_left.shapes(),
                    "left shape unchanged");
                expectTrue(right.shapes() == original_right.shapes(),
                    "right shape unchanged");
                for (std::size_t i = 0; i < left.size(); i++) {
                    expectNear(left.get(i),
                        original_left.get(i),
                        1e-12,
                        "left values unchanged");
                }
                for (std::size_t i = 0; i < right.size(); i++) {
                    expectNear(right.get(i),
                        original_right.get(i),
                        1e-12,
                        "right values unchanged");
                }
            }
        }
    }
}

DNP_TEST(
    tensor_batched_matmul_rejects_incompatible_dimensions
) {
    const Tensor<float> left({2, 2, 3});
    for (const Shapes shape :
        {Shapes{2, 4, 2}, Shapes{3, 3, 2}, Shapes{1, 2, 3, 2}, Shapes{3, 2}}) {
        const Tensor<float> right(shape);
        expectThrows<std::invalid_argument>(
            [&]() {
                (void)left.batched_matmul(right,
                    MatrixOp::None,
                    MatrixOp::None);
            },
            "invalid batched product dimensions"
        );
    }
    const Tensor<float> left4d({2, 3, 2, 3});
    const Tensor<float> wrong_batches({3, 2, 3, 4});
    expectThrows<std::invalid_argument>(
        [&]() {
            (void)left4d.batched_matmul(wrong_batches,
                MatrixOp::None,
                MatrixOp::None);
        },
        "equal batch count with different batch axes"
    );
    const Tensor<float> right({2, 3, 4});
    expectThrows<std::invalid_argument>(
        [&]() {
            (void)left.batched_matmul(right,
                MatrixOp::Transpose,
                MatrixOp::None);
        },
        "incompatible transposed left"
    );
    expectThrows<std::invalid_argument>(
        [&]() {
            (void)left.batched_matmul(right,
                MatrixOp::None,
                MatrixOp::Transpose);
        },
        "incompatible transposed right"
    );
}

} // namespace
