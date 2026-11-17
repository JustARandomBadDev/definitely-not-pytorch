#include "test.hpp"

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "dnp/neural_network.hpp"
#include "dnp/training/loss/mse.hpp"
#include "dnp/training/optimizers/sgd.hpp"
#include "dnp/training/trainer.hpp"

namespace {

using dnp::Activation;
using dnp::LayerDescriptor;
using dnp::NeuralNetwork;
using dnp::None;
using dnp::ReLu;
using dnp::Tan;
using dnp::Tensor;
using dnp::test::expectEqual;
using dnp::test::expectNear;
using dnp::test::expectThrows;
using dnp::test::expectTrue;
using dnp::training::MeanSquaredError;
using dnp::training::SGD;
using dnp::training::Trainer;

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

template <typename T>
void expectArchitecturesEqual(
    const NeuralNetwork<T>& p_left,
    const NeuralNetwork<T>& p_right,
    const std::string& p_context
) {
    expectEqual(p_left.getInputSize(),
        p_right.getInputSize(),
        p_context + " input size");
    expectEqual(p_left.getLayers().size(),
        p_right.getLayers().size(),
        p_context + " layer count");

    for (std::size_t index = 0; index < p_left.getLayers().size(); index++) {
        const auto& left = p_left.getLayers()[index];
        const auto& right = p_right.getLayers()[index];

        expectEqual(left.weights.shapes()[0],
            right.weights.shapes()[0],
            p_context + " weight rows");
        expectEqual(left.weights.shapes()[1],
            right.weights.shapes()[1],
            p_context + " weight columns");
        expectEqual(left.biais.shapes()[0],
            right.biais.shapes()[0],
            p_context + " bias rows");
        expectEqual(left.biais.shapes()[1],
            right.biais.shapes()[1],
            p_context + " bias columns");
        expectEqual(static_cast<int>(left.activation),
            static_cast<int>(right.activation),
            p_context + " activation");
    }
}

NeuralNetwork<float> makeLinearNetwork() {
    NeuralNetwork<float> network{{2}, {2, None}, {1, None}};
    auto& layers = network.getLayers();

    layers[0].weights = makeTensor<float>({{1, 0}, {0, 1}});
    layers[0].biais = makeTensor<float>({{0, 0}});
    layers[1].weights = makeTensor<float>({{2}, {-1}});
    layers[1].biais = makeTensor<float>({{0.5f}});

    return network;
}

NeuralNetwork<float> makeGradientNetwork() {
    NeuralNetwork<float> network{{2}, {2, Tan}, {1, None}};
    auto& layers = network.getLayers();

    layers[0].weights = makeTensor<float>({{0.2f, -0.3f}, {0.4f, 0.1f}});
    layers[0].biais = makeTensor<float>({{0.05f, -0.02f}});
    layers[1].weights = makeTensor<float>({{0.7f}, {-0.5f}});
    layers[1].biais = makeTensor<float>({{0.1f}});

    return network;
}

float meanSquaredLoss(
    NeuralNetwork<float>& p_network,
    const Tensor<float>& p_input,
    const Tensor<float>& p_target
) {
    const Tensor<float> prediction = p_network.forward(p_input);
    float loss = 0;

    for (std::size_t row = 0; row < prediction.shapes()[0]; row++) {
        for (std::size_t col = 0; col < prediction.shapes()[1]; col++) {
            const float difference =
                get2D(prediction, row, col) - get2D(p_target, row, col);
            loss += difference * difference;
        }
    }

    return loss / static_cast<float>(prediction.size());
}

void backwardMeanSquaredError(
    NeuralNetwork<float>& p_network,
    const Tensor<float>& p_input,
    const Tensor<float>& p_target
) {
    const MeanSquaredError<float> mse;
    const Tensor<float> prediction = p_network.forward(p_input);
    const Tensor<float> gradient = mse.gradient(prediction, p_target);
    p_network.backward(p_input, gradient);
}

DNP_TEST(
    neural_network_constructors_create_equivalent_architectures
) {
    const NeuralNetwork<float> from_list{{2}, {3, Tan}, {2, ReLu}, {1}};
    const std::vector<LayerDescriptor> descriptors =
        {{2}, {3, Tan}, {2, ReLu}, {1}};
    const NeuralNetwork<float> from_vector(descriptors);

    expectArchitecturesEqual(from_list, from_vector, "network constructors");
    expectEqual(from_list.getInputSize(), std::size_t{2}, "network input size");
    expectEqual(from_list.getLayers().size(),
        std::size_t{3},
        "network layer count");

    expectShape(from_list.getLayers()[0].weights, 2, 3, "first layer weights");
    expectShape(from_list.getLayers()[0].biais, 1, 3, "first layer bias");
    expectShape(from_list.getLayers()[1].weights, 3, 2, "second layer weights");
    expectShape(from_list.getLayers()[1].biais, 1, 2, "second layer bias");
    expectShape(from_list.getLayers()[2].weights, 2, 1, "output layer weights");
    expectShape(from_list.getLayers()[2].biais, 1, 1, "output layer bias");
    expectEqual(static_cast<int>(from_list.getLayers()[0].activation),
        static_cast<int>(Tan),
        "first layer activation");
    expectEqual(static_cast<int>(from_list.getLayers()[1].activation),
        static_cast<int>(ReLu),
        "second layer activation");
    expectEqual(static_cast<int>(from_list.getLayers()[2].activation),
        static_cast<int>(None),
        "output layer activation");
}

DNP_TEST(
    neural_network_rejects_invalid_architectures
) {
    expectThrows<std::invalid_argument>(
        []() {
            NeuralNetwork<float> network(
                std::initializer_list<LayerDescriptor>{}
            );
            (void)network;
        },
        "network with no layers"
    );
    expectThrows<std::invalid_argument>(
        []() {
            NeuralNetwork<float> network{{2}};
            (void)network;
        },
        "network with only input layer"
    );
    expectThrows<std::invalid_argument>(
        []() {
            NeuralNetwork<float> network{{0}, {1}};
            (void)network;
        },
        "network with zero input size"
    );
    expectThrows<std::invalid_argument>(
        []() {
            NeuralNetwork<float> network{{2}, {0, Tan}, {1}};
            (void)network;
        },
        "network with zero hidden size"
    );
    expectThrows<std::invalid_argument>(
        []() {
            NeuralNetwork<float> network{{2}, {0}};
            (void)network;
        },
        "network with zero output size"
    );
}

DNP_TEST(
    neural_network_forward_supports_changing_batch_sizes
) {
    NeuralNetwork<float> network = makeLinearNetwork();
    Tensor<float> batch_four =
        makeTensor<float>({{0, 0}, {1, 0}, {0, 1}, {1, 1}});
    Tensor<float> batch_one = makeTensor<float>({{2, 3}});

    const Tensor<float> first_four = network.forward(batch_four);
    const Tensor<float> one = network.forward(batch_one);
    const Tensor<float> second_four = network.forward(batch_four);

    expectTensorNear(first_four,
        4,
        1,
        {0.5, 2.5, -0.5, 1.5},
        1e-6,
        "first batch-four forward");
    expectTensorNear(one, 1, 1, {1.5}, 1e-6, "batch-one forward");
    expectTensorNear(second_four,
        4,
        1,
        {0.5, 2.5, -0.5, 1.5},
        1e-6,
        "second batch-four forward");
}

DNP_TEST(
    neural_network_batch_forward_matches_individual_forward_calls
) {
    NeuralNetwork<float> network = makeLinearNetwork();
    Tensor<float> batch =
        makeTensor<float>({{-1, 2}, {0.5f, -3}, {4, 1}, {2, 3}});
    const Tensor<float> batch_result = network.forward(batch);

    for (std::size_t row = 0; row < batch.shapes()[0]; row++) {
        Tensor<float> single =
            makeTensor<float>({{get2D(batch, row, 0), get2D(batch, row, 1)}});
        const Tensor<float> single_result = network.forward(single);
        expectNear(get2D(single_result, 0, 0),
            get2D(batch_result, row, 0),
            1e-6,
            "batch and individual forward equivalence");
    }
}

DNP_TEST(
    neural_network_backward_aggregates_batch_gradients
) {
    NeuralNetwork<float> network{{2}, {1}};
    Tensor<float> input = makeTensor<float>({{1, 2}, {3, 4}});
    Tensor<float> target = makeTensor<float>({{1}, {2}});
    SGD<float> optimizer(0.1f);

    backwardMeanSquaredError(network, input, target);
    optimizer.step(network);

    const auto& layer = network.getLayers()[0];
    expectShape(layer.weights, 2, 1, "weights after batch backward");
    expectShape(layer.biais, 1, 1, "bias after batch backward");
    expectNear(get2D(layer.weights, 0, 0),
        0.7f,
        1e-6,
        "first weight after batch gradient");
    expectNear(get2D(layer.weights, 1, 0),
        1.0f,
        1e-6,
        "second weight after batch gradient");
    expectNear(get2D(layer.biais, 0, 0),
        0.3f,
        1e-6,
        "bias after aggregated batch gradient");

    expectNear((0.0f - get2D(layer.weights, 0, 0)) / 0.1f,
        -7.0f,
        1e-5,
        "inferred first weight gradient");
    expectNear((0.0f - get2D(layer.weights, 1, 0)) / 0.1f,
        -10.0f,
        1e-5,
        "inferred second weight gradient");
    expectNear((0.0f - get2D(layer.biais, 0, 0)) / 0.1f,
        -3.0f,
        1e-5,
        "inferred aggregated bias gradient");
}

DNP_TEST(
    neural_network_gradients_match_finite_differences
) {
    NeuralNetwork<float> base = makeGradientNetwork();
    Tensor<float> input = makeTensor<float>({{0.6f, -0.2f}, {-0.1f, 0.8f}});
    Tensor<float> target = makeTensor<float>({{0.3f}, {-0.4f}});

    NeuralNetwork<float> updated = base;
    SGD<float> optimizer(1.0f);
    backwardMeanSquaredError(updated, input, target);
    optimizer.step(updated);

    const float analytical_hidden_weight =
        get2D(base.getLayers()[0].weights, 0, 0) -
        get2D(updated.getLayers()[0].weights, 0, 0);
    const float analytical_output_weight =
        get2D(base.getLayers()[1].weights, 1, 0) -
        get2D(updated.getLayers()[1].weights, 1, 0);
    const float analytical_hidden_bias =
        get2D(base.getLayers()[0].biais, 0, 1) -
        get2D(updated.getLayers()[0].biais, 0, 1);
    const float analytical_output_bias =
        get2D(base.getLayers()[1].biais, 0, 0) -
        get2D(updated.getLayers()[1].biais, 0, 0);

    const float epsilon = 1e-3f;
    auto numericalGradient = [&](std::size_t p_layer,
                                 bool p_bias,
                                 std::size_t p_row,
                                 std::size_t p_col) {
        NeuralNetwork<float> plus = base;
        NeuralNetwork<float> minus = base;

        Tensor<float>& plus_parameter = p_bias
                                            ? plus.getLayers()[p_layer].biais
                                            : plus.getLayers()[p_layer].weights;
        Tensor<float>& minus_parameter =
            p_bias ? minus.getLayers()[p_layer].biais
                   : minus.getLayers()[p_layer].weights;

        set2D(plus_parameter,
            p_row,
            p_col,
            get2D(plus_parameter, p_row, p_col) + epsilon);
        set2D(minus_parameter,
            p_row,
            p_col,
            get2D(minus_parameter, p_row, p_col) - epsilon);

        const float plus_loss = meanSquaredLoss(plus, input, target);
        const float minus_loss = meanSquaredLoss(minus, input, target);
        return (plus_loss - minus_loss) / (2.0f * epsilon);
    };

    expectNear(analytical_hidden_weight,
        numericalGradient(0, false, 0, 0),
        2e-4,
        "hidden weight numerical gradient");
    expectNear(analytical_output_weight,
        numericalGradient(1, false, 1, 0),
        2e-4,
        "output weight numerical gradient");
    expectNear(analytical_hidden_bias,
        numericalGradient(0, true, 0, 1),
        2e-4,
        "hidden bias numerical gradient");
    expectNear(analytical_output_bias,
        numericalGradient(1, true, 0, 0),
        2e-4,
        "output bias numerical gradient");
}

DNP_TEST(
    neural_network_multi_output_gradient_uses_all_elements
) {
    NeuralNetwork<float> network{{2}, {3, None}};
    const Tensor<float> input = makeTensor<float>({{1, 2}, {3, 4}});
    const Tensor<float> target = makeTensor<float>({{1, 2, 3}, {4, 5, 6}});
    SGD<float> optimizer(0.1f);

    backwardMeanSquaredError(network, input, target);
    optimizer.step(network);

    const auto& layer = network.getLayers()[0];
    expectShape(layer.weights, 2, 3, "multi-output weights");
    expectShape(layer.biais, 1, 3, "multi-output bias");
    expectNear(get2D(layer.weights, 0, 0),
        13.0f / 30.0f,
        1e-6,
        "multi-output normalized weight update");
    expectNear(get2D(layer.biais, 0, 0),
        1.0f / 6.0f,
        1e-6,
        "multi-output normalized bias update");
}

DNP_TEST(
    neural_network_backward_validates_forward_state_and_gradient_shape
) {
    NeuralNetwork<float> network{{2}, {2, None}};
    const Tensor<float> input = makeTensor<float>({{1, 2}, {3, 4}});
    const Tensor<float> invalid_gradient = makeTensor<float>({{1}, {2}});

    expectThrows<std::logic_error>(
        [&]() { network.backward(input, invalid_gradient); },
        "backward without forward",
        "requires a forward pass"
    );

    (void)network.forward(input);

    expectThrows<std::invalid_argument>(
        [&]() { network.backward(input, invalid_gradient); },
        "backward gradient shape mismatch",
        "gradient shape"
    );
}

DNP_TEST(
    neural_network_trains_xor_as_one_batch
) {
    NeuralNetwork<float> network{{2}, {4, Tan}, {1}};
    auto& layers = network.getLayers();
    layers[0].weights = makeTensor<float>(
        {{0.5f, -0.7f, 0.2f, 0.9f}, {0.3f, 0.8f, -0.6f, 0.4f}}
    );
    layers[0].biais = makeTensor<float>({{-0.2f, 0.1f, 0.3f, -0.4f}});
    layers[1].weights = makeTensor<float>({{0.6f}, {-0.4f}, {0.5f}, {-0.3f}});
    layers[1].biais = makeTensor<float>({{0.0f}});

    Tensor<float> input = makeTensor<float>({{0, 0}, {1, 0}, {0, 1}, {1, 1}});
    Tensor<float> target = makeTensor<float>({{0}, {1}, {1}, {0}});

    const float initial_loss = meanSquaredLoss(network, input, target);
    SGD<float> optimizer(0.1f);
    MeanSquaredError<float> mse;
    Trainer<float> trainer(network, optimizer, mse);

    trainer.train(input, target, 3000);

    const Tensor<float> prediction = network.forward(input);
    const float final_loss = meanSquaredLoss(network, input, target);

    expectTrue(final_loss < initial_loss * 0.1f,
        "XOR training significantly reduces loss");
    expectTrue(final_loss < 0.01f, "XOR final loss is small");
    expectTensorNear(prediction, 4, 1, {0, 1, 1, 0}, 0.1, "XOR predictions");
}

DNP_TEST(
    neural_network_to_string_describes_layers
) {
    const NeuralNetwork<float> network{{2}, {3, ReLu}, {1, Tan}};
    const std::string output = network.toString();

    expectTrue(output.find("input_size=2") != std::string::npos,
        "network string input size");
    expectTrue(output.find("weights=2x3, bias=1x3, activation=relu") !=
                   std::string::npos,
        "network string hidden layer");
    expectTrue(output.find("weights=3x1, bias=1x1, activation=tan") !=
                   std::string::npos,
        "network string output layer");
}

} // namespace
