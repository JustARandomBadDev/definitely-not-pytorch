#include "test.hpp"

#include <cstddef>
#include <string>

#include "dnp/neural_network.hpp"
#include "dnp/training/loss/mse.hpp"
#include "dnp/training/optimizers/sgd.hpp"
#include "dnp/training/trainer.hpp"

namespace {

using dnp::NeuralNetwork;
using dnp::Tensor;
using dnp::test::expectEqual;
using dnp::test::expectNear;
using dnp::test::expectTensorNear;
using dnp::test::expectThrows;
using dnp::training::MeanSquaredError;
using dnp::training::SGD;
using dnp::training::Trainer;

template <typename T> NeuralNetwork<T> makeTrainingNetwork() {
    NeuralNetwork<T> network{{2}, {1}};
    auto& layer = network.getLayers()[0];

    layer.weights = {{static_cast<T>(0.25)}, {static_cast<T>(-0.5)}};
    layer.biais = {{static_cast<T>(0.1)}};

    return network;
}

template <typename T>
void backwardMeanSquaredError(
    NeuralNetwork<T>& p_network,
    const Tensor<T>& p_input,
    const Tensor<T>& p_target,
    const MeanSquaredError<T>& p_loss
) {
    const Tensor<T> prediction = p_network.forward(p_input);
    const Tensor<T> gradient = p_loss.gradient(prediction, p_target);
    p_network.backward(p_input, gradient);
}

template <typename T>
void expectParametersNear(
    const NeuralNetwork<T>& p_actual,
    const NeuralNetwork<T>& p_expected,
    T p_tolerance,
    const std::string& p_context
) {
    expectEqual(p_actual.getLayers().size(),
        p_expected.getLayers().size(),
        p_context + " layer count");

    for (std::size_t layer_index = 0; layer_index < p_actual.getLayers().size();
        layer_index++) {
        const auto& actual = p_actual.getLayers()[layer_index];
        const auto& expected = p_expected.getLayers()[layer_index];

        expectEqual(actual.weights.size(),
            expected.weights.size(),
            p_context + " weight count");
        expectEqual(actual.biais.size(),
            expected.biais.size(),
            p_context + " bias count");

        for (std::size_t index = 0; index < actual.weights.size(); index++) {
            expectNear(actual.weights.data()[index],
                expected.weights.data()[index],
                p_tolerance,
                p_context + " weight");
        }

        for (std::size_t index = 0; index < actual.biais.size(); index++) {
            expectNear(actual.biais.data()[index],
                expected.biais.data()[index],
                p_tolerance,
                p_context + " bias");
        }
    }
}

DNP_TEST(
    training_sgd_applies_expected_parameter_update
) {
    NeuralNetwork<float> network{{2}, {1}};
    const Tensor<float> input{{1, 2}, {3, 4}};
    const Tensor<float> target{{1}, {2}};
    SGD<float> optimizer(0.1f);
    MeanSquaredError<float> mse;

    backwardMeanSquaredError(network, input, target, mse);
    optimizer.step(network);

    const auto& layer = network.getLayers()[0];
    expectNear(layer.weights.get(0, 0), 0.7f, 1e-6, "SGD first weight update");
    expectNear(layer.weights.get(1, 0), 1.0f, 1e-6, "SGD second weight update");
    expectNear(layer.biais.get(0, 0), 0.3f, 1e-6, "SGD bias update");
}

DNP_TEST(
    training_sgd_scales_updates_with_learning_rate
) {
    NeuralNetwork<float> slow_network{{2}, {1}};
    NeuralNetwork<float> fast_network = slow_network;
    const Tensor<float> input{{1, 2}, {3, 4}};
    const Tensor<float> target{{1}, {2}};
    SGD<float> slow(0.1f);
    SGD<float> fast(0.2f);
    MeanSquaredError<float> mse;

    backwardMeanSquaredError(slow_network, input, target, mse);
    backwardMeanSquaredError(fast_network, input, target, mse);
    slow.step(slow_network);
    fast.step(fast_network);

    const auto& slow_layer = slow_network.getLayers()[0];
    const auto& fast_layer = fast_network.getLayers()[0];

    expectNear(fast_layer.weights.get(0, 0),
        2.0f * slow_layer.weights.get(0, 0),
        1e-6,
        "learning rate scales first weight displacement");
    expectNear(fast_layer.weights.get(1, 0),
        2.0f * slow_layer.weights.get(1, 0),
        1e-6,
        "learning rate scales second weight displacement");
    expectNear(fast_layer.biais.get(0, 0),
        2.0f * slow_layer.biais.get(0, 0),
        1e-6,
        "learning rate scales bias displacement");
}

DNP_TEST(
    training_trainer_single_step_matches_manual_training
) {
    NeuralNetwork<float> manual = makeTrainingNetwork<float>();
    NeuralNetwork<float> trained = manual;
    const Tensor<float> input{{0.5f, -1.0f}, {2.0f, 0.25f}};
    const Tensor<float> target{{0.75f}, {-0.5f}};
    SGD<float> manual_optimizer(0.05f);
    SGD<float> trainer_optimizer(0.05f);
    MeanSquaredError<float> mse;
    Trainer<float> trainer(trained, trainer_optimizer, mse);

    backwardMeanSquaredError(manual, input, target, mse);
    manual_optimizer.step(manual);
    trainer.train(input, target);

    expectParametersNear(trained,
        manual,
        1e-6f,
        "Trainer single-step delegation");
}

DNP_TEST(
    training_trainer_multiple_epochs_matches_manual_loop
) {
    NeuralNetwork<float> trainer_network = makeTrainingNetwork<float>();
    NeuralNetwork<float> manual_network = trainer_network;
    const Tensor<float> input{{0.5f, -1.0f}, {2.0f, 0.25f}};
    const Tensor<float> target{{0.75f}, {-0.5f}};
    SGD<float> trainer_optimizer(0.05f);
    SGD<float> manual_optimizer(0.05f);
    MeanSquaredError<float> mse;
    Trainer<float> trainer(trainer_network, trainer_optimizer, mse);
    constexpr std::size_t epochs = 5;

    trainer.train(input, target, epochs);

    for (std::size_t epoch = 0; epoch < epochs; epoch++) {
        backwardMeanSquaredError(manual_network, input, target, mse);
        manual_optimizer.step(manual_network);
    }

    expectParametersNear(trainer_network,
        manual_network,
        1e-6f,
        "Trainer multi-epoch delegation");
}

DNP_TEST(
    training_supports_double_parameters
) {
    NeuralNetwork<double> network{{2}, {1}};
    auto& layer = network.getLayers()[0];
    layer.weights = {{0.5}, {-0.25}};
    layer.biais = {{0.1}};

    const Tensor<double> input{{2.0, -1.0}};
    const Tensor<double> target{{3.0}};
    SGD<double> optimizer(0.1);
    MeanSquaredError<double> mse;
    Trainer<double> trainer(network, optimizer, mse);

    trainer.train(input, target);

    expectNear(layer.weights.get(0, 0),
        1.16,
        1e-12,
        "double first weight update");
    expectNear(layer.weights.get(1, 0),
        -0.58,
        1e-12,
        "double second weight update");
    expectNear(layer.biais.get(0, 0), 0.43, 1e-12, "double bias update");
}

DNP_TEST(
    training_mse_computes_values_and_rejects_invalid_inputs
) {
    const MeanSquaredError<float> mse;
    const Tensor<float> prediction{{1, 3}};
    const Tensor<float> target{{0, 1}};

    expectNear(mse.compute(prediction, target), 2.5f, 1e-6, "MSE value");
    expectTensorNear(mse.gradient(prediction, target),
        1,
        2,
        {1, 2},
        1e-6,
        "MSE gradient");

    const Tensor<float> different_shape{{0}, {1}};
    expectThrows<std::invalid_argument>(
        [&]() { (void)mse.compute(prediction, different_shape); },
        "MSE compute shape mismatch",
        "shapes must match"
    );
    expectThrows<std::invalid_argument>(
        [&]() { (void)mse.gradient(prediction, different_shape); },
        "MSE gradient shape mismatch",
        "shapes must match"
    );

    const Tensor<float> empty(dnp::Shape{0, 1});
    expectThrows<std::invalid_argument>(
        [&]() { (void)mse.compute(empty, empty); },
        "MSE compute empty tensors",
        "cannot be empty"
    );
    expectThrows<std::invalid_argument>(
        [&]() { (void)mse.gradient(empty, empty); },
        "MSE gradient empty tensors",
        "cannot be empty"
    );
}

} // namespace
