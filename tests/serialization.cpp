#include "test.hpp"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "dnp/neural_network.hpp"
#include "dnp/serialization/model.hpp"

namespace {

using dnp::NeuralNetwork;
using dnp::None;
using dnp::Tan;
using dnp::Tensor;
using dnp::serialization::loadModel;
using dnp::serialization::saveModel;
using dnp::test::expectEqual;
using dnp::test::expectNear;
using dnp::test::expectShape;
using dnp::test::expectTensorNear;
using dnp::test::expectThrows;
using dnp::test::TempDirectory;

template <typename T> NeuralNetwork<T> makeModel() {
    NeuralNetwork<T> network{{2}, {3, Tan}, {1, None}};
    auto& layers = network.getLayers();

    layers[0].weights = {
        {static_cast<T>(0.1), static_cast<T>(-0.2), static_cast<T>(0.3)},
        {static_cast<T>(0.4), static_cast<T>(0.5), static_cast<T>(-0.6)}
    };
    layers[0].biais = {
        {static_cast<T>(0.01), static_cast<T>(-0.02), static_cast<T>(0.03)}
    };
    layers[1].weights =
        {{static_cast<T>(0.7)}, {static_cast<T>(-0.8)}, {static_cast<T>(0.9)}};
    layers[1].biais = {{static_cast<T>(-0.04)}};

    return network;
}

template <typename T>
void expectModelsEqual(
    const NeuralNetwork<T>& p_expected,
    const NeuralNetwork<T>& p_actual,
    T p_tolerance,
    const std::string& p_context
) {
    expectEqual(p_actual.getInputSize(),
        p_expected.getInputSize(),
        p_context + " input size");
    expectEqual(p_actual.getLayers().size(),
        p_expected.getLayers().size(),
        p_context + " layer count");

    for (std::size_t layer_index = 0;
        layer_index < p_expected.getLayers().size();
        layer_index++) {
        const auto& expected_layer = p_expected.getLayers()[layer_index];
        const auto& actual_layer = p_actual.getLayers()[layer_index];

        expectEqual(static_cast<int>(actual_layer.activation),
            static_cast<int>(expected_layer.activation),
            p_context + " activation");
        expectShape(actual_layer.weights,
            expected_layer.weights.shape().rows,
            expected_layer.weights.shape().cols,
            p_context + " weights");
        expectShape(actual_layer.biais,
            expected_layer.biais.shape().rows,
            expected_layer.biais.shape().cols,
            p_context + " bias");

        for (std::size_t index = 0; index < expected_layer.weights.size();
            index++) {
            expectNear(actual_layer.weights.data()[index],
                expected_layer.weights.data()[index],
                p_tolerance,
                p_context + " weight value");
        }

        for (std::size_t index = 0; index < expected_layer.biais.size();
            index++) {
            expectNear(actual_layer.biais.data()[index],
                expected_layer.biais.data()[index],
                p_tolerance,
                p_context + " bias value");
        }
    }
}

void writeFile(
    const std::filesystem::path& p_path,
    const std::string& p_contents
) {
    std::ofstream file(p_path, std::ios::binary);
    file << p_contents;

    if (!file) throw std::runtime_error("Failed to create malformed test file");
}

DNP_TEST(
    serialization_float_model_round_trip
) {
    TempDirectory directory("float-round-trip");
    const auto path = directory.path() / "model.dnp";
    NeuralNetwork<float> original = makeModel<float>();
    saveModel(original, path);
    NeuralNetwork<float> loaded = loadModel<float>(path);

    expectModelsEqual(original, loaded, 0.0f, "float round trip");

    Tensor<float> input{{0.5f, -0.25f}, {1.0f, 2.0f}};
    const Tensor<float> original_output = original.forward(input);
    const Tensor<float> loaded_output = loaded.forward(input);

    for (std::size_t row = 0; row < input.shape().rows; row++) {
        expectNear(loaded_output.get(row, 0),
            original_output.get(row, 0),
            1e-6,
            "float round-trip inference");
    }
}

DNP_TEST(
    serialization_double_model_round_trip
) {
    TempDirectory directory("double-round-trip");
    const auto path = directory.path() / "model.dnp";
    NeuralNetwork<double> original = makeModel<double>();
    saveModel(original, path);
    NeuralNetwork<double> loaded = loadModel<double>(path);

    expectModelsEqual(original, loaded, 0.0, "double round trip");

    Tensor<double> input{{-0.75, 1.25}};
    const Tensor<double> original_output = original.forward(input);
    const Tensor<double> loaded_output = loaded.forward(input);
    expectNear(loaded_output.get(0, 0),
        original_output.get(0, 0),
        1e-12,
        "double round-trip inference");
}

DNP_TEST(
    serialization_loaded_model_supports_changing_batch_sizes
) {
    TempDirectory directory("loaded-batches");
    const auto path = directory.path() / "model.dnp";
    NeuralNetwork<float> original = makeModel<float>();
    saveModel(original, path);
    NeuralNetwork<float> loaded = loadModel<float>(path);

    Tensor<float> batch_four{{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    Tensor<float> batch_one{{0.25f, -0.5f}};
    const Tensor<float> first_four = loaded.forward(batch_four);
    const Tensor<float> one = loaded.forward(batch_one);
    const Tensor<float> second_four = loaded.forward(batch_four);

    expectShape(first_four, 4, 1, "loaded model batch four");
    expectShape(one, 1, 1, "loaded model batch one");
    expectShape(second_four, 4, 1, "loaded model second batch four");

    for (std::size_t row = 0; row < batch_four.shape().rows; row++) {
        expectNear(second_four.get(row, 0),
            first_four.get(row, 0),
            1e-6,
            "loaded model repeated batch result");
    }

    expectShape(loaded.getLayers()[0].weights,
        2,
        3,
        "loaded model unchanged first weight shape");
    expectShape(loaded.getLayers()[0].biais,
        1,
        3,
        "loaded model unchanged first bias shape");
    expectShape(loaded.getLayers()[1].weights,
        3,
        1,
        "loaded model unchanged output weight shape");
    expectShape(loaded.getLayers()[1].biais,
        1,
        1,
        "loaded model unchanged output bias shape");
}

DNP_TEST(
    serialization_rejects_type_mismatch
) {
    TempDirectory directory("type-mismatch");
    const auto path = directory.path() / "model.dnp";
    const NeuralNetwork<float> model = makeModel<float>();
    saveModel(model, path);

    expectThrows<std::runtime_error>([&]() { (void)loadModel<double>(path); },
        "serialization type mismatch",
        "data type mismatch");
}

DNP_TEST(
    serialization_rejects_invalid_magic
) {
    TempDirectory directory("invalid-magic");
    const auto path = directory.path() / "model.dnp";
    writeFile(path, "NOT_DNP\ntype f32\ninput 2\nlayer 1 none\nDATA\n");

    expectThrows<std::runtime_error>([&]() { (void)loadModel<float>(path); },
        "invalid model magic",
        "Invalid DNP model file");
}

DNP_TEST(
    serialization_rejects_unknown_activation
) {
    TempDirectory directory("unknown-activation");
    const auto path = directory.path() / "model.dnp";
    writeFile(path, "DNP\ntype f32\ninput 2\nlayer 1 swish\nDATA\n");

    expectThrows<std::runtime_error>([&]() { (void)loadModel<float>(path); },
        "unknown serialized activation",
        "Unknown activation");
}

DNP_TEST(
    serialization_rejects_missing_data_section
) {
    TempDirectory directory("missing-data");
    const auto path = directory.path() / "model.dnp";
    writeFile(path, "DNP\ntype f32\ninput 2\nlayer 1 none\n");

    expectThrows<std::runtime_error>([&]() { (void)loadModel<float>(path); },
        "missing DATA section",
        "Missing DATA section");
}

DNP_TEST(
    serialization_rejects_truncated_binary_payload
) {
    TempDirectory directory("truncated-payload");
    const auto path = directory.path() / "model.dnp";
    const NeuralNetwork<float> model = makeModel<float>();
    saveModel(model, path);

    const auto size = std::filesystem::file_size(path);
    std::filesystem::resize_file(path, size - 1);

    expectThrows<std::runtime_error>([&]() { (void)loadModel<float>(path); },
        "truncated model payload",
        "Failed to read");
}

DNP_TEST(
    serialization_survives_two_save_load_cycles
) {
    TempDirectory directory("regression-cycle");
    const auto first_path = directory.path() / "first.dnp";
    const auto second_path = directory.path() / "second.dnp";
    NeuralNetwork<float> original = makeModel<float>();

    saveModel(original, first_path);
    NeuralNetwork<float> first_loaded = loadModel<float>(first_path);
    saveModel(first_loaded, second_path);
    NeuralNetwork<float> second_loaded = loadModel<float>(second_path);

    expectModelsEqual(original,
        second_loaded,
        0.0f,
        "two serialization cycles");

    Tensor<float> input{{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    const Tensor<float> original_output = original.forward(input);
    const Tensor<float> loaded_output = second_loaded.forward(input);

    for (std::size_t row = 0; row < input.shape().rows; row++) {
        expectNear(loaded_output.get(row, 0),
            original_output.get(row, 0),
            1e-6,
            "two-cycle inference");
    }
}

} // namespace
