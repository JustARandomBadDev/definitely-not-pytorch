#ifndef DNP_MODEL_TPP
#define DNP_MODEL_TPP

#include "dnp/activation.hpp"
#include "dnp/neural_network.hpp"
#include "dnp/serialization/model.hpp"

#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace dnp::serialization {

template <typename T>
void saveModel(
    const NeuralNetwork<T>& p_network,
    const std::filesystem::path& p_path
) {
    std::ofstream file(p_path, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + p_path.string());
    }

    file << "DNP\n";

    if constexpr (std::is_same_v<T, float>) {
        file << "type f32\n";
    } else if constexpr (std::is_same_v<T, double>) {
        file << "type f64\n";
    } else {
        throw std::runtime_error("Unsupported DNP model data type");
    }

    file << "input " << p_network.getInputSize() << '\n';

    for (const typename NeuralNetwork<T>::Layer& layer :
        p_network.getLayers()) {
        file << "layer " << layer.weights.shape().cols << ' '
             << getStringFromActivation(layer.activation) << '\n';
    }

    file << "DATA\n";

    for (const typename NeuralNetwork<T>::Layer& layer :
        p_network.getLayers()) {
        const std::size_t weights_size = layer.weights.size() * sizeof(T);

        const std::size_t biais_size = layer.biais.size() * sizeof(T);

        file.write(reinterpret_cast<const char*>(layer.weights.data()),
            static_cast<std::streamsize>(weights_size));

        file.write(reinterpret_cast<const char*>(layer.biais.data()),
            static_cast<std::streamsize>(biais_size));
    }

    if (!file) {
        throw std::runtime_error("Failed to write model: " + p_path.string());
    }
}

template <typename T>
NeuralNetwork<T> loadModel(
    const std::filesystem::path& p_path
) {
    std::ifstream file(p_path, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + p_path.string());
    }

    std::string token;

    file >> token;
    if (token != "DNP") throw std::runtime_error("Invalid DNP model file");

    file >> token;
    if (token != "type") throw std::runtime_error("Expected model data type");

    std::string data_type;
    file >> data_type;

    if constexpr (std::is_same_v<T, float>) {
        if (data_type != "f32")
            throw std::runtime_error("DNP model data type mismatch");
    } else if constexpr (std::is_same_v<T, double>) {
        if (data_type != "f64")
            throw std::runtime_error("DNP model data type mismatch");
    } else
        throw std::runtime_error("Unsupported DNP model data type");

    file >> token;
    if (token != "input") throw std::runtime_error("Expected model input size");

    std::size_t input_size = 0;
    file >> input_size;

    std::vector<LayerDescriptor> layers;

    layers.push_back({input_size, Activation::None});

    while (file >> token) {
        if (token == "DATA") {
            char newline;
            file.get(newline);

            if (newline != '\n')
                throw std::runtime_error("Invalid DATA delimiter");

            break;
        }

        if (token != "layer")
            throw std::runtime_error("Unexpected token in DNP model: " + token);

        std::size_t layer_size = 0;
        std::string activation_string;

        file >> layer_size >> activation_string;

        layers.push_back(
            {layer_size, getActivationFromString(activation_string)}
        );
    }

    if (token != "DATA")
        throw std::runtime_error("Missing DATA section in DNP model");

    NeuralNetwork<T> network(layers);

    for (typename NeuralNetwork<T>::Layer& layer : network.getLayers()) {
        const std::size_t weights_size = layer.weights.size() * sizeof(T);

        const std::size_t biais_size = layer.biais.size() * sizeof(T);

        file.read(reinterpret_cast<char*>(layer.weights.data()),
            static_cast<std::streamsize>(weights_size));

        if (!file) throw std::runtime_error("Failed to read layer weights");

        file.read(reinterpret_cast<char*>(layer.biais.data()),
            static_cast<std::streamsize>(biais_size));

        if (!file) throw std::runtime_error("Failed to read layer biais");
    }

    return network;
}

} // namespace dnp::serialization

#endif