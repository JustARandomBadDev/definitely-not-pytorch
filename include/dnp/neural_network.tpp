#ifndef DNP_NEURAL_NETWORK_TPP
#define DNP_NEURAL_NETWORK_TPP

#include "dnp/activation.hpp"
#include "dnp/neural_network.hpp"
#include "dnp/tensor.hpp"

#include <cstddef>
#include <random>
#include <sstream>
#include <stdexcept>

namespace dnp {

template <typename T>
NeuralNetwork<T>::NeuralNetwork(
    const std::vector<LayerDescriptor>& p_layers
) {
    initializeLayers(p_layers);
}

template <typename T>
NeuralNetwork<T>::NeuralNetwork(
    std::initializer_list<LayerDescriptor> p_layers
) {
    initializeLayers(std::span<const LayerDescriptor>(p_layers.begin(),
        p_layers.size()));
}

template <typename T>
void NeuralNetwork<T>::initializeLayers(
    std::span<const LayerDescriptor> p_layers
) {
    if (p_layers.size() < 2) {
        throw std::invalid_argument("NeuralNetwork requires an input layer and "
                                    "at least one output layer");
    }

    if (p_layers[0].size == 0) {
        throw std::invalid_argument("NeuralNetwork input size cannot be 0");
    }

    _input_size = p_layers[0].size;

    _layers.reserve(p_layers.size() - 1);
    _train_data.reserve(p_layers.size() - 1);
    _gradients.reserve(p_layers.size() - 1);

    std::size_t prev_layer_size = _input_size;

    for (std::size_t i = 1; i < p_layers.size(); i++) {
        const LayerDescriptor& layer = p_layers[i];

        if (layer.size == 0) {
            throw std::invalid_argument(
                "Cannot initialize NeuralNetwork if a layer has a size of 0"
            );
        }

        addLayer(prev_layer_size, layer.size, layer.activation);

        prev_layer_size = layer.size;
    }
}

template <typename T>
void NeuralNetwork<T>::initialize(
    Initializer p_initializer
) {
    if (p_initializer != Initializer::Random) {
        throw std::invalid_argument("Unknown neural network initializer");
    }

    std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<float> distribution(-1, 1);

    for (Layer& layer : _layers) {
        layer.weights.generate([&]() { return distribution(generator); });
    }
}

template <typename T>
Tensor<T> NeuralNetwork<T>::forward(
    Tensor<T>& p_input
) {
    auto previous = p_input;

    for (std::size_t index = 0; index < _layers.size(); index++) {
        Layer& layer = _layers[index];
        TrainData& train_data = _train_data[index];

        train_data.linear =
            previous.matmul(layer.weights).addRowVector(layer.biais);
        train_data.activated = train_data.linear;
        train_data.activated.map([&](T x) {
            return activate(x, layer.activation);
        });

        previous = train_data.activated;
    }

    return _train_data.back().activated;
}

/*
 * Backward math reminder
 *
 * Start with:
 *
 *     gradient = 2 * (p_y - p_output)
 *
 * For each layer, from last to first:
 *
 *     d_activation = activationDerivative(linear)
 *     d_linear = gradient * d_activation
 *
 *     gradients.biais = d_linear
 *     gradients.weights = previous^T * d_linear
 *
 *     gradient = d_linear * layer.weights^T
 *
 * previous is p_input for the first layer, otherwise the activated output
 * of the previous layer.
 *
 * The last gradient is propagated backward using the chain rule.
 */
template <typename T>
void NeuralNetwork<T>::backward(
    Tensor<T>& p_input,
    Tensor<T>& p_output
) {
    std::size_t batch_size = p_input.shape().rows;

    Tensor<T> p_y = forward(p_input);
    Tensor<T> gradient =
        (p_y - p_output) * (static_cast<T>(2) / static_cast<T>(batch_size));

    for (std::size_t index = _layers.size(); index-- > 0;) {
        Layer& layer = _layers[index];

        Tensor<T> d_activation = _train_data[index].linear;
        d_activation.map([&](T x) {
            return activationDerivative(x, layer.activation);
        });

        Tensor<T> d_linear = gradient.elementwise_mul(d_activation);

        _gradients[index].biais = d_linear.sumRows();

        const Tensor<T>& previous =
            index == 0 ? p_input : _train_data[index - 1].activated;

        _gradients[index].weights =
            previous.matmul(d_linear, MatrixOp::Transpose);

        gradient =
            d_linear.matmul(layer.weights, MatrixOp::None, MatrixOp::Transpose);
    }
}

template <typename T> std::string NeuralNetwork<T>::toString() const {
    std::ostringstream stream;

    stream << "NeuralNetwork(input_size=" << _input_size
           << ", layers=" << _layers.size() << ')';

    for (std::size_t index = 0; index < _layers.size(); index++) {
        const Layer& layer = _layers[index];
        const Shape weights_shape = layer.weights.shape();
        const Shape bias_shape = layer.biais.shape();

        stream << "\n    Layer " << index << ": weights=" << weights_shape.rows
               << 'x' << weights_shape.cols << ", bias=" << bias_shape.rows
               << 'x' << bias_shape.cols
               << ", activation=" << getStringFromActivation(layer.activation);
    }

    return stream.str();
}

template <typename T>
void NeuralNetwork<T>::addLayer(
    std::size_t p_prev_layer_size,
    std::size_t p_size,
    Activation p_activation
) {
    _layers.push_back(Layer{
        Tensor<T>(Shape{p_prev_layer_size, p_size}),
        Tensor<T>(Shape{1, p_size}),
        p_activation
    });

    _train_data.push_back(
        TrainData{Tensor<T>(Shape{1, p_size}), Tensor<T>(Shape{1, p_size})}
    );

    _gradients.push_back(Gradient{
        Tensor<T>(Shape{p_prev_layer_size, p_size}),
        Tensor<T>(Shape{1, p_size}),
    });
}

} // namespace dnp

#endif
