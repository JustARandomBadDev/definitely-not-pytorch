#ifndef DNP_NEURAL_NETWORK_HPP
#define DNP_NEURAL_NETWORK_HPP

#include <cstddef>
#include <dnp/tensor.hpp>

#include "dnp/activation.hpp"

namespace dnp {

enum Initializer { Random };

struct LayerDescriptor {
    std::size_t size = 0;
    Activation activation = Activation::None;
};

template <typename T> class NeuralNetwork {
  public:
    NeuralNetwork(std::initializer_list<LayerDescriptor> p_layers);

    void initialize(Initializer p_initializer);

    Tensor<T> forward(Tensor<T>& p_input);
    void backward(Tensor<T>& p_input, Tensor<T>& p_output);

    void gradientDescent(float p_learning_rate);

  private:
    struct Layer {
        Tensor<T> weights;
        Tensor<T> biais;
        Activation activation;
    };

    struct TrainData {
        Tensor<T> linear;
        Tensor<T> activated;
    };

    struct Gradient {
        Tensor<T> weights;
        Tensor<T> biais;
    };

    std::size_t _size_entry_layer;

    std::vector<Layer> _layers;
    std::vector<TrainData> _train_data;
    std::vector<Gradient> _gradients;

    void addLayer(std::size_t p_prev_layer_size,
        std::size_t p_size,
        Activation p_activation = Activation::None);
};

} // namespace dnp

#ifndef DNP_NEURAL_NETWORK_TPP
#include "dnp/neural_network.tpp"
#endif

#endif