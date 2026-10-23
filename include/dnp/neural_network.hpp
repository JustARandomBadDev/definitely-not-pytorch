#ifndef DNP_NEURAL_NETWORK_HPP
#define DNP_NEURAL_NETWORK_HPP

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "dnp/activation.hpp"
#include "dnp/tensor.hpp"

namespace dnp {

enum Initializer { Random };

struct LayerDescriptor {
    std::size_t size = 0;
    Activation activation = Activation::None;
};

template <typename T> class NeuralNetwork {
  public:
    struct Layer {
        Tensor<T> weights;
        Tensor<T> biais;
        Activation activation;
    };

    struct Gradient {
        Tensor<T> weights;
        Tensor<T> biais;
    };

    NeuralNetwork(const std::vector<LayerDescriptor>& p_layers);
    NeuralNetwork(std::initializer_list<LayerDescriptor> p_layers);

    void initialize(Initializer p_initializer);

    Tensor<T> forward(Tensor<T>& p_input);
    void backward(Tensor<T>& p_input, Tensor<T>& p_output);

    std::string toString() const;

    std::size_t getInputSize() const {
        return _input_size;
    }

    std::vector<Layer>& getLayers() {
        return _layers;
    }
    const std::vector<Layer>& getLayers() const {
        return _layers;
    }

    const std::vector<Gradient>& getGradients() const {
        return _gradients;
    }

  private:
    struct TrainData {
        Tensor<T> linear;
        Tensor<T> activated;
    };

    std::size_t _input_size;

    std::vector<Layer> _layers;
    std::vector<TrainData> _train_data;
    std::vector<Gradient> _gradients;

    void initializeLayers(std::span<const LayerDescriptor> p_layers);

    void addLayer(std::size_t p_prev_layer_size,
        std::size_t p_size,
        Activation p_activation = Activation::None);
};

} // namespace dnp

#ifndef DNP_NEURAL_NETWORK_TPP
#include "dnp/neural_network.tpp"
#endif

#endif
