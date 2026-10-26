#ifndef DNP_SGD_TPP
#define DNP_SGD_TPP

#include "dnp/training/optimizers/sgd.hpp"
#include <cstddef>

namespace dnp::training {

template <typename T>
void SGD<T>::step(
    NeuralNetwork<T>& p_network
) {
    std::vector<typename NeuralNetwork<T>::Layer>& layers =
        p_network.getLayers();
    const std::vector<typename NeuralNetwork<T>::Gradient>& gradients =
        p_network.getGradients();

    for (std::size_t index = 0; index < layers.size(); index++) {
        layers[index].weights -= gradients[index].weights * _learning_rate;
        layers[index].biais -= gradients[index].biais * _learning_rate;
    }
}

} // namespace dnp::training

#endif