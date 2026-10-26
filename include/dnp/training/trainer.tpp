#ifndef DNP_TRAINER_TPP
#define DNP_TRAINER_TPP

#include "dnp/tensor.hpp"
#include "dnp/training/trainer.hpp"
#include <cstddef>

namespace dnp::training {

template <typename T>
void Trainer<T>::train(
    const Tensor<T>& p_input,
    const Tensor<T>& p_target
) {
    Tensor<T> prediction = _network.forward(p_input);
    Tensor<T> gradient = _loss.gradient(prediction, p_target);
    _network.backward(p_input, gradient);
    _optimizer.step(_network);
}

template <typename T>
void Trainer<T>::train(
    const Tensor<T>& p_input,
    const Tensor<T>& p_target,
    std::size_t p_epochs
) {
    for (std::size_t epoch = 0; epoch < p_epochs; epoch++) {
        train(p_input, p_target);
    }
}

} // namespace dnp::training

#endif
