#ifndef DNP_TRAINER_TPP
#define DNP_TRAINER_TPP

#include "dnp/training/trainer.hpp"
#include <cstddef>

namespace dnp::training {

template <typename T>
void Trainer<T>::train(
    Tensor<T> p_input,
    Tensor<T> p_target
) {
    _network.backward(p_input, p_target);
    _optimizer.step(_network);
}

template <typename T>
void Trainer<T>::train(
    Tensor<T> p_input,
    Tensor<T> p_target,
    std::size_t p_epochs
) {
    for (std::size_t epoch = 0; epoch < p_epochs; epoch++) {
        train(p_input, p_target);
    }
}

} // namespace dnp::training

#endif