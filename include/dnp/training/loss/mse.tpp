#ifndef DNP_MSE_TPP
#define DNP_MSE_TPP

#include "dnp/training/loss/mse.hpp"
#include <cstddef>
#include <stdexcept>
namespace dnp::training {

template <typename T>
T MeanSquaredError<T>::compute(
    const Tensor<T>& p_prediction,
    const Tensor<T>& p_target
) const {
    if (p_prediction.shape().rows != p_target.shape().rows ||
        p_prediction.shape().cols != p_target.shape().cols) {
        throw std::invalid_argument(
            "MSE compute: prediction and target shapes must match"
        );
    }

    if (p_prediction.size() == 0) {
        throw std::invalid_argument("MSE compute: tensors cannot be empty");
    }

    const Tensor<T> diff = p_prediction - p_target;

    T squared_sum = static_cast<T>(0);

    for (std::size_t i = 0; i < diff.size(); i++) {
        const T value = diff.get(i);
        squared_sum += value * value;
    }

    return squared_sum / static_cast<T>(diff.size());
}

template <typename T>
Tensor<T> MeanSquaredError<T>::gradient(
    const Tensor<T>& p_prediction,
    const Tensor<T>& p_target
) const {
    if (p_prediction.shape().rows != p_target.shape().rows ||
        p_prediction.shape().cols != p_target.shape().cols) {
        throw std::invalid_argument(
            "MSE gradient: prediction and target shapes must match"
        );
    }

    if (p_prediction.size() == 0) {
        throw std::invalid_argument("MSE gradient: tensors cannot be empty");
    }

    return (p_prediction - p_target) *
           (static_cast<T>(2) / static_cast<T>(p_target.size()));
}

} // namespace dnp::training

#endif
