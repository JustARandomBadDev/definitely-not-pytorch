#ifndef DNP_MSE_HPP
#define DNP_MSE_HPP

#include "dnp/training/loss.hpp"

namespace dnp::training {

template <typename T> class MeanSquaredError : public Loss<T> {
  public:
    T compute(const Tensor<T>& p_prediction,
        const Tensor<T>& p_target) const override;

    Tensor<T> gradient(const Tensor<T>& p_prediction,
        const Tensor<T>& p_target) const override;
};

} // namespace dnp::training

#ifndef DNP_MSE_TPP
#include "dnp/training/loss/mse.tpp"
#endif

#endif