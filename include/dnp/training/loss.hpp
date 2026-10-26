#ifndef DNP_LOSS_HPP
#define DNP_LOSS_HPP

#include "dnp/tensor.hpp"
namespace dnp::training {

template <typename T> class Loss {
  public:
    virtual ~Loss() = default;

    virtual T compute(const Tensor<T>& p_prediction,
        const Tensor<T>& p_target) const = 0;

    virtual Tensor<T> gradient(const Tensor<T>& p_prediction,
        const Tensor<T>& p_target) const = 0;
};

} // namespace dnp::training

#endif