#ifndef DNP_OPTIMIZER_HPP
#define DNP_OPTIMIZER_HPP

#include "dnp/neural_network.hpp"

namespace dnp::training {

template <typename T> class Optimizer {
  public:
    virtual ~Optimizer() = default;
    virtual void step(NeuralNetwork<T>& p_network) = 0;
};

} // namespace dnp::training

#endif