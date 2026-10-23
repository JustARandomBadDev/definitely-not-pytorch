#ifndef DNP_SGD_HPP
#define DNP_SGD_HPP

#include "dnp/neural_network.hpp"
#include "dnp/training/optimizer.hpp"

namespace dnp::training {

template <typename T> class SGD : public Optimizer<T> {
  public:
    SGD(
        T p_learning_rate
    )
        : _learning_rate(p_learning_rate) {}

    void step(NeuralNetwork<T>& p_network) override;

  private:
    T _learning_rate;
};

} // namespace dnp::training

#ifndef DNP_SGD_TPP
#include "dnp/training/sgd.tpp"
#endif

#endif