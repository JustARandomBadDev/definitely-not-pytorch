#ifndef DNP_TRAINER_HPP
#define DNP_TRAINER_HPP

#include "dnp/neural_network.hpp"
#include "dnp/tensor.hpp"
#include "dnp/training/loss.hpp"
#include "dnp/training/optimizer.hpp"
#include <cstddef>

namespace dnp::training {

template <typename T> class Trainer {
  public:
    Trainer(
        NeuralNetwork<T>& p_network,
        Optimizer<T>& p_optimizer,
        Loss<T>& p_loss
    )
        : _network(p_network), _optimizer(p_optimizer), _loss(p_loss) {}

    void train(const Tensor<T>& p_input, const Tensor<T>& p_target);
    void train(const Tensor<T>& p_input,
        const Tensor<T>& p_target,
        std::size_t p_epochs);

  private:
    NeuralNetwork<T>& _network;
    Optimizer<T>& _optimizer;
    Loss<T>& _loss;
};

} // namespace dnp::training

#ifndef DNP_TRAINER_TPP
#include "dnp/training/trainer.tpp"
#endif

#endif
