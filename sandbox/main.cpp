#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

#include "dnp/neural_network.hpp"
#include "dnp/tensor.hpp"
#include "dnp/training/loss/mse.hpp"
#include "dnp/training/optimizers/sgd.hpp"
#include "dnp/training/trainer.hpp"

int main() {
    int epochs = 1000;
    double learning_rate = 0.1f;

    const std::vector<double> data_input =
        {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    const std::vector<double> data_output = {0.0f, 1.0f, 1.0f, 0.0f};

    std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<double> distribution(-1, 1);

    dnp::NeuralNetwork<double>
        nn{{2}, {8, dnp::Tan}, {8, dnp::Tan}, {4, dnp::Tan}, {1}};

    nn.initialize(dnp::Initializer::Random);

    dnp::Tensor<double> input = {4, 2};

    for (std::size_t i = 0; i < data_input.size(); i++) {
        input.set(i, data_input[i]);
    }

    dnp::Tensor<double> output = {4, 1};

    for (std::size_t i = 0; i < data_output.size(); i++) {
        output.set(i, data_output[i]);
    }

    dnp::training::SGD<double> sgd(learning_rate);
    dnp::training::MeanSquaredError<double> mse;
    dnp::training::Trainer<double> trainer(nn, sgd, mse);

    trainer.train(input, output, epochs);

    auto prediction = nn.forward(input);

    std::cout << prediction.toString() << "\n";

    return 0;
}
