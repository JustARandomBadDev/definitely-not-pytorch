#include <iostream>
#include <random>

#include "dnp/neural_network.hpp"
#include "dnp/tensor.hpp"
#include "dnp/training/sgd.hpp"
#include "dnp/training/trainer.hpp"

int main() {
    int epochs = 1000;
    double learning_rate = 0.1f;

    std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<double> distribution(-1, 1);

    dnp::NeuralNetwork<double>
        nn{{2}, {8, dnp::Tan}, {8, dnp::Tan}, {4, dnp::Tan}, {1}};

    nn.initialize(dnp::Initializer::Random);

    dnp::Tensor<double> input =
        {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};

    dnp::Tensor<double> output = {{0.0f}, {1.0f}, {1.0f}, {0.0f}};

    dnp::training::SGD<double> sgd(learning_rate);
    dnp::training::Trainer<double> trainer(nn, sgd);

    trainer.train(input, output, epochs);

    auto prediction = nn.forward(input);

    std::cout << prediction.toString() << "\n";

    return 0;
}