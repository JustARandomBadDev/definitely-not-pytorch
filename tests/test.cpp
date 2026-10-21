#include <iostream>
#include <random>
#include <vector>

#include "dnp/neural_network.hpp"
#include "dnp/tensor.hpp"

struct Possibility {
    dnp::Tensor<float> e;
    dnp::Tensor<float> y;
};

int main() {
    int epochs = 100;
    float learning_rate = 0.1f;

    std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<float> distribution(-1, 1);

    dnp::NeuralNetwork<float>
        nn{{2}, {8, dnp::Tan}, {8, dnp::Tan}, {4, dnp::Tan}, {1}};

    nn.initialize(dnp::Initializer::Random);

    std::vector<Possibility> xor_table = {
        {{{0.0f, 0.0f}}, {{0.0f}}},
        {{{1.0f, 0.0f}}, {{1.0f}}},
        {{{0.0f, 1.0f}}, {{1.0f}}},
        {{{1.0f, 1.0f}}, {{0.0f}}}
    };

    for (int i = 0; i < epochs; i++) {
        for (Possibility& p : xor_table) {
            nn.backward(p.e, p.y);
            nn.gradientDescent(learning_rate);
        }
    }

    for (Possibility& p : xor_table) {
        auto prediction = nn.forward(p.e);

        std::cout << p.e.get(0, 0) << " XOR " << p.e.get(0, 1)
                  << " -> predicted: " << prediction.get(0, 0)
                  << " | expected: " << p.y.get(0, 0) << "\n";
    }

    return 0;
}