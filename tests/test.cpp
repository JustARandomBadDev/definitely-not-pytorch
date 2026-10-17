#include <iostream>
#include <random>
#include <vector>

#include "dnp/tensor.hpp"
#include "dnp/test.hpp"

struct Possibility {
    dnp::Tensor<float> e;
    float y;
};

int main() {
    int epochs = 1'000'000;
    float learning_rate = 0.0001f;

    std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<float> distribution(-1, 1);

    std::vector<Possibility> xor_table = {
        {{{0.0f, 0.0f}}, 0.0f},
        {{{1.0f, 0.0f}}, 1.0f},
        {{{0.0f, 1.0f}}, 1.0f},
        {{{1.0f, 1.0f}}, 0.0f}
    };

    dnp::Network network {
        dnp::Tensor<float>({2, 2}),
        dnp::Tensor<float>({2, 1}),
        dnp::Tensor<float>({1, 2}),
        dnp::Tensor<float>({1, 1})
    };

    network.w1.generate([&]() {
        return distribution(generator);
    });
    network.w2.generate([&]() {
        return distribution(generator);
    });

    for (int i = 0; i < epochs; i++) {
        for (const Possibility& p : xor_table) {
            backprob(
                p.e,
                p.y,
                learning_rate,
                network
            );
        }
    }

    for (const Possibility& p : xor_table) {
        auto prediction = forward(
            p.e,
            network
        );

        std::cout
            << p.e.get(0, 0) << " XOR " << p.e.get(0, 1)
            << " -> predicted: " << prediction.get(0, 0)
            << " | expected: " << p.y
            << "\n";
    }

    return 0;
}