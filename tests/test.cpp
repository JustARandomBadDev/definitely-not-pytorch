#include <iostream>
#include <vector>

#include "dnp/test.hpp"

struct Possibility {
    float e1;
    float e2;
    float y;
};

int main() {
    int epochs = 1'000'000;
    float learning_rate = 0.0001f;

    std::vector<Possibility> xor_table = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 1.0f},
        {0.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 0.0f}
    };

    dnp::Network network;

    for (int i = 0; i < epochs; i++) {
        for (const Possibility& p : xor_table) {
            backprob(
                p.e1,
                p.e2,
                p.y,
                learning_rate,
                network
            );
        }
    }

    for (const Possibility& p : xor_table) {
        float prediction = forward(
            p.e1,
            p.e2,
            network
        );

        std::cout
            << p.e1 << " XOR " << p.e2
            << " -> predicted: " << prediction
            << " | expected: " << p.y
            << "\n";
    }

    return 0;
}