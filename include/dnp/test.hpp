#ifndef DNP_TEST_HPP
#define DNP_TEST_HPP

#include <cmath>

#include "dnp/tensor.hpp"

namespace dnp {

struct Network {
    Tensor<float> w1;
    Tensor<float> w2;

    Tensor<float> b1;
    Tensor<float> b2;
};

Tensor<float> forward(
    Tensor<float> entry,
    const Network& network
);

void backprob(
    Tensor<float> entry,
    float y,
    float lr,
    Network& network
);

inline float tanh(float z) {
    return std::tanh(z);
}

inline float d_tanh(float z) {
    float t = std::tanh(z);
    return 1.0f - t * t;
}

inline float loss(float y, float p_y) {
    float t = p_y - y;

    return t * t;
}

inline float d_loss(float y, float p_y) {
    return 2.0f * (p_y - y);
}

}

#endif