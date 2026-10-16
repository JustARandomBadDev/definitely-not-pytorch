#ifndef DNP_TEST_HPP
#define DNP_TEST_HPP

#include <cmath>

namespace dnp {

struct Network {
    float w11 = 0.1f;
    float w12 = -0.2f;
    float w21 = -0.3f;
    float w22 = 0.4f;

    float w31 = 0.5f;
    float w32 = -0.6f;

    float b1 = 0.0f;
    float b2 = 0.0f;
    float b3 = 0.0f;
};

float forward(
    float e1,
    float e2,
    const Network& network
);

void backprob(
    float e1,
    float e2,
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