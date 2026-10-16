#include "dnp/test.hpp"

namespace dnp {

float forward(
    float e1,
    float e2,
    const Network& network
) {
    float z1 = e1 * network.w11 + e2 * network.w12 + network.b1;
    float n1 = tanh(z1);

    float z2 = e1 * network.w21 + e2 * network.w22 + network.b2;
    float n2 = tanh(z2);

    float p_y = n1 * network.w31 + n2 * network.w32 + network.b3;

    return p_y;
}

void backprob(
    float e1,
    float e2,
    float y,
    float lr,
    Network& network
) {
    /*
     * Forward
     */

    float z1 = e1 * network.w11 + e2 * network.w12 + network.b1;
    float n1 = tanh(z1);

    float z2 = e1 * network.w21 + e2 * network.w22 + network.b2;
    float n2 = tanh(z2);

    float p_y = n1 * network.w31 + n2 * network.w32 + network.b3;

    /*
     * Loss
     */

    float l = loss(y, p_y);
    float dL_dy = d_loss(y, p_y);

    /*
     * Output layer
     */

    float d_w31 = dL_dy * n1;
    float d_w32 = dL_dy * n2;
    float d_b3 = dL_dy;

    /*
     * Gradient propagated to hidden neurons
     */

    float d_n1 = dL_dy * network.w31;
    float d_n2 = dL_dy * network.w32;

    /*
     * Through tanh
     */

    float d_z1 = d_n1 * d_tanh(z1);
    float d_z2 = d_n2 * d_tanh(z2);

    /*
     * Hidden layer
     */

    float d_w11 = d_z1 * e1;
    float d_w12 = d_z1 * e2;

    float d_w21 = d_z2 * e1;
    float d_w22 = d_z2 * e2;

    float d_b1 = d_z1;
    float d_b2 = d_z2;

    /*
     * Gradient descent
     */

    network.w11 -= lr * d_w11;
    network.w12 -= lr * d_w12;
    network.w21 -= lr * d_w21;
    network.w22 -= lr * d_w22;
    network.w31 -= lr * d_w31;
    network.w32 -= lr * d_w32;

    network.b1 -= lr * d_b1;
    network.b2 -= lr * d_b2;
    network.b3 -= lr * d_b3;
}

}