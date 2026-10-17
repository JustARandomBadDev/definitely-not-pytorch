#include "dnp/test.hpp"
#include "dnp/tensor.hpp"

namespace dnp {

Tensor<float> forward(
    Tensor<float> entry,
    const Network& network
) {

    auto z = entry.matmul(network.w1) + network.b1;
    z.map([&](float x) { return std::tanh(x); });
    
    auto p_y = z.matmul(network.w2) + network.b2;

    return p_y;
}

void backprob(
    Tensor<float> entry,
    float y,
    float lr,
    Network& network
) {
    // Forward

    auto z = entry.matmul(network.w1) + network.b1;
    auto n = z;
    n.map([&](float x) { return std::tanh(x); });
    
    auto p_y = n.matmul(network.w2) + network.b2;

    // Loss

    float l = loss(y, p_y.get(0, 0));
    Tensor<float> d_y = {{d_loss(y, p_y.get(0, 0))}};

    // Output layer

    auto d_w2 = n.matmul(
        {{d_y}},
        MatrixOp::Transpose,
        MatrixOp::None
    );
    
    Tensor<float> d_b2 {{d_y}};

    // Gradient propagated to hidden neurons

    auto d_n = d_y.matmul(
        network.w2,
        MatrixOp::None,
        MatrixOp::Transpose
    );

    // Through tanh

    auto z_dtanh = z;
    z_dtanh.map(d_tanh);

    auto d_z = d_n.elementwise_mul(z_dtanh);

    // Hidden layer

    auto d_w1 = entry.matmul(
        d_z,
        MatrixOp::Transpose
    );

    auto d_b1 = d_z;

    // Gradient descent

    network.w1 -= d_w1 * lr;
    network.w2 -= d_w2 * lr;

    network.b1 -= d_b1 * lr;
    network.b2 -= d_b2 * lr;
}

}