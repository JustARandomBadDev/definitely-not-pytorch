#ifndef DNP_ACTIVATION_HPP
#define DNP_ACTIVATION_HPP

#include <cmath>

namespace dnp {

enum Activation { None, Tan, ReLu, LeakyRelu };

template <typename T>
T activate(
    T p_x,
    Activation p_activation
) {
    switch (p_activation) {
    case Activation::Tan:
        return std::tanh(p_x);

    case Activation::ReLu:
        return p_x > 0 ? p_x : 0;

    case Activation::LeakyRelu:
        return p_x > 0 ? p_x : static_cast<T>(0.01) * p_x;

    case Activation::None:
        return p_x;
    }
}

template <typename T>
T activationDerivative(
    T p_x,
    Activation p_activation
) {
    switch (p_activation) {
    case Activation::Tan: {
        T t = std::tanh(p_x);
        return 1.f - t * t;
    }

    case Activation::ReLu:
        return p_x > 0 ? static_cast<T>(1) : static_cast<T>(0);

    case Activation::LeakyRelu:
        return p_x > 0 ? static_cast<T>(1) : static_cast<T>(0.01);

    case Activation::None:
        return static_cast<T>(1);
    }
}

} // namespace dnp

#endif