#ifndef DNP_ACTIVATION_HPP
#define DNP_ACTIVATION_HPP

#include <cmath>
#include <stdexcept>
#include <string>

namespace dnp {

enum Activation { None, Tan, ReLu, LeakyRelu };

inline std::string getStringFromActivation(
    const Activation p_activation
) {
    switch (p_activation) {
    case Activation::None:
        return "none";

    case Activation::Tan:
        return "tan";

    case Activation::ReLu:
        return "relu";

    case Activation::LeakyRelu:
        return "leakyrelu";
    }

    throw std::runtime_error("Unknown activation");
}

inline Activation getActivationFromString(
    const std::string& p_activation
) {
    if (p_activation == "none") return Activation::None;
    if (p_activation == "tan") return Activation::Tan;
    if (p_activation == "relu") return Activation::ReLu;
    if (p_activation == "leakyrelu") return Activation::LeakyRelu;

    throw std::runtime_error("Unknown activation: " + p_activation);
}

template <typename T>
T activate(
    T p_x,
    Activation p_activation
) {
    switch (p_activation) {
    case Activation::Tan:
        return std::tanh(p_x);

    case Activation::ReLu:
        return p_x > 0 ? p_x : static_cast<T>(0);

    case Activation::LeakyRelu:
        return p_x > 0 ? p_x : static_cast<T>(0.01) * p_x;

    case Activation::None:
        return p_x;
    }

    throw std::runtime_error("Unknown activation");
}

template <typename T>
T activationDerivative(
    T p_x,
    Activation p_activation
) {
    switch (p_activation) {
    case Activation::Tan: {
        const T t = std::tanh(p_x);

        return static_cast<T>(1) - t * t;
    }

    case Activation::ReLu:
        return p_x > 0 ? static_cast<T>(1) : static_cast<T>(0);

    case Activation::LeakyRelu:
        return p_x > 0 ? static_cast<T>(1) : static_cast<T>(0.01);

    case Activation::None:
        return static_cast<T>(1);
    }

    throw std::runtime_error("Unknown activation");
}

} // namespace dnp

#endif