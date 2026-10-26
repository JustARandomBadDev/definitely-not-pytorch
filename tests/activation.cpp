#include "test.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "dnp/activation.hpp"

namespace {

using dnp::activate;
using dnp::Activation;
using dnp::activationDerivative;
using dnp::getActivationFromString;
using dnp::getStringFromActivation;
using dnp::LeakyRelu;
using dnp::None;
using dnp::ReLu;
using dnp::Tan;
using dnp::test::expectEqual;
using dnp::test::expectNear;
using dnp::test::expectThrows;

DNP_TEST(
    activation_values
) {
    expectNear(activate(-2.0, None), -2.0, 1e-12, "none negative");
    expectNear(activate(0.0, None), 0.0, 1e-12, "none zero");
    expectNear(activate(2.0, None), 2.0, 1e-12, "none positive");

    expectNear(activate(-0.75, Tan), std::tanh(-0.75), 1e-12, "tanh negative");
    expectNear(activate(0.0, Tan), 0.0, 1e-12, "tanh zero");
    expectNear(activate(0.75, Tan), std::tanh(0.75), 1e-12, "tanh positive");

    expectNear(activate(-2.0, ReLu), 0.0, 1e-12, "relu negative");
    expectNear(activate(0.0, ReLu), 0.0, 1e-12, "relu zero");
    expectNear(activate(2.0, ReLu), 2.0, 1e-12, "relu positive");

    expectNear(activate(-2.0, LeakyRelu), -0.02, 1e-12, "leaky relu negative");
    expectNear(activate(0.0, LeakyRelu), 0.0, 1e-12, "leaky relu zero");
    expectNear(activate(2.0, LeakyRelu), 2.0, 1e-12, "leaky relu positive");
}

DNP_TEST(
    activation_derivatives
) {
    expectNear(activationDerivative(-2.0, None),
        1.0,
        1e-12,
        "none derivative negative");
    expectNear(activationDerivative(0.0, None),
        1.0,
        1e-12,
        "none derivative zero");
    expectNear(activationDerivative(2.0, None),
        1.0,
        1e-12,
        "none derivative positive");

    const double negative = -0.75;
    const double positive = 0.75;
    expectNear(activationDerivative(negative, Tan),
        1.0 - std::tanh(negative) * std::tanh(negative),
        1e-12,
        "tanh derivative negative");
    expectNear(activationDerivative(0.0, Tan),
        1.0,
        1e-12,
        "tanh derivative zero");
    expectNear(activationDerivative(positive, Tan),
        1.0 - std::tanh(positive) * std::tanh(positive),
        1e-12,
        "tanh derivative positive");

    expectNear(activationDerivative(-1.0, ReLu),
        0.0,
        1e-12,
        "relu derivative negative");
    expectNear(activationDerivative(0.0, ReLu),
        0.0,
        1e-12,
        "relu derivative zero");
    expectNear(activationDerivative(1.0, ReLu),
        1.0,
        1e-12,
        "relu derivative positive");

    expectNear(activationDerivative(-1.0, LeakyRelu),
        0.01,
        1e-12,
        "leaky relu derivative negative");
    expectNear(activationDerivative(0.0, LeakyRelu),
        0.01,
        1e-12,
        "leaky relu derivative zero");
    expectNear(activationDerivative(1.0, LeakyRelu),
        1.0,
        1e-12,
        "leaky relu derivative positive");
}

DNP_TEST(
    activation_string_round_trip
) {
    const Activation activations[] = {None, Tan, ReLu, LeakyRelu};
    const std::string names[] = {"none", "tan", "relu", "leakyrelu"};

    for (std::size_t index = 0; index < 4; index++) {
        expectEqual(getStringFromActivation(activations[index]),
            names[index],
            "activation enum to string");
        expectEqual(static_cast<int>(getActivationFromString(names[index])),
            static_cast<int>(activations[index]),
            "activation string to enum");
        expectEqual(static_cast<int>(getActivationFromString(
                        getStringFromActivation(activations[index])
                    )),
            static_cast<int>(activations[index]),
            "activation round trip");
    }
}

DNP_TEST(
    activation_invalid_values_throw
) {
    expectThrows<std::runtime_error>(
        []() { (void)getActivationFromString("swish"); },
        "unknown activation string",
        "Unknown activation"
    );
}

} // namespace
