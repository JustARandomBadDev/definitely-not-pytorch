# Definitely Not PyTorch

Definitely Not PyTorch is a small C++20 deep-learning library I am building from scratch to understand how neural networks actually work behind higher-level frameworks.

The current version focuses on a simple CPU MLP implementation: tensors, dense layers, forward/backward propagation, batching, losses, optimizers, training and model serialization.

It is still experimental and mainly intended to be used in my own projects while I keep extending it.

## Current state

DNP currently supports:

- 2D `Tensor<T>` with basic arithmetic and matrix multiplication;
- dense multilayer perceptrons;
- variable batch sizes;
- `None`, `Tan`, `ReLu` and `LeakyRelu` activations;
- explicit forward and backward propagation;
- a `Loss<T>` interface with MSE;
- an `Optimizer<T>` interface with SGD;
- a simple `Trainer<T>`;
- `.dnp` model save/load;
- `float` and `double`.

Everything currently runs on CPU.

## Build

```bash
cmake -S . -B build
cmake --build build
```

This builds:

```text
dnp
dnp_sandbox
dnp_tests
```

`dnp` is a header-only CMake target.

## Use DNP in another project

### FetchContent

```cmake
include(FetchContent)

FetchContent_Declare(
    dnp
    GIT_REPOSITORY https://github.com/JustARandomBadDev/definitely-not-pytorch.git
    GIT_TAG <tag-or-commit>
)

FetchContent_MakeAvailable(dnp)

target_link_libraries(
    my_target
    PRIVATE
        dnp
)
```

Until proper release tags exist, pinning a commit is preferable.

### Use a local checkout

When developing DNP and another project at the same time, keep the same `FetchContent` declaration and override it at configure time:

```bash
cmake -S . -B build \
    -DFETCHCONTENT_SOURCE_DIR_DNP=/absolute/path/to/definitely-not-pytorch
```

The consumer will then build against the local DNP repository instead of fetching it.

This is the setup I use while developing DNP alongside other projects.

A direct local setup is also possible:

```cmake
add_subdirectory(
    /absolute/path/to/definitely-not-pytorch
    ${CMAKE_BINARY_DIR}/dnp
)

target_link_libraries(my_target PRIVATE dnp)
```

## Example

```cpp
#include <iostream>

#include "dnp/neural_network.hpp"
#include "dnp/tensor.hpp"
#include "dnp/training/loss/mse.hpp"
#include "dnp/training/optimizers/sgd.hpp"
#include "dnp/training/trainer.hpp"

int main() {
    dnp::NeuralNetwork<float> network{
        {2},
        {4, dnp::Tan},
        {1},
    };

    network.initialize(dnp::Initializer::Random);

    const dnp::Tensor<float> input{
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 1.0f},
    };

    const dnp::Tensor<float> target{
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f},
    };

    dnp::training::MeanSquaredError<float> loss;
    dnp::training::SGD<float> optimizer(0.1f);
    dnp::training::Trainer<float> trainer(network, optimizer, loss);

    trainer.train(input, target, 3000);

    const auto prediction = network.forward(input);

    std::cout << prediction.toString() << '\n';
}
```

The training path is intentionally explicit:

```text
input
  ↓
network.forward()
  ↓
prediction
  ↓
loss.gradient()
  ↓
network.backward()
  ↓
optimizer.step()
```

The idea is to keep each part visible instead of hiding the whole training process behind one large abstraction.

## Batching

Inputs use:

```text
[B, F]
```

where `B` is the batch size and `F` the number of input features.

For an output size `O`:

```text
[B, O]
```

The batch size is not part of the network architecture, so the same network can process different batch sizes between calls.

DNP does not currently split datasets into mini-batches automatically. The tensors passed to `Trainer` are used directly as the current batch.

## Model serialization

```cpp
#include "dnp/serialization/model.hpp"

dnp::serialization::saveModel(network, "model.dnp");

auto loaded =
    dnp::serialization::loadModel<float>("model.dnp");
```

A `.dnp` file currently stores the network architecture, activations, weights and biases.

`float` and `double` models are supported.

This is model serialization only; optimizer state and training progress are not saved yet.

## Sandbox

`sandbox/main.cpp` is the place used for manual experiments while developing the library.

```bash
./build/dnp_sandbox
```

Its content is expected to change frequently.

## Tests

Automated tests live in `tests/`.

They cover the important CPU behavior such as tensor operations, batching, forward/backward propagation, numerical gradient checking, MSE, SGD, Trainer and model serialization.

Run them directly with:

```bash
./build/dnp_tests
```

or through CTest:

```bash
ctest --test-dir build --output-on-failure
```

CTest currently sees `dnp_tests` as one test executable. The individual `DNP_TEST` cases are handled internally by the small test harness.

## Where this is going

The current CPU MLP is mainly the foundation.

The next important step is using DNP in the fluid simulator for a real ML workload, then extending the parts that become necessary from that integration.

CUDA, additional optimizers, checkpoints, dataset utilities and ONNX support can come later as the project actually needs them.