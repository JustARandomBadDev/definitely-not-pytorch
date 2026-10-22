#ifndef DNP_MODEL_HPP
#define DNP_MODEL_HPP

#include <filesystem>

#include "dnp/neural_network.hpp"

namespace dnp::serialization {

template <typename T>
void saveModel(const NeuralNetwork<T>& p_network,
    const std::filesystem::path& p_path);

template <typename T>
NeuralNetwork<T> loadModel(const std::filesystem::path& p_path);

} // namespace dnp::serialization

#ifndef DNP_MODEL_TPP
#include "dnp/serialization/model.tpp"
#endif

#endif