#ifndef _ACTIVATION_FUNCTION_HPP
#define _ACTIVATION_FUNCTION_HPP
#include <cmath>
#include <concepts>
#include <iostream>
#include <vector>

#include "NumCpp.hpp"

namespace deepLearningZero {
template <typename T>
concept Number = std::integral<T> or std::floating_point<T>;

// Step function
template <Number N>
std::vector<N> step_function(const std::vector<N> &v) {
  std::vector<N> result = v;
  for (N &value : result) value = value > 0;
  return result;
}

// Sigmoid
template <Number N>
numcpp::NdArray<double> sigmoid(const numcpp::NdArray<N> &m) {
  numcpp::NdArray<double> result = m;
  for (int i = 0; i < result.size(); i++) result[i] = 1 / (1 + exp(-result[i]));
  return result;
}

template <Number N>
std::vector<N> ReLU(const std::vector<N> &v) {
  std::vector<N> result = v;
  for (auto &value : result) value = std::max((N)0, value);
  return result;
}

}  // namespace deepLearningZero

#endif