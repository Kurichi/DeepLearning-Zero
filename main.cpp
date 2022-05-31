#include <iostream>
#include <string>
#include <unordered_map>

#include "Inc/matplotlibcpp.h"
#include "NeuralNetwork/ActivationFunction.hpp"
#include "NeuralNetwork/NumCpp.hpp"
namespace dl = deepLearningZero;
namespace plt = matplotlibcpp;
namespace nc = numcpp;

namespace myDeepLearning {
using network_t = std::unordered_map<std::string, nc::NdArray<double>>;

network_t init_network() {
  network_t result;
  // Weight
  result["W1"] = {{0.1, 0.3, 0.5}, {0.2, 0.4, 0.6}};
  result["W2"] = {{0.1, 0.4}, {0.2, 0.5}, {0.3, 0.6}};
  result["W3"] = {{0.1, 0.3}, {0.2, 0.4}};

  // Bias
  result["B1"] = {{0.1, 0.2, 0.3}};
  result["B2"] = {{0.1, 0.2}};
  result["B3"] = {{0.1, 0.2}};

  return result;
}

nc::NdArray<double> forward(network_t &network, nc::NdArray<double> &x) {
  auto &&a1 = (x & network["W1"]) + network["B1"];
  auto &&z1 = dl::sigmoid(a1);

  auto &&a2 = (z1 & network["W2"]) + network["B2"];
  auto &&z2 = dl::sigmoid(a2);

  auto &&a3 = (z2 & network["W3"]) + network["B3"];
  return a3;
}

}  // namespace myDeepLearning

namespace mdl = myDeepLearning;

int main() {
  auto &&network = mdl::init_network();
  nc::NdArray x(1, 2);
  x[0] = 1.0;
  x[1] = 0.5;
  auto &&y = mdl::forward(network, x);
  std::cout << y << std::endl;

  return 0;
}