#ifndef _NUM_CPP_H
#define _NUM_CPP_H
#include <omp.h>

#include <array>
#include <concepts>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace numcpp {
template <typename T>
concept Integer = std::integral<T>;
template <typename T>
concept Number = std::integral<T> or std::floating_point<T>;

template <Number dtype = double>
class NdArray {
  std::shared_ptr<dtype[]> _array;
  std::vector<int> _shape;
  int _size = 1;

 public:
  template <Integer... Args>
  NdArray(Args... args) : _shape{args...} {
    for (const auto arg : _shape) _size *= arg;
    _array = std::shared_ptr<dtype[]>(new dtype[_size]);
  }
  // template <Integer... Args>
  // NdArray(Args... args) : _shape{args...} {
  //   for (const auto arg : {args...}) _size *= arg;
  //   _array = std::shared_ptr<dtype[]>(new dtype[_size]);
  // }

  // NdArray(std::vector<std::vector<double>> &matrix) {
  //   _shape = {matrix.size(), matrix[0].size()};
  //   _size = _shape[0] * _shape[1];
  //   _array = std::make_shared<dtype[]>(new dtype[_shape]);
  //   for (int i = 0; i < _shape[0]; i++)
  //     for (int j = 0; j < _shape[1]; j++) _array[{i, j}] = matrix[i][j];
  // }

  std::vector<int> shape() const { return _shape; }
  int size() const { return _size; }

  /* =============================
   * operator overload
   * operator [{a, b, c}]
   * =============================
   */

  template <Integer Int>
  dtype &operator[](const std::initializer_list<Int> &&args) {
    int index = 0;
    for (const auto &arg : args) {
      int tmp = arg;
      int i = &arg - args.begin();
      for (int j = std::size(args) - 1; j > i; --j) tmp *= _shape[j];
      index += tmp;
    }
    return _array[index];
  }

  template <Integer Int>
  const dtype &operator[](const std::initializer_list<Int> &&args) const {
    int index = 0;
    for (const auto &arg : args) {
      int tmp = arg;
      int i = &arg - args.begin();
      for (int j = std::size(args) - 1; j > i; --j) tmp *= _shape[j];
      index += tmp;
    }
    return _array[index];
  }

  template <Integer Int>
  dtype &operator[](const Int index) {
    return _array[index];
  }

  template <Integer Int>
  const dtype &operator[](const Int index) const {
    return _array[index];
  }

  /* =============================
   * operator overload
   * operator =
   * =============================
   */

  NdArray<dtype> operator=(const std::vector<std::vector<double>> &matrix) {
    _shape = {static_cast<int>(matrix.size()),
              static_cast<int>(matrix[0].size())};
    _size = _shape[0] * _shape[1];
    _array = std::shared_ptr<dtype[]>(new dtype[_size]);
    for (int i = 0; i < _shape[0]; i++)
      for (int j = 0; j < _shape[1]; j++) (*this)[{i, j}] = matrix[i][j];
    return *(this);
  }

  NdArray<dtype> operator=(const NdArray &ndarray) {
    _shape = ndarray.shape();
    *(this) = ndarray;
    return *(this);
  }

  //   NdArray<dtype> operator=(const NdArray &&ndarray) {
  //     _shape = ndarray.shape();
  //     *(this) = std::move(ndarray);
  //     return *(this);
  //   }

  template <Number Num>
  void operator=(const Num value) {
    for (int i = 0; i < _size; i++) _array[i] = static_cast<dtype>(value);
    // return *this;
  }
  /* =============================
   * operator overload
   * operator + - * /
   * =============================
   */

  // operatr +
  template <Number Num>
  NdArray<dtype> operator+(const Num value) {
    NdArray<dtype> result = *(this);
#pragma omp parallel for
    for (int i = 0; i < _size; i++) result[i] += static_cast<dtype>(value);
    return result;
  }

  template <Number Num>
  NdArray<dtype> operator+(const NdArray<Num> &ndarray) {
    NdArray<dtype> result = *(this);
#pragma omp parallel for
    for (int i = 0; i < _size; i++) result[i] += ndarray[i];
    return result;
  }

  // operator "-"
  template <Number Num>
  NdArray<dtype> operator-(const Num value) {
    NdArray<dtype> result = *(this);
#pragma omp parallel for
    for (int i = 0; i < _size; i++) result[i] -= static_cast<dtype>(value);
    return result;
  }

  template <Number Num>
  NdArray<dtype> operator-(const NdArray<Num> &ndarray) {
    NdArray<dtype> result = *(this);
#pragma omp parallel for
    for (int i = 0; i < _size; i++) result[i] -= ndarray[i];
    return result;
  }

  // operator "*"
  template <Number Num>
  NdArray<dtype> operator*(const Num value) {
    NdArray<dtype> result = *(this);
#pragma omp parallel for
    for (int i = 0; i < _size; i++) result[i] *= static_cast<dtype>(value);
    return result;
  }

  template <Number Num>
  NdArray<dtype> operator*(const NdArray<Num> &ndarray) {
    NdArray<dtype> result = *(this);
#pragma omp parallel for
    for (int i = 0; i < _size; i++) result[i] *= ndarray[i];
    return result;
  }

  // operator "/"
  template <Number Num>
  NdArray<dtype> operator/(const Num value) {
    NdArray<dtype> result = *(this);
#pragma omp parallel for
    for (int i = 0; i < _size; i++) result[i] /= static_cast<dtype>(value);
    return result;
  }

  template <Number Num>
  NdArray<dtype> operator/(const NdArray<Num> &ndarray) {
    NdArray<dtype> result = *(this);
#pragma omp parallel for
    for (int i = 0; i < _size; i++) result[i] /= ndarray[i];
    return result;
  }

  /* =============================
   * operator overload
   * matrix product
   * operator &
   * =============================
   */

  template <Number Num>
  NdArray<dtype> operator&(const NdArray<Num> &ndarray) {
    const int hight = this->_shape[0];
    const int width = ndarray.shape()[1];
    const int common = this->_shape[1];
    if (common != ndarray.shape()[0])
      throw std::runtime_error("can't pruduct there array");

    NdArray<dtype> result(hight, width);
    result = 0;
#pragma omp parallel for
    for (int i = 0; i < hight; i++)
      for (int j = 0; j < width; j++)
        for (int k = 0; k < common; k++)
          result[{i, j}] += (*this)[{i, k}] * ndarray[{k, j}];
    return result;
  }
};

std::ostream &operator<<(std::ostream &os, const NdArray<double> &ndarray) {
  for (int i = 0; i < ndarray.size(); i++)
    os << std::setw(5) << ndarray[i] << " ";
  return os;
}

}  // namespace numcpp

#endif
