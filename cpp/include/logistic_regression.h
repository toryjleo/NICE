/// The MIT License (MIT)
//
// Copyright (c) 2016 Northeastern University
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cmath>
#include <functional>
#include "include/matrix.h"
#include "include/vector.h"
#include "include/cpu_operations.h"
#include "include/util.h"

namespace Nice {

template<typename T>
class LogisticRegression {

 private:
  Matrix<T> x;
  Vector<T> y;
  Vector<T> theta;
  int iterations = 100;
  T lambda = 1;
  float alpha = .001;


 public:
/*  LogisticRegression(const Matrix<T> &xin, const Vector<T> &yin) {
    x = xin;
    y = yin;
    theta.resize(x.cols(), 1);
    theta.setZero();
  }
*/  LogisticRegression() {}

  void SetX(const Matrix<T> &xin) {
    x = xin;
    theta.resize(x.cols(), 1);
    theta.setZero();
  }
  void SetY(const Matrix<T> &yin) {
    y = yin;
  }
  void SetIter(int iter) {
    iterations = iter;
  }
  void SetLambda(T l) {
    lambda = l;
  }
  void SetAlpha(float a) {
    alpha = a;
  }

  void Fit(const Matrix<T> &xin, const Vector<T> &yin) {
    x.resize(xin.rows(), xin.cols() + 1);
    x.col(0).setOnes();
    for(int i = 1; i <= xin.cols(); ++i) {
      x.col(i) = xin.col(i - 1);
    }
    y = yin;
    theta.resize(x.cols());
    theta.setZero();
//    std::cout << Cost() << std::endl;
    for(int i = 0; i < iterations; ++i) {
      Gradient();
      std::cout << Cost() << std::endl;
    }
//    std::cout << Cost() << std::endl;
  }


 private:
  // Computes the sigmoid of z
  Matrix<T> Sigmoid(Matrix<T> &z) {
    Matrix<T> m(z.rows(), z.cols());
    m.setOnes(m.rows(), m.cols());
    return (m.array() / (m.array() +
            (-z).array().unaryExpr(std::ptr_fun(util::exp<T>)))).matrix();
  }
  // Calls gradient descent and updates Theta
  void Gradient() {
    Matrix<T> prod = x * theta;
    Matrix<T> h_of_x = Sigmoid(prod);
    theta -= alpha * (x.transpose() * (h_of_x - y)) / y.size();
    // Normalizes all thetas except for theta zero
    theta.segment(1, theta.size() - 1) += (lambda/y.size() * theta.segment(1, theta.size() - 1));
  }
  // Computes the current cost
  T Cost() {
    Matrix<T> prod = x * theta;
    Matrix<T> h_of_x = Sigmoid(prod);
    Matrix<T> m(h_of_x.rows(), h_of_x.cols());
    Vector<T> v(y.size());
    m.setOnes(m.rows(), m.cols());
    v.setOnes(y.size());
    T J = (-1 * y.transpose() * h_of_x.array().log().matrix() -
    (v - y.transpose()) * (m - h_of_x).array().log().matrix()).sum() / y.size();
    J += lambda/(2 * y.size()) * theta.segment(1, theta.size() - 1).array().square().sum();
    return J;
  }

};

}
