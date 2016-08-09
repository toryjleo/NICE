
// The MIT License (MIT)
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


#include <stdio.h>
#include <iostream>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/logistic_regression.h"
#include "include/matrix.h"

template<class T>
class LogisticRegressionTest : public ::testing::Test {
 public:
  Nice::LogisticRegression<T> *lr;
  Nice::Matrix<T> x;
  Nice::Matrix<T> z;
  Nice::Vector<T> y;

  void Instantiate() {
	lr = new Nice::LogisticRegression<T>();
    //lr = Nice::LogisticRegression(x, y);
  }
};

typedef ::testing::Types<float> MyTypes;
TYPED_TEST_CASE(LogisticRegressionTest, MyTypes);


TYPED_TEST(LogisticRegressionTest, Test) {
  this->z.resize(118, 3);
/*  this->z << 0.051267,0.69956,1,
-0.092742,0.68494,1,
-0.21371,0.69225,1,
-0.375,0.50219,1,
-0.51325,0.46564,1,
-0.52477,0.2098,1,
-0.39804,0.034357,1,
-0.30588,-0.19225,1,
0.016705,-0.40424,1,
0.13191,-0.51389,1,
0.38537,-0.56506,1,
0.52938,-0.5212,1,
0.63882,-0.24342,1,
0.73675,-0.18494,1,
0.54666,0.48757,1,
0.322,0.5826,1,
0.16647,0.53874,1,
-0.046659,0.81652,1,
-0.17339,0.69956,1,
-0.47869,0.63377,1,
-0.60541,0.59722,1,
-0.62846,0.33406,1,
-0.59389,0.005117,1,
-0.42108,-0.27266,1,
-0.11578,-0.39693,1,
0.20104,-0.60161,1,
0.46601,-0.53582,1,
0.67339,-0.53582,1,
-0.13882,0.54605,1,
-0.29435,0.77997,1,
-0.26555,0.96272,1,
-0.16187,0.8019,1,
-0.17339,0.64839,1,
-0.28283,0.47295,1,
-0.36348,0.31213,1,
-0.30012,0.027047,1,
-0.23675,-0.21418,1,
-0.06394,-0.18494,1,
0.062788,-0.16301,1,
0.22984,-0.41155,1,
0.2932,-0.2288,1,
0.48329,-0.18494,1,
0.64459,-0.14108,1,
0.46025,0.012427,1,
0.6273,0.15863,1,
0.57546,0.26827,1,
0.72523,0.44371,1,
0.22408,0.52412,1,
0.44297,0.67032,1,
0.322,0.69225,1,
0.13767,0.57529,1,
-0.0063364,0.39985,1,
-0.092742,0.55336,1,
-0.20795,0.35599,1,
-0.20795,0.17325,1,
-0.43836,0.21711,1,
-0.21947,-0.016813,1,
-0.13882,-0.27266,1,
0.18376,0.93348,0,
0.22408,0.77997,0,
0.29896,0.61915,0,
0.50634,0.75804,0,
0.61578,0.7288,0,
0.60426,0.59722,0,
0.76555,0.50219,0,
0.92684,0.3633,0,
0.82316,0.27558,0,
0.96141,0.085526,0,
0.93836,0.012427,0,
0.86348,-0.082602,0,
0.89804,-0.20687,0,
0.85196,-0.36769,0,
0.82892,-0.5212,0,
0.79435,-0.55775,0,
0.59274,-0.7405,0,
0.51786,-0.5943,0,
0.46601,-0.41886,0,
0.35081,-0.57968,0,
0.28744,-0.76974,0,
0.085829,-0.75512,0,
0.14919,-0.57968,0,
-0.13306,-0.4481,0,
-0.40956,-0.41155,0,
-0.39228,-0.25804,0,
-0.74366,-0.25804,0,
-0.69758,0.041667,0,
-0.75518,0.2902,0,
-0.69758,0.68494,0,
-0.4038,0.70687,0,
-0.38076,0.91886,0,
-0.50749,0.90424,0,
-0.54781,0.70687,0,
0.10311,0.77997,0,
0.057028,0.91886,0,
-0.10426,0.99196,0,
-0.081221,1.1089,0,
0.28744,1.087,0,
0.39689,0.82383,0,
0.63882,0.88962,0,
0.82316,0.66301,0,
0.67339,0.64108,0,
1.0709,0.10015,0,
-0.046659,-0.57968,0,
-0.23675,-0.63816,0,
-0.15035,-0.36769,0,
-0.49021,-0.3019,0,
-0.46717,-0.13377,0,
-0.28859,-0.060673,0,
-0.61118,-0.067982,0,
-0.66302,-0.21418,0,
-0.59965,-0.41886,0,
-0.72638,-0.082602,0,
-0.83007,0.31213,0,
-0.72062,0.53874,0,
-0.59389,0.49488,0,
-0.48445,0.99927,0,
-0.0063364,0.99927,0,
0.63265,-0.030612,0;
*/  this->x.resize(10, 2);
    this->x << 2, 5,
               2, 0,
               4, 1,
               5, 1,
               7, 2,
               1, 3,
               2, 1.5,
               3, 5,
               4, 3,
               6, 3.5;
//  this->x.col(0) = this->z.col(0);
//  this->x.col(1) = this->z.col(1);
  this->y.resize(10);
  this->y << 0, 0, 0, 0, 0, 1, 1, 1, 1, 1;
//  this->y << this->z.col(2);
  this->Instantiate();
  this->lr->Fit(this->x, this->y);
  ASSERT_TRUE( 0 == 0);
}
