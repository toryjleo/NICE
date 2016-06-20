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

#include "include/cpu_operations.h"
#include <unistd.h>
#include <iostream>
#include "Eigen/Dense"
#include "include/matrix.h"
#include "include/vector.h"

namespace Nice {

// This function returns the transpose of a matrix
template<typename T>
Matrix<T> CpuOperations<T>::Transpose(const Matrix<T> &a) {
  return a.transpose();  // Return transpose
}

template<typename T>
Vector<T> CpuOperations<T>::Transpose(const Vector<T> &a) {
  return a.transpose();
}

//Returns a matrix that is the logical opposite of the input
template<typename T>
Matrix<T> CpuOperations<T>::LogicalNot(const Matrix<T> &a) {
  Matrix<T> aln = a.replicate(1,1);
  //Iterate through the copied matrix
  for(int r = 0; r < aln.rows(); ++r) {
    for(int c = 0; c < aln.cols(); ++c) {
      if(aln(r,c) == 0) {
        aln(r,c) = 1;
      } else {
        aln(r,c) = 0;
      }
    }
  }
  return aln;
}

//Returns a Vector that is the logical opposite of the input
template<typename T>
Vector<T> CpuOperations<T>::LogicalNot(const Vector<T> &a) {
  Vector<T> aln = a.replicate(1,1);
  //Iterate through copied vector
  for(int i = 0; i < a.size(); ++i) {
    if(a(i) == 0) {
      aln(i) = 1;
    } else {
    	aln(i) = 0;
    }
  }
  return aln;
}

template class CpuOperations<int>;
template class CpuOperations<float>;
template class CpuOperations<double>;

}  //  namespace Nice
