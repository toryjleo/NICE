\
#include <iostream>
#include <stdio.h>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/matrix.h"

TEST(LogicalNot, BasicMatrixTest) {
  Nice::Matrix<int> m1(4,4);
  m1 << 1, 1, 1, 1,
        1, 0, 1, 1,
        0, 0, 0, 0,
        0, 0, 1, 0;
  Nice::Matrix<int> m2(4,4);
  m2 << 0, 0, 0, 0,
        0, 1, 0, 0,
        1, 1, 1, 1,
        1, 1, 0, 1;
  std::cout << "original matrix: \n" << m1 << std::endl;
  std::cout << "expected output from logicalnot: \n" << m2 << std::endl;
  ASSERT_TRUE( m2.isApprox( Nice::CpuOperations<int>::LogicalNot( m1 ) ) );
}

TEST(LogicalNot, BasicVectorTest) {
  Nice::Vector<int> v1(4);
  v1 << 0, 1, 0, 1;
  Nice::Vector<int> v2(4);
  v2 << 1, 0, 1, 0;
  std::cout << "original vector: \n" << v1 << std::endl;
  std::cout << "expected output from logicalnot: \n" << v2 << std::endl;
  ASSERT_TRUE( v2.isApprox( Nice::CpuOperations<int>::LogicalNot( v1 ) ) );
}


