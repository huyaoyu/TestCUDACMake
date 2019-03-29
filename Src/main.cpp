#include <iostream>

#include "Common.hpp"

#include <eigen3/Eigen/Dense>

#include "CUDARoutines/Add.hpp"
#include "CUDARoutines/KernelEigen.hpp"

using namespace Eigen;

int main(void)
{
    int res = 0;

    std::cout << "Call crAdd()." << std::endl;

    res = crAdd();

    std::cout << "Call crExponent()." << std::endl;

    // Prepare a matrix as 1000 x 1000 single-point floating number.
    Matrix<float, Dynamic, Dynamic, RowMajor> x = Matrix<float, Dynamic, Dynamic, RowMajor>::Zero( 1000, 3000 );
    Matrix<float, Dynamic, Dynamic, RowMajor> y = Matrix<float, Dynamic, Dynamic, RowMajor>::Zero( 1000, 3000 );

    for ( int i = 0; i < 1000; i += 1)
    {
        for ( int j = 1; j < 3000; j += 3 )
        {
            x(i, j)   = 1;
            x(i, j+1) = 2;
        }
    }

    res = crExponent( x.data(), 1000, 1000, 3, y.data() );

    std::cout << "y(0, 0) = " << y(0, 0) << "." << std::endl;
    std::cout << "y(1, 1) = " << y(1, 1) << "." << std::endl;
    std::cout << "y(2, 2) = " << y(2, 2) << "." << std::endl;
    std::cout << "y(2, 3) = " << y(2, 3) << "." << std::endl;
    std::cout << "y(2, 6) = " << y(2, 6) << "." << std::endl;
    std::cout << "y(3, 3) = " << y(3, 3) << "." << std::endl;
    std::cout << "y(3, 6) = " << y(3, 6) << "." << std::endl;
    std::cout << "y(0, 1) = " << y(0, 1) << "." << std::endl;
    std::cout << "y(0, 2) = " << y(0, 2) << "." << std::endl;
    std::cout << "y(0, 3) = " << y(0, 3) << "." << std::endl;
    std::cout << "y(0, 4) = " << y(0, 4) << "." << std::endl;
    std::cout << "y(0, 5) = " << y(0, 5) << "." << std::endl;
    std::cout << "y(0, 6) = " << y(0, 6) << "." << std::endl;
    std::cout << "y(1, 0) = " << y(1, 0) << "." << std::endl;
    std::cout << "y(1, 1) = " << y(1, 1) << "." << std::endl;
    std::cout << "y(1, 2) = " << y(1, 2) << "." << std::endl;
    std::cout << "y(1, 3) = " << y(1, 3) << "." << std::endl;
    std::cout << "y(1, 4) = " << y(1, 4) << "." << std::endl;
    std::cout << "y(1, 5) = " << y(1, 5) << "." << std::endl;
    std::cout << "y(1, 6) = " << y(1, 6) << "." << std::endl;
    std::cout << "y(0, 2997) = " << y(0, 2997) << "." << std::endl;
    std::cout << "y(999,  999) = " << y(999,  999) << "." << std::endl;
    std::cout << "y(999, 1000) = " << y(999, 1000) << "." << std::endl;
    std::cout << "y(999, 1001) = " << y(999, 1001) << "." << std::endl;

    std::cout << "End of main." << std::endl;

    return res;
}