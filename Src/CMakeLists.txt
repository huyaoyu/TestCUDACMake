cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(TestCUDACMake LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Debug)

add_subdirectory( CUDARoutines )
set( LOCAL_LIBS ${LOCAL_LIBS} cr )

set( PROJECT_INCLUDES "../Includes" )

# find_package (Eigen3 REQUIRED NO_MODULE)

include_directories(
# include
  ${PROJECT_INCLUDES}
)

add_executable( TestCUDACMake main.cpp )

target_link_libraries( TestCUDACMake
    ${LOCAL_LIBS}
    # Eigen3::Eigen
)
