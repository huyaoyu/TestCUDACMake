cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(CUDARoutines LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Debug)

set( PROJECT_INCLUDES "../../Includes" )

include_directories(
# include
  ${PROJECT_INCLUDES}
)

file(GLOB_RECURSE SRC_LIST *.cu *.cpp)

if (SRC_LIST)
	foreach (SFILE ${SRC_LIST})
		message(STATUS "CUDA: ${SFILE}")
	endforeach (SFILE)
endif (SRC_LIST) 

# find_package(Eigen3 REQUIRED NO_MODULE)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -g -G
)

add_library( cr ${SRC_LIST} )

target_compile_features( cr PUBLIC cxx_std_11 )
set_target_properties( cr PROPERTIES CUDA_SEPARABLE_COMPILATION ON )

# target_link_libraries( cr Eigen3::Eigen )

