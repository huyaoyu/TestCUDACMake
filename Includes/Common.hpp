#ifndef __COMMON_HPP__
#define __COMMON_HPP__

// ====== Make Eigen work with CUDA. ======

// This modification is recommended on the official documentation of Eigen.
// https://eigen.tuxfamily.org/dox-devel/TopicCUDA.html
// At the time being, the version of Eigen is 3.3.90.

#ifdef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#undef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#endif

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#endif /* __COMMON_HPP__ */