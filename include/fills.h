/** \file
 *
 *  \brief            Matrix fills.
 *
 *  \date             Created:  Jul 18, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcaldrara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_FILLS_H_
#define LINALG_FILLS_H_

#include <algorithm>      // std::fill_n
#include <random>         // std::mt19337, std::uniform_real_distribution
#include <ctime>          // std::time

#include "dense.h"
#include "sparse.h"
#include "LAPACK/lapack.h"

namespace LinAlg {

namespace Fills {

/** \brief            Fills a matrix with a constant value (preserving sparcity
 *                    for sparse matrices)
 *
 *  \param[in,out]    matrix
 *                    Matrix to operate on.
 *
 *  \param[in]        value
 *                    Value to set the matrix elements to.
 *
 *  \note             For sparse matrices, only the non-zero elements are set to
 *                    the requiested value.
 */
template <typename T>
inline void val(Dense<T>& matrix, T value) {

  LAPACK::xLASET(value, value, matrix);

}
/** \overload
 */
template <typename T>
inline void val(Sparse<T>& matrix, T value) {

#ifndef LINALG_NO_CHECK
  if (matrix.location() != Location::host) {
    throw excUnimplemented("val(): operation not supported for matrices not in "
                           "main memory");
  }
#endif

  auto begin      = matrix._values.get();
  auto n_nonzeros = matrix._n_nonzeros;

  std::fill_n(begin, n_nonzeros, value);

}

/** \brief            Fills a matrix with zeros
 *
 *  \param[in,out]    matrix
 *                    Matrix to operate on.
 *
 *  \note             For sparse matrices, only the non-zero elements are set to
 *                    the requiested value.
 */
template <typename T>
inline void zero(Dense<T>& matrix) {

  val(matrix, cast<T>(0.0));

}
/** \overload
 */
template <typename T>
inline void zero(Sparse<T>& matrix) {

  val(matrix, cast<T>(0.0));

}

/** \brief            Sets diagonal elements to value a, offdiagonal elements
 *                    to value b
 *
 *  \param[in,out]    matrix
 *                    Matrix to operate on.
 *
 *  \param[in]        a
 *                    Value to set the diagonal elements to.
 *
 *  \param[in]        b
 *                    Value to set the offdiagonal elements to.
 */
template <typename T>
inline void val_diag_offdiag(Dense<T>& matrix, T a, T b) {

  LAPACK::xLASET(a, b, matrix);

}

/** \brief            Sets a (square) matrix to unity
 *
 *  \param[in,out]    matrix
 *                    Matrix to operate on.
 *
 */
template <typename T>
inline void unity(Dense<T>& matrix) {

#ifdef LINALG_NO_CHECKS
  if (matrix.rows() != matrix.cols()) {
    throw excBadArgument("unity(): input matrix not square")
  }
#endif

  LAPACK::xLASET(cast<T>(0.0), cast<T>(1.0), matrix);

}

/** \brief            Sets the elements of a matrix to random values using
 *                    LAPACK's xLARNV function
 *
 *  \param[in,out]    matrix
 *                    Matrix to operate on.
 *
 *  \param[in]        distribution
 *                    1: uniform on (0, 1)
 *                    2: uniform on (-1, 1)
 *                    3: normal on (0, 1)
 *                    4: uniform on abs(z) < 1 (unit disc) (C_t, Z_t only)
 *                    5: uniform on abs(z) = 1 (unit circle) (C_t, Z_t only)
 *
 *  \param[in|out]    seed
 *                    A array of 4 integers in [0, 4095] with seed[3] being 
 *                    odd, serving as the seed for the random number 
 *                    generator. On exit, seed is updated.
 *
 */
template <typename T>
inline void lapack_rand(Dense<T>& matrix, I_t distribution, I_t* seed) {

#ifdef LINALG_NO_CHECKS
  if (distribution < 1 || distribution > 5) {
    throw excBadArgument("lapack_rand(matrix, distribution, seed), "
                         "distribution: input value '%d' is invalid (must be "
                         "1, 2, or 3 for this matrix type)");
  } else if ((distribution == 4 || distribution == 5) && 
             (matrix._is_complex() == false)             ) {
    throw excBadArgument("lapack_rand(matrix, distribution, seed), "
                         "distribution: input value '%d' is only supported for "
                         "complex matrices", distribution);
  }
  for (int i = 0; i < 4; ++i) {
    if (seed[i] < 0 || seed[i] > 4095) {
      throw excBadArgument("lapack_rand(matrix, distribution, seed), seed: "
                           "input value seed[%d]=%d is invalid", i, seed[i]);
    }
  }
  if (seed[3] % 2 == 0) {
    throw excBadArgument("lapack_rand(matrix, distribution, seed), seed: "
                         "input value seed[3]=%d is invalid (must be uneven)", 
                         seed[3]);
  }
#endif

  LAPACK::xLARNV(distribution, seed, matrix);

}

/** \brief            Sets the elements of a matrix to uniform random values
 *                    using on the 1D (S_t, D_t) or 2D (C_t, Z_t) unit ball 
 *                    (|x| < 1) using LAPACK's xLARNV function
 *
 *  \param[in,out]    matrix
 *                    Matrix to operate on.
 */
template <typename T>
inline void lapack_urand(Dense<T>& matrix) {

  std::mt19937 random_engine;
  random_engine.seed(std::time(0));
  std::uniform_int_distribution<I_t> random_integer(0, 4095);

  I_t distribution = (matrix._is_complex()) ? 1 : 4;
  I_t seed[] = { random_integer(random_engine), random_integer(random_engine), 
                 random_integer(random_engine), 1 };

  lapack_rand(matrix, distribution, &seed[0]);

}

} /* namespace LinAlg::Fills */

} /* namespace LinAlg */

#endif /* LINALG_FILLS_H_ */
