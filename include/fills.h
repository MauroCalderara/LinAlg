/** \file             fills.h
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

#include "dense.h"
#include "sparse.h"

namespace LinAlg {

namespace Fills {

/** \brief            Fills a matrix with a constant value (preserving sparcity)
 *
 *  \param[in|out]    matrix
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

  auto leading_dimension = matrix._leading_dimension;
  auto rows              = matrix._rows;
  auto cols              = matrix._cols;
  auto begin             = matrix._begin();

  // For contiguous matrices we can do it in one go
  if (leading_dimension == rows) {

    std::fill_n(begin, rows * cols, value);

  } else { // Fill each column separately

    for (int column = 0; column < cols; ++column) {

      std::fill_n(begin + (column * leading_dimension), rows, value);

    }

  }

}
/** \overload
 */
template <typename T>
inline void val(Sparse<T>& matrix, T value) {

  auto begin      = matrix._values.get();
  auto n_nonzeros = matrix._n_nonzeros;

  std::fill_n(begin, n_nonzeros, value);

}

/** \brief            Fills a matrix with zeros
 *
 *  \param[in|out]    matrix
 *                    Matrix to operate on.
 *
 *  \note             For sparse matrices, only the non-zero elements are set to
 *                    the requiested value.
 */
template <typename T>
inline void zero(Dense<T>& matrix) {

  // In IEEE 754 floating point and on many systems also integers could be set
  // to zero using memset. Depending on the platform this could go terribly
  // wrong but iff you feel like you need this feel free to override the below
  // pass through with something using memset. Remember that this is memory
  // bound anyway.

  val(matrix, T(0));

}
/** \overload
 */
template <typename T>
inline void zero(Sparse<T>& matrix) {

  val(matrix, T(0));

}

} /* namespace LinAlg::Fills */

} /* namespace LinAlg */

#endif /* LINALG_FILLS_H_ */
