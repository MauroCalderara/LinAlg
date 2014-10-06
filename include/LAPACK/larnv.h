/** \file
 *
 *  \brief            xLARNV
 *
 *  \date             Created:  Jul 16, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_LAPACK_LARNV_H_
#define LINALG_LAPACK_LARNV_H_

/* Organization of the namespace:
 *
 *    LinAlg::LAPACK
 *        convenience bindings supporting different locations for Dense<T>
 *
 *    LinAlg::LAPACK::<NAME>
 *        bindings to the <NAME> LAPACK backend
 */

#include "../types.h"
#include "../exceptions.h"
#include "../dense.h"

#ifndef DOXYGEN_SKIP
extern "C" {

  using LinAlg::I_t;
  using LinAlg::S_t;
  using LinAlg::D_t;
  using LinAlg::C_t;
  using LinAlg::Z_t;

  void fortran_name(slarnv, SLARNV)(const int* idist, int* iseed, const int* n,
                               S_t* x);
  void fortran_name(dlarnv, DLARNV)(const int* idist, int* iseed, const int* n,
                               D_t* x);
  void fortran_name(clarnv, CLARNV)(const int* idist, int* iseed, const int* n,
                               C_t* x);
  void fortran_name(zlarnv, ZLARNV)(const int* idist, int* iseed, const int* n,
                               Z_t* x);
}
#endif

namespace LinAlg {

namespace LAPACK {

namespace FORTRAN {

/** \brief            xLARNV
 *
 *  Return a vector of n random numbers from a uniform or normal distribution.
 *
 *  \param[in]        idist
 *
 *  \param[in]        iseed
 *
 *  \param[in]        n
 *
 *  \param[in,out]    x
 *
 *  See [DLARNV](http://www.mathkeisan.com/usersguide/man/dlarnv.html)
 */
inline void xLARNV(I_t idist, I_t* iseed, I_t n, S_t* x){
  fortran_name(slarnv, SLARNV)(&idist, iseed, &n, x);
};
/** \overload
 */
inline void xLARNV(I_t idist, I_t* iseed, I_t n, D_t* x){
  fortran_name(dlarnv, DLARNV)(&idist, iseed, &n, x);
};
/** \overload
 */
inline void xLARNV(I_t idist, I_t* iseed, I_t n, C_t* x){
  fortran_name(clarnv, CLARNV)(&idist, iseed, &n, x);
};
/** \overload
 */
inline void xLARNV(I_t idist, I_t* iseed, I_t n, Z_t* x){
  fortran_name(zlarnv, ZLARNV)(&idist, iseed, &n, x);
};

} /* namespace FORTRAN */


/** \brief            xLARNV
 *
 *  Fill a matrix with random numbers from a uniform or normal distribution.  
 *  The parameters idist and iseed have the same meaning as with the standard 
 *  LAPACK function, however, X can be a matrix as well as a vector.
 *
 *  \param[in]        idist
 *
 *  \param[in]        iseed
 *
 *  \param[in,out]    X
 *
 *  See [DLARNV](http://www.mathkeisan.com/usersguide/man/dlarnv.html)
 */
template <typename T>
inline void xLARNV(I_t idist, I_t* iseed, Dense<T>& X) {

  if (X.is_empty()) {
    return;
  }

  if (X._location == Location::host) {

    if ((X._format == Format::ColMajor && X._rows == X._leading_dimension) || 
        (X._format == Format::RowMajor && X._cols == X._leading_dimension)  ) {
      
      // The matrix is continuous in memory, one call to xLARNV suffices
      FORTRAN::xLARNV(idist, iseed, X._rows * X._cols, X._begin());

    } else if (X._format == Format::ColMajor) {

      for (I_t col = 0; col < X._cols; ++col) {
      
        auto col_start = X._begin() + col * X._leading_dimension;
        FORTRAN::xLARNV(idist, iseed, X._rows, col_start);
      
      }

    } else {
    
      for (I_t row = 0; row < X._rows; ++row) {
      
        auto row_start = X._begin() + row * X._leading_dimension;
        FORTRAN::xLARNV(idist, iseed, X._cols, row_start);
      
      }
    
    }
    
  }

#ifndef LINALG_NO_CHECKS
  else {
    throw excUnimplemented("xLARNV(): LAPACK LARNV not supported on selected "
                           "location");
  }
#endif

};

} /* namespace LinAlg::LAPACK */

} /* namespace LinAlg */

#endif /* LINALG_LAPACK_LARNV_H_ */
