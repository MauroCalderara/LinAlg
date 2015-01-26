/** \file
 *
 *  \brief            xCOPY (BLAS-1)
 *
 *  \date             Created:  Jul 16, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_BLAS_COPY_H_
#define LINALG_BLAS_COPY_H_

/* Organization of the namespace:
 *
 *    LinAlg::BLAS
 *        convenience bindings supporting different locations for Dense<T>
 *
 *    LinAlg::BLAS::<NAME>
 *        bindings to the <NAME> BLAS backend
 */

#include "../preprocessor.h"

#ifdef HAVE_CUDA
# include <cuda_runtime.h>
# include <cublas_v2.h>
# include "../CUDA/cuda_checks.h"
# include "../CUDA/cuda_cublas.h"
#endif

#include "../types.h"
#include "../profiling.h"

#include "../utilities/checks.h"
#include "../dense.h"

#ifndef DOXYGEN_SKIP
extern "C" {

  using LinAlg::I_t;
  using LinAlg::S_t;
  using LinAlg::D_t;
  using LinAlg::C_t;
  using LinAlg::Z_t;

  void fortran_name(scopy, SCOPY)(const I_t* n, const S_t* x, const I_t* incx,
                                  S_t* y, const I_t* incy);
  void fortran_name(dcopy, DCOPY)(const I_t* n, const D_t* x, const I_t* incx,
                                  D_t* y, const I_t* incy);
  void fortran_name(ccopy, CCOPY)(const I_t* n, const C_t* x, const I_t* incx,
                                  C_t* y, const I_t* incy);
  void fortran_name(zcopy, ZCOPY)(const I_t* n, const Z_t* x, const I_t* incx,
                                  Z_t* y, const I_t* incy);
}
#endif

namespace LinAlg {

namespace BLAS {

namespace FORTRAN {


/** \brief            Vector copy
 *
 *  y <- x
 *
 *  \param[in]        n
 *
 *  \param[in]        x
 *
 *  \param[in]        incx
 *
 *  \param[in]        y
 *
 *  \param[in]        incy
 *
 *  See [DCOPY](http://www.mathkeisan.com/usersguide/man/dcopy.html)
 */
inline void xCOPY(const int n, const S_t* x, const int incx, S_t* y,
                  const int incy) {

  PROFILING_FUNCTION_HEADER

  fortran_name(scopy, SCOPY)(&n, x, &incx, y, &incy);

}
/** \overload
 */
inline void xCOPY(const int n, const D_t* x, const int incx, D_t* y,
                  const int incy) {

  PROFILING_FUNCTION_HEADER

  fortran_name(dcopy, DCOPY)(&n, x, &incx, y, &incy);

}
/** \overload
 */
inline void xCOPY(const int n, const C_t* x, const int incx, C_t* y,
                  const int incy) {

  PROFILING_FUNCTION_HEADER

  fortran_name(ccopy, CCOPY)(&n, x, &incx, y, &incy);

}
/** \overload
 */
inline void xCOPY(const int n, const Z_t* x, const int incx, Z_t* y,
                  const int incy) {

  PROFILING_FUNCTION_HEADER

  fortran_name(zcopy, ZCOPY)(&n, x, &incx, y, &incy);

}

} /* namespace LinAlg::BLAS::FORTRAN */

#ifdef HAVE_CUDA
namespace cuBLAS {

/** \brief            Vector copy
 *
 *  y <- x
 *
 *  \param[in]        handle
 *
 *  \param[in]        n
 *
 *  \param[in]        x
 *
 *  \param[in]        incx
 *
 *  \param[in]        y
 *
 *  \param[in]        incy
 *
 *  See [cuBLAS Documentation](http://docs.nvidia.com/cuda/cublas/)
 */
inline void xCOPY(cublasHandle_t handle, const int n, const S_t* x,
                  const int incx, S_t* y, const int incy) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasScopy(handle, n, x, incx, y, incy));

}
/** \overload
 */
inline void xCOPY(cublasHandle_t handle, const int n, const D_t* x,
                  const int incx, D_t* y, const int incy) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasDcopy(handle, n, x, incx, y, incy));

}
/** \overload
 */
inline void xCOPY(cublasHandle_t handle, const int n, const C_t* x,
                  const int incx, C_t* y, const int incy) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasCcopy(handle, n, x, incx, y, incy));

}
/** \overload
 */
inline void xCOPY(cublasHandle_t handle, const int n, const Z_t* x,
                  const int incx, Z_t* y, const int incy) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasZcopy(handle, n, x, incx, y, incy));

}

} /* namespace LinAlg::BLAS::cuBLAS */
#endif

} /* namespace LinAlg::BLAS */

} /* namespace LinAlg */

#endif /* LINALG_BLAS_COPY_H_ */
