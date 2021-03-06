/** \file
 *
 *  \brief            xAXPY (BLAS-1)
 *
 *  \date             Created:  Jul 16, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_BLAS_AXPY_H_
#define LINALG_BLAS_AXPY_H_

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
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../CUDA/cuda_checks.h"
#endif

#include "../types.h"
#include "../profiling.h"
#include "../exceptions.h"
#include "../dense.h"

#ifndef DOXYGEN_SKIP
extern "C" {

  using LinAlg::I_t;
  using LinAlg::S_t;
  using LinAlg::D_t;
  using LinAlg::C_t;
  using LinAlg::Z_t;

  void fortran_name(saxpy, SAXPY)(const I_t* n, const S_t* alpha, const S_t* x,
                                  const I_t* incx, S_t* y, const I_t* incy);
  void fortran_name(daxpy, DAXPY)(const I_t* n, const D_t* alpha, const D_t* x,
                                  const I_t* incx, D_t* y, const I_t* incy);
  void fortran_name(caxpy, CAXPY)(const I_t* n, const C_t* alpha, const C_t* x,
                                  const I_t* incx, C_t* y, const I_t* incy);
  void fortran_name(zaxpy, ZAXPY)(const I_t* n, const Z_t* alpha, const Z_t* x,
                                  const I_t* incx, Z_t* y, const I_t* incy);
}
#endif


namespace LinAlg {

namespace BLAS {

namespace FORTRAN {

/** \brief            Vector addition with prefactor
 *
 *  y = alpha * x + y
 *
 *  \param[in]        n
 *
 *  \param[in]        alpha
 *
 *  \param[in]        x
 *
 *  \param[in]        incx
 *
 *  \param[in,out]    y
 *
 *  \param[in]        incy
 *
 *  See [DAXPY](http://www.mathkeisan.com/usersguide/man/daxpy.html)
 */
inline void xAXPY(I_t n, S_t alpha, S_t* x, I_t incx, S_t* y, I_t incy) {

  PROFILING_FUNCTION_HEADER

  fortran_name(saxpy, SAXPY)(&n, &alpha, x, &incx, y, &incy);

}
/** \overload
 */
inline void xAXPY(I_t n, D_t alpha, D_t* x, I_t incx, D_t* y, I_t incy) {

  PROFILING_FUNCTION_HEADER

  fortran_name(daxpy, DAXPY)(&n, &alpha, x, &incx, y, &incy);

}
/** \overload
 */
inline void xAXPY(I_t n, C_t alpha, C_t* x, I_t incx, C_t* y, I_t incy) {

  PROFILING_FUNCTION_HEADER

  fortran_name(caxpy, CAXPY)(&n, &alpha, x, &incx, y, &incy);

}
/** \overload
 */
inline void xAXPY(I_t n, Z_t alpha, Z_t* x, I_t incx, Z_t* y, I_t incy) {

  PROFILING_FUNCTION_HEADER

  fortran_name(zaxpy, ZAXPY)(&n, &alpha, x, &incx, y, &incy);

}

} /* namespace FORTRAN */

#ifdef HAVE_CUDA
namespace cuBLAS {

/** \brief            Vector addition with prefactor
 *
 *  y = alpha * x + y
 *
 *  \param[in]        handle
 *
 *  \param[in]        n
 *
 *  \param[in]        alpha
 *
 *  \param[in]        x
 *
 *  \param[in]        incx
 *
 *  \param[in,out]    y
 *
 *  \param[in]        incy
 *
 *  See [cuBLAS Documentation](http://docs.nvidia.com/cuda/cublas/)
 */
inline void xAXPY(cublasHandle_t handle, I_t n, S_t alpha, S_t* x, I_t incx,
                  S_t* y, I_t incy) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasSaxpy(handle, n, &alpha, x, incx, y, incy));

}
/** \overload
 */
inline void xAXPY(cublasHandle_t handle, I_t n, D_t alpha, D_t* x, I_t incx,
                  D_t* y, I_t incy) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasDaxpy(handle, n, &alpha, x, incx, y, incy));

}
/** \overload
 */
inline void xAXPY(cublasHandle_t handle, I_t n, C_t alpha, C_t* x, I_t incx,
                  C_t* y, I_t incy) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasCaxpy(handle, n, &alpha, x, incx, y, incy));

}
/** \overload
 */
inline void xAXPY(cublasHandle_t handle, I_t n, Z_t alpha, Z_t* x, I_t incx,
                  Z_t* y, I_t incy) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasZaxpy(handle, n, &alpha, x, incx, y, incy));

}

} /* namespace cuBLAS */
#endif /* HAVE_CUDA */

// No convenience bindings (BLAS level 1)

} /* namespace LinAlg::BLAS */

} /* namespace LinAlg */

#endif /* LINALG_BLAS_AXPY_H_ */
