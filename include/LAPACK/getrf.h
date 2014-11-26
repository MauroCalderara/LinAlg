/** \file
 *
 *  \brief            xGETRF
 *
 *  \date             Created:  Jul 16, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_LAPACK_GETRF_H_
#define LINALG_LAPACK_GETRF_H_

/* Organization of the namespace:
 *
 *    LinAlg::LAPACK
 *        convenience bindings supporting different locations for Dense<T>
 *
 *    LinAlg::LAPACK::<NAME>
 *        bindings to the <NAME> LAPACK backend
 */

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../CUDA/cuda_checks.h"

#ifdef HAVE_MAGMA
#include <magma.h>
#endif /* HAVE_MAGMA */

#endif /* HAVE_CUDA */


#include "../types.h"
#include "../profiling.h"
#include "../exceptions.h"
#include "../utilities/checks.h"
#include "../dense.h"

#ifndef DOXYGEN_SKIP
extern "C" {

  using LinAlg::I_t;
  using LinAlg::S_t;
  using LinAlg::D_t;
  using LinAlg::C_t;
  using LinAlg::Z_t;

  void fortran_name(sgetrf, SGETRF)(const I_t* m, const I_t* n, S_t* A,
                                    const I_t* lda, I_t* ipiv, int* info);
  void fortran_name(dgetrf, DGETRF)(const I_t* m, const I_t* n, D_t* A,
                                    const I_t* lda, I_t* ipiv, int* info);
  void fortran_name(cgetrf, CGETRF)(const I_t* m, const I_t* n, C_t* A,
                                    const I_t* lda, I_t* ipiv, int* info);
  void fortran_name(zgetrf, ZGETRF)(const I_t* m, const I_t* n, Z_t* A,
                                    const I_t* lda, I_t* ipiv, int* info);
}
#endif

namespace LinAlg {

namespace LAPACK {

namespace FORTRAN {

/** \brief            Compute LU factorization
 *
 *  A <- P * L * U
 *
 *  \param[in]        m
 *
 *  \param[in]        n
 *
 *  \param[in,out]    A
 *
 *  \param[in]        lda
 *
 *  \param[in,out]    ipiv
 *
 *  \param[in,out]    info
 *
 *  See
 *  [DGETRF](http://www.math.utah.edu/software/lapack/lapack-d/dgetrf.html)
 */
inline void xGETRF(I_t m, I_t n, S_t* A, I_t lda, I_t* ipiv, int* info) {

  PROFILING_FUNCTION_HEADER

  fortran_name(sgetrf, SGETRF)(&m, &n, A, &lda, ipiv, info);

}
/** \overload
 */
inline void xGETRF(I_t m, I_t n, D_t* A, I_t lda, I_t* ipiv, int* info) {

  PROFILING_FUNCTION_HEADER

  fortran_name(dgetrf, DGETRF)(&m, &n, A, &lda, ipiv, info);

}
/** \overload
 */
inline void xGETRF(I_t m, I_t n, C_t* A, I_t lda, I_t* ipiv, int* info) {

  PROFILING_FUNCTION_HEADER

  fortran_name(cgetrf, CGETRF)(&m, &n, A, &lda, ipiv, info);

}
/** \overload
 */
inline void xGETRF(I_t m, I_t n, Z_t* A, I_t lda, I_t* ipiv, int* info) {

  PROFILING_FUNCTION_HEADER

  fortran_name(zgetrf, ZGETRF)(&m, &n, A, &lda, ipiv, info);

}

} /* namespace FORTRAN */


#ifdef HAVE_CUDA
namespace CUBLAS {

/** \brief            Compute LU factorization
 *
 *  A <- P * L * U
 *
 *  \param[in]        handle
 *
 *  \param[in]        n
 *                    Number of columns in A (must be the same as the number of 
 *                    rows for CUDA GETRF)
 *
 *  \param[in,out]    A_ptr
 *                    Single pointer to the memory region on the device where 
 *                    the matrix A begins (this is unlike the raw CUBLAS 
 *                    interface and unlike the CUBLAS::xGETRF_batched() 
 *                    interface)
 *
 *  \param[in]        lda
 *                    Leading dimension of A
 *
 *  \param[in,out]    ipiv_device
 *                    Pointer to the memory region on the device where the  
 *                    pivot vector begins. Length of the pivot vector: n. Can 
 *                    also be <nullptr>
 *
 *  \param[in,out]    info
 *                    Reference to a variable in main memory (this is again 
 *                    unlike the raw CUBLAS interface and unlike the 
 *                    CUBLAS::xGETRF_batched() interface)
 */
inline void xGETRF(cublasHandle_t handle, I_t n, S_t* A_ptr, I_t lda, 
                   int* ipiv_device, int& info) {

  PROFILING_FUNCTION_HEADER

  S_t*  Aarray[] = { A_ptr };

  S_t** Aarray_device;
  checkCUDA(cudaMalloc<S_t*>(&Aarray_device, 1 * sizeof(Aarray)));
  checkCUDA(cudaMemcpy(Aarray_device, Aarray, 1 * sizeof(Aarray), \
                       cudaMemcpyHostToDevice));

  int lda_device = lda;

  int* info_device;
  checkCUDA(cudaMalloc<int>(&info_device, 1 * sizeof(int)));

  checkCUBLAS(cublasSgetrfBatched(handle, n, Aarray_device, lda_device, 
                                  ipiv_device, info_device, 1));

  checkCUDA(cudaMemcpy(&info, info_device, 1 * sizeof(int), \
                       cudaMemcpyDeviceToHost));

  checkCUDA(cudaFree(Aarray_device));
  checkCUDA(cudaFree(info_device));

}
/** \overload
 */
inline void xGETRF(cublasHandle_t handle, I_t n, D_t* A_ptr, I_t lda, 
                   int* ipiv_device, int& info) {

  PROFILING_FUNCTION_HEADER

  D_t*  Aarray[] = { A_ptr };

  D_t** Aarray_device;
  checkCUDA(cudaMalloc<D_t*>(&Aarray_device, 1 * sizeof(Aarray)));
  checkCUDA(cudaMemcpy(Aarray_device, Aarray, 1 * sizeof(Aarray), \
                       cudaMemcpyHostToDevice));

  int lda_device = lda;

  int* info_device;
  checkCUDA(cudaMalloc<int>(&info_device, 1 * sizeof(int)));

  checkCUBLAS(cublasDgetrfBatched(handle, n, Aarray_device, lda_device, 
                                  ipiv_device, info_device, 1));

  checkCUDA(cudaMemcpy(&info, info_device, 1 * sizeof(int), \
                       cudaMemcpyDeviceToHost));

  checkCUDA(cudaFree(Aarray_device));
  checkCUDA(cudaFree(info_device));

}
/** \overload
 */
inline void xGETRF(cublasHandle_t handle, I_t n, C_t* A_ptr, I_t lda, 
                   int* ipiv_device, int& info) {

  PROFILING_FUNCTION_HEADER

  C_t*  Aarray[] = { A_ptr };

  C_t** Aarray_device;
  checkCUDA(cudaMalloc<C_t*>(&Aarray_device, 1 * sizeof(Aarray)));
  checkCUDA(cudaMemcpy(Aarray_device, Aarray, 1 * sizeof(Aarray), \
                       cudaMemcpyHostToDevice));

  int* info_device;
  checkCUDA(cudaMalloc<int>(&info_device, sizeof(int)));

  checkCUBLAS(cublasCgetrfBatched(handle, n, Aarray_device, lda, ipiv_device, 
                                  info_device, 1));

  checkCUDA(cudaMemcpy(&info, info_device, 1 * sizeof(int), \
                       cudaMemcpyDeviceToHost));

  checkCUDA(cudaFree(Aarray_device));
  checkCUDA(cudaFree(info_device));

}
/** \overload
 */
inline void xGETRF(cublasHandle_t handle, I_t n, Z_t* A_ptr, I_t lda, 
                   int* ipiv_device, int& info) {

  PROFILING_FUNCTION_HEADER

  Z_t*  Aarray[] = { A_ptr };

  Z_t** Aarray_device;
  checkCUDA(cudaMalloc<Z_t*>(&Aarray_device, 1 * sizeof(Aarray)));
  checkCUDA(cudaMemcpy(Aarray_device, Aarray, 1 * sizeof(Aarray), \
                       cudaMemcpyHostToDevice));

  int* info_device;
  checkCUDA(cudaMalloc<int>(&info_device, sizeof(int)));

  checkCUBLAS(cublasZgetrfBatched(handle, n, Aarray_device, lda, ipiv_device, 
                                  info_device, 1));

  checkCUDA(cudaMemcpy(&info, info_device, 1 * sizeof(int), \
                       cudaMemcpyDeviceToHost));

  checkCUDA(cudaFree(Aarray_device));
  checkCUDA(cudaFree(info_device));

}

/** \brief            Compute LU factorization for multiple matrices
 *
 *  A <- P * L * U
 *
 *  \param[in]        handle
 *
 *  \param[in]        n
 *
 *  \param[in,out]    Aarray[]
 *                    Array, stored on the GPU, that contains the pointers to 
 *                    the begin of the matrices. Length: batchSize 
 *
 *  \param[in]        lda
 *
 *  \param[in,out]    PivotArray
 *                    Array, stored on the GPU, of length: batchSize * n
 *
 *  \param[in,out]    infoArray
 *                    Array, stored on the GPU, of length: batchSize
 *
 *  \param[in]        batchSize
 *
 *  See [CUBLAS Documentation](http://docs.nvidia.com/cuda/cublas/)
 */
inline void xGETRF_batched(cublasHandle_t handle, I_t n, S_t* Aarray[], I_t lda,
                           int* PivotArray, int* infoArray, I_t batchSize) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasSgetrfBatched(handle, n, Aarray, lda, PivotArray, \
                                  infoArray, batchSize));

}
/** \overload
 */
inline void xGETRF_batched(cublasHandle_t handle, I_t n, D_t* Aarray[], I_t lda,
                           int* PivotArray, int* infoArray, I_t batchSize) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasDgetrfBatched(handle, n, Aarray, lda, PivotArray, \
                                  infoArray, batchSize));

}
/** \overload
 */
inline void xGETRF_batched(cublasHandle_t handle, I_t n, C_t* Aarray[], I_t lda,
                           int* PivotArray, int* infoArray, I_t batchSize) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasCgetrfBatched(handle, n, \
                                  reinterpret_cast<cuComplex**>(Aarray), lda, \
                                  PivotArray, infoArray, batchSize));

}
/** \overload
 */
inline void xGETRF_batched(cublasHandle_t handle, I_t n, Z_t* Aarray[], I_t lda,
                           int* PivotArray, int* infoArray, I_t batchSize) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasZgetrfBatched(handle, n, \
                                  reinterpret_cast<cuDoubleComplex**>(Aarray), \
                                  lda, PivotArray, infoArray, batchSize));

}

} /* namespace LinAlg::LAPACK::CUBLAS */

#ifdef HAVE_MAGMA
namespace MAGMA {

/** \brief            Compute LU factorization
 *
 *  A <- P * L * U
 *
 *  \param[in]        m
 *
 *  \param[in]        n
 *
 *  \param[in,out]    A
 *
 *  \param[in]        lda
 *
 *  \param[in,out]    ipiv
 *
 *  \param[in,out]    info
 *
 *  See
 *  [DGETRF](http://www.math.utah.edu/software/lapack/lapack-d/dgetrf.html) or 
 *  the MAGMA sources
 */
inline void xGETRF(I_t m, I_t n, S_t* A, I_t lda, I_t* ipiv, int* info) {

  PROFILING_FUNCTION_HEADER

  magma_sgetrf_gpu(m, n, A, lda, ipiv, info);

}
/** \overload
 */
inline void xGETRF(I_t m, I_t n, D_t* A, I_t lda, I_t* ipiv, int* info) {

  PROFILING_FUNCTION_HEADER

  magma_dgetrf_gpu(m, n, A, lda, ipiv, info);

}
/** \overload
 */
inline void xGETRF(I_t m, I_t n, C_t* A, I_t lda, I_t* ipiv, int* info) {

  PROFILING_FUNCTION_HEADER

  magma_cgetrf_gpu(m, n, A, lda, ipiv, info);

}
/** \overload
 */
inline void xGETRF(I_t m, I_t n, Z_t* A, I_t lda, I_t* ipiv, int* info) {

  PROFILING_FUNCTION_HEADER

  magma_zgetrf_gpu(m, n, A, lda, ipiv, info);

}

} /* namespace LinAlg::LAPACK::MAGMA */
#endif /* HAVE_MAGMA */

#endif /* HAVE_CUDA */


using LinAlg::Utilities::check_device;
using LinAlg::Utilities::check_input_transposed;
#ifdef HAVE_CUDA
using LinAlg::Utilities::check_gpu_handles;
#endif

/** \brief            Compute the LU decomposition of a matrix
 *
 *  A = P * L * U     A is overwritten with L and U
 *
 *  \param[in]        A
 *
 *  \param[in]        ipiv
 *                    ipiv is A.rows()*1 matrix (a vector). When using the
 *                    CUBLAS backend and specifying an empty matrix for ipiv,
 *                    the routine performs a non-pivoting LU decomposition.
 *
 *  \note             The return value of the routine is checked and a
 *                    corresponding exception is thrown if the matrix is
 *                    singular.
 *
 *  \todo             MAGMA's GETRF function also supports non-square matrices.
 */
template <typename T>
inline void xGETRF(Dense<T>& A, Dense<int>& ipiv) {

  PROFILING_FUNCTION_HEADER

#ifndef LINALG_NO_CHECKS
  check_input_transposed(A, "xGETRF(A, ipiv), A:");
# ifndef HAVE_CUDA
  check_input_transposed(ipiv, "xGETRF(A, ipiv), ipiv:");
# else
  if (!ipiv.is_empty()) {
    check_input_transposed(ipiv, "xGETRF(A, ipiv), ipiv:");
  }
# endif

# ifndef HAVE_CUDA
  if (A.rows() != ipiv.rows()) {
    throw excBadArgument("xGETRF(A, ipiv), A, ipiv: must have same number of "
                         "rows");
  }
# else
  if (A.rows() != A.cols() && A._location == Location::GPU) {
    throw excBadArgument("xGETRF(A, ipiv), A: matrix A must be a square matrix "
                         "(CUBLAS restriction)");
  }
  if (!ipiv.is_empty()) {
    if (A.rows() != ipiv.rows()) {
      throw excBadArgument("xGETRF(A, ipiv): A, ipiv: must have same number "
                           "of rows");
    }
  }
# endif
#endif /* LINALG_NO_CHECKS */

  auto n = A.cols();
  auto A_ptr = A._begin();
  auto lda = A._leading_dimension;
  int  info = 0;

  if (A._location == Location::host) {

#ifndef LINALG_NO_CHECKS
    check_device(A, ipiv, "xGETRF(A, ipiv)");
#endif

    auto m = A.cols();
    auto ipiv_ptr = ipiv._begin();
    FORTRAN::xGETRF(m, n, A_ptr, lda, ipiv_ptr, &info);

  }

#ifdef HAVE_CUDA
  else if (A._location == Location::GPU) {

# ifndef LINALG_NO_CHECKS
    check_gpu_handles("xGETRF()");
# endif

# ifndef USE_MAGMA_GETRF

#   ifndef LINALG_NO_CHECKS
    if (!ipiv.is_empty()) {
      check_device(A, ipiv, "xGETRF(A, ipiv)");
    }
#   endif

    using LinAlg::CUDA::CUBLAS::handles;
    auto device_id = A._device_id;
    int* ipiv_ptr = (ipiv.is_empty()) ? NULL : ipiv._begin();

    CUBLAS::xGETRF(handles[device_id], n, A_ptr, lda, ipiv_ptr, info);

# else /* USE_MAGMA_GETRF */

#   ifndef LINALG_NO_CHECKS
    if (ipiv.location() != Location::host) {
      throw excBadArgument("xGETRF(A, ipiv): ipiv must be allocated in main "
                           "memory (using MAGMA's getrf)");
    }
    if (ipiv.is_empty()) {
      throw excBadArgument("xGETRF(A, ipiv): ipiv must not be empty (using "
                           "MAGMA's getrf)");
    }
#   endif

    auto m = A.cols();
    auto ipiv_ptr = ipiv._begin();
    MAGMA::xGETRF(m, n, A_ptr, lda, ipiv_ptr, &info);

# endif /* not USE_MAGMA_GETRF */

  }
#endif /* HAVE_CUDA */

#ifndef LINALG_NO_CHECKS
  else {
    throw excUnimplemented("xGETRF(): LAPACK GETRF not supported on selected "
                           "location");
  }

  if (info != 0) {
    throw excMath("xGETRF(): error: info = %d", info);
  }
#endif

}

} /* namespace LinAlg::LAPACK */

} /* namespace LinAlg */

#endif /* LINALG_LAPACK_GETRF_H_ */
