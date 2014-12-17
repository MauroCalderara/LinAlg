/** \file
 *
 *  \brief            csr2dense (cuSPARSE)
 *
 *  \date             Created:  Jul 16, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_BLAS_CUSPARSE_CSR2DENSE_H_
#define LINALG_BLAS_CUSPARSE_CSR2DENSE_H_

/* Organization of the namespace:
 *
 *    LinAlg::BLAS
 *        bindings to routines handing sparse matrices
 *
 *    LinAlg::BLAS::cuSPARSE
 *        bindings from cuSPARSE backend
 */

#include "../../preprocessor.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../../CUDA/cuda_checks.h"
#endif

#include "../../types.h"
#include "../../profiling.h"
#include "../../exceptions.h"
#include "../../utilities/checks.h"
#include "../../sparse.h"

namespace LinAlg {

namespace BLAS {

#ifdef HAVE_CUDA
namespace cuSPARSE {

/** \brief            Converts a CSR matrix to a dense matrix in ColMajor format
 *
 *  \param[in]        handle
 *
 *  \param[in]        m
 *
 *  \param[in]        n
 *
 *  \param[in]        descrA
 *
 *  \param[in]        csrValA
 *
 *  \param[in]        csrRowPtrA
 *
 *  \param[in]        csrColIndA
 *
 *  \param[in,out]    A
 *
 *  \param[in]        lda
 *
 *  See [CUSPARSE](http://docs.nvidia.com/cuda/cusparse/)
 */
inline void xcsr2dense(cusparseHandle_t handle, int m, int n, 
                      const cusparseMatDescr_t descrA, const S_t* csrValA,
                      const int* csrRowPtrA, const int* csrColIndA, S_t* A,
                      int lda) {

  PROFILING_FUNCTION_HEADER

  checkCUSPARSE(cusparseScsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, \
                                   csrColIndA, A, lda));

}
/** \overload
 */
inline void xcsr2dense(cusparseHandle_t handle, int m, int n, 
                      const cusparseMatDescr_t descrA, const D_t* csrValA,
                      const int* csrRowPtrA, const int* csrColIndA, D_t* A,
                      int lda) {

  PROFILING_FUNCTION_HEADER

  checkCUSPARSE(cusparseDcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, \
                                   csrColIndA, A, lda));

}
/** \overload
 */
inline void xcsr2dense(cusparseHandle_t handle, int m, int n, 
                      const cusparseMatDescr_t descrA, const C_t* csrValA,
                      const int* csrRowPtrA, const int* csrColIndA, C_t* A,
                      int lda) {

  PROFILING_FUNCTION_HEADER

  checkCUSPARSE(cusparseCcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, \
                                   csrColIndA, A, lda));

}
/** \overload
 */
inline void xcsr2dense(cusparseHandle_t handle, int m, int n, 
                      const cusparseMatDescr_t descrA, const Z_t* csrValA,
                      const int* csrRowPtrA, const int* csrColIndA, Z_t* A,
                      int lda) {

  PROFILING_FUNCTION_HEADER

  checkCUSPARSE(cusparseZcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, \
                                   csrColIndA, A, lda));

}

// Convenience bindings (bindings for Sparse<T>)
/** \brief            Converts a CSR matrix to a dense matrix in ColMajor format
 *
 *  \param[in]        A
 *
 *  \param[in,out]    B
 */
template <typename T>
inline void xcsr2dense(const Sparse<T>& A, Dense<T>& B) {

  PROFILING_FUNCTION_HEADER

#ifndef LINALG_NO_CHECKS
  using LinAlg::Utilities::check_gpu_handles;
  using LinAlg::Utilities::check_format;
  using LinAlg::Utilities::check_input_transposed;
  using LinAlg::Utilities::check_output_transposed;
  using LinAlg::Utilities::check_dimensions;
  using LinAlg::Utilities::check_device;
  using LinAlg::CUDA::cuSPARSE::handles;

  check_gpu_handles("xcsr2dense()");
  check_format(Format::CSR, A, "xcsr2dense(A, B), A");
  check_input_transposed(A, "xcsr2dense(A, B), A");
  check_format(Format::ColMajor, B, "xcsr2dense(A, B), B");
  check_output_transposed(B, "xcsr2dense(A, B), B");
  check_dimensions(B.rows(), B.cols(), A, "xcsr2dense(A, B), A)");
  if (A._location != Location::GPU) {
    throw excBadArgument("xcsr2dense(A, A), A: matrix must reside on GPU");
  }
  if (B._location != Location::GPU) {
    throw excBadArgument("xcsr2dense(A, B), B: matrix must reside on GPU");
  }
  check_device(A, B, "xcsr2dense()");
#endif

  auto device_id  = A._device_id;
  auto m          = B.rows();
  auto n          = B.cols();
  //auto descr    = A._cusparse_descriptor;
  auto csrValA    = A._values.get();
  auto csrRowPtrA = A._edges.get();
  auto csrColIndA = A._indices.get();
  auto A_ptr      = B._memory.get();
  auto lda        = B._leading_dimension;

  xcsr2dense(handles[device_id], m, n, A._cusparse_descriptor, csrValA, 
             csrRowPtrA, csrColIndA, A_ptr, lda);

}
#endif /* HAVE_CUDA */

} /* namespace LinAlg::BLAS::cuSPARSE */

} /* namespace LinAlg::BLAS */

} /* namespace LinAlg */

#endif /* LINALG_BLAS_CUSPARSE_CSR2DENSE_H_ */
