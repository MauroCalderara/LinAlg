/** \file             cuda_checks.h
 *
 *  \brief            Routines to check for CUDA/CUBLAS/CUSPARSE error messages
 *
 *  \date             Created:  Jul 12, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_CUDA_CUDA_CHECKS_H_
#define LINALG_CUDA_CUDA_CHECKS_H_

#ifdef HAVE_CUDA

#include <vector>         // std::vector for cublas handles
#include <memory>         // std::shared_ptr
#include <cuda_runtime.h> // various CUDA routines
#include <cublas_v2.h>
#include <cusparse_v2.h>

#include "types.h"
#include "exceptions.h"   // LinAlg::excLinAlg

#ifndef LINALG_NO_CHECKS

/** \def              checkCUDA(ans)
 *
 *  \brief            Macro to facilitate CUDA error checking
 */
#define checkCUDA(expr) { LinAlg::CUDA::check_CUDA((expr), \
                                                      __FILE__, __LINE__); }

/** \def              checkCUBLAS(ans)
 *
 *  \brief            Macro to facilitate CUBLAS error checking
 */
#define checkCUBLAS(expr) { LinAlg::CUDA::check_CUBLAS((expr), \
                                                      __FILE__, __LINE__); }

/** \def              checkCUSPARSE(ans)
 *
 *  \brief            Macro to facilitiate CUSPARSE error checking
 */
#define checkCUSPARSE(expr) { LinAlg::CUDA::check_CUSPARSE((expr), \
                                                      __FILE__, __LINE__); }

#else

#define checkCUDA(expr) expr
#define checkCUBLAS(expr) expr
#define checkCUSPARSE(expr) expr

#endif /* LINALG_NO_CHECKS */


namespace LinAlg {

namespace CUDA {

#ifndef LINALG_NO_CHECKS

/** \brief            Function to check the return value of CUDA calls.
 *
 *  \param[in]        code
 *                    Error code to check.
 *
 *  \param[in]        file
 *                    Name of the file in which the call to be checked is made.
 *
 *  \param[in]        line
 *                    Line number on which the call to be checked is made.
 *
 *  \post             If code indicates an error, the application is terminated.
 */
inline void check_CUDA(cudaError_t code, const char* file, int line) {

  if (code == cudaSuccess) {

    return;

  } else {

    throw excCUDAError("%s in %s:%d", cudaGetErrorString(code), file, line);

  }

};

/** \brief            Function to check the return value of CUBLAS calls.
 *
 *  \param[in]        code
 *                    Error code to check.
 *
 *  \param[in]        file
 *                    Name of the file in which the call to be checked is made.
 *
 *  \param[in]        line
 *                    Line number on which the call to be checked is made.
 *
 *  \post             If code indicates an error, the application is terminated.
 */
inline void check_CUBLAS(cublasStatus_t code, const char *file, int line) {

  if (code == CUBLAS_STATUS_SUCCESS) {

    return;

  } else {

    std::string cublas_error_string;

    switch (code) {

      case CUBLAS_STATUS_NOT_INITIALIZED:
        cublas_error_string = "CUBLAS_STATUS_NOT_INITIALIZED";
        break;
      case CUBLAS_STATUS_ALLOC_FAILED:
        cublas_error_string = "CUBLAS_STATUS_ALLOC_FAILED";
        break;
      case CUBLAS_STATUS_INVALID_VALUE:
        cublas_error_string = "CUBLAS_STATUS_INVALID_VALUE";
        break;
      case CUBLAS_STATUS_ARCH_MISMATCH:
        cublas_error_string = "CUBLAS_STATUS_ARCH_MISMATCH";
        break;
      case CUBLAS_STATUS_MAPPING_ERROR:
        cublas_error_string = "CUBLAS_STATUS_MAPPING_ERROR";
        break;
      case CUBLAS_STATUS_EXECUTION_FAILED:
        cublas_error_string = "CUBLAS_STATUS_EXECUTION_FAILED";
        break;
      case CUBLAS_STATUS_INTERNAL_ERROR:
        cublas_error_string = "CUBLAS_STATUS_INTERNAL_ERROR";
        break;
      default:
        cublas_error_string = "unknown CUBLAS error";
        break;

    }

    throw excCUBLASError("%s in %s:%d", cublas_error_string.c_str(), file,
                         line);

  }

};

/** \brief            Function to check the return value of CUSPARSE calls.
 *
 *  \param[in]        code
 *                    Error code to check.
 *
 *  \param[in]        file
 *                    Name of the file in which the call to be checked is made.
 *
 *  \param[in]        line
 *                    Line number on which the call to be checked is made.
 *
 *  \post             If code indicates an error, the application is terminated.
 */
inline void check_CUSPARSE(cusparseStatus_t code, const char *file, int line) {

  if (code == CUSPARSE_STATUS_SUCCESS) {

    return;

  } else {

    std::string cusparse_error_string;

    switch (code) {

      case CUSPARSE_STATUS_NOT_INITIALIZED:
        cusparse_error_string = "CUSPARSE_STATUS_NOT_INITIALIZED";
        break;
      case CUSPARSE_STATUS_ALLOC_FAILED:
        cusparse_error_string = "CUSPARSE_STATUS_ALLOC_FAILED";
        break;
      case CUSPARSE_STATUS_INVALID_VALUE:
        cusparse_error_string = "CUSPARSE_STATUS_INVALID_VALUE";
        break;
      case CUSPARSE_STATUS_ARCH_MISMATCH:
        cusparse_error_string = "CUSPARSE_STATUS_ARCH_MISMATCH";
        break;
      case CUSPARSE_STATUS_MAPPING_ERROR:
        cusparse_error_string = "CUSPARSE_STATUS_MAPPING_ERROR";
        break;
      case CUSPARSE_STATUS_EXECUTION_FAILED:
        cusparse_error_string = "CUSPARSE_STATUS_EXECUTION_FAILED";
        break;
      case CUSPARSE_STATUS_INTERNAL_ERROR:
        cusparse_error_string = "CUSPARSE_STATUS_INTERNAL_ERROR";
        break;
      case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        cusparse_error_string = "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        break;
      default:
        cusparse_error_string = "unknown CUSPARSE error";
        break;

    }

    throw excCUSPARSEError("%s in %s:%d", cusparse_error_string.c_str(), file,
                           line);

  }

};

#endif /* LINALG_NO_CHECKS */

} /* namespace LinAlg::CUDA */

} /* namespace LinAlg */

#endif /* HAVE_CUDA */

#endif /* LINALG_CUDA_CUDA_CHECKS_H_ */
