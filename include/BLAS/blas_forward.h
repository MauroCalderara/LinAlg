/** \file
 *
 *  \brief            Forward declaration of BLAS functions required in
 *                    dense.h and sparse.h
 */

#ifndef LINALG_BLAS_FORWARD_H_
#define LINALG_BLAS_FORWARD_H_

#include "../preprocessor.h"

#ifdef HAVE_CUDA
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#include "../types.h"

namespace LinAlg {

namespace BLAS {

#ifdef HAVE_CUDA
namespace cuBLAS {

#ifndef DOXYGEN_SKIP
inline void xGEAM(int, cublasOperation_t, cublasOperation_t, I_t, I_t,
                  const S_t, const S_t*, I_t, const S_t, const S_t*, I_t, S_t*,
                  I_t);
inline void xGEAM(int, cublasOperation_t, cublasOperation_t, I_t, I_t,
                  const D_t, const D_t*, I_t, const D_t, const D_t*, I_t, D_t*,
                  I_t);
inline void xGEAM(int, cublasOperation_t, cublasOperation_t, I_t, I_t,
                  const C_t, const C_t*, I_t, const C_t, const C_t*, I_t, C_t*,
                  I_t);
inline void xGEAM(int, cublasOperation_t, cublasOperation_t, I_t, I_t,
                  const Z_t, const Z_t*, I_t, const Z_t, const Z_t*, I_t, Z_t*,
                  I_t);
#endif

} /* namespace LinAlg::BLAS::cuBLAS */
#endif /* HAVE_CUDA */

} /* namespace LinAlg::BLAS */

} /* namespace LinAlg */

#endif /* LINALG_BLAS_FORWARD_H_ */
