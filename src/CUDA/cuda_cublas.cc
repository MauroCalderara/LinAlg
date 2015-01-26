/** \file
 *
 *  \brief            The global cuBLAS handle vector
 *
 *  \date             Created:  Jan 16, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */

#ifdef HAVE_CUDA

# ifndef USE_LOCAL_CUDA_HANDLES

#   include <cublas_v2.h>

#   include "CUDA/cuda_cublas.h"

namespace LinAlg {

namespace CUDA {

namespace cuBLAS {

// Global array of cuBLAS handles (extern in cuda_cublas.h)
cublasHandle_t* handles;

} /* namespace LinAlg::CUDA::cuBLAS */

} /* namespace LinAlg::CUDA */

} /* namespace LinAlg */

# endif /* not USE_LOCAL_CUDA_HANDLES */

#endif /* HAVE_CUDA */
