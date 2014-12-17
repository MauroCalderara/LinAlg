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
#include <cublas_v2.h>

#include "CUDA/cuda_cublas.h"

namespace LinAlg {

namespace CUDA {

namespace cuBLAS {

// The vector of handles (extern in cuda_cublas.h)
std::vector<cublasHandle_t> handles;

} /* namespace LinAlg::CUDA::cuBLAS */

} /* namespace LinAlg::CUDA */

} /* namespace LinAlg */

#endif /* HAVE_CUDA */
