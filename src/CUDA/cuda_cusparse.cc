/** \file
 *
 *  \brief            The global cuSPARSE handle vector
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

#   include <cusparse_v2.h>

#   include "CUDA/cuda_cusparse.h"

namespace LinAlg {

namespace CUDA {

namespace cuSPARSE {

// Global array of cuSPARSE handles (extern in cuda_cusparse.h)
cusparseHandle_t* handles;

} /* namespace LinAlg::CUDA::cuSPARSE */

} /* namespace LinAlg::CUDA */

} /* namespace LinAlg */

# endif /* not USE_LOCAL_CUDA_HANDLES */

#endif /* HAVE_CUDA */
