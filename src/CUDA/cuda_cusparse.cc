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
#include <cusparse_v2.h>

#include "CUDA/cuda_cusparse.h"

namespace LinAlg {

namespace CUDA {

namespace cuSPARSE {

// The vector of handles (extern in cuda_cusparse.h)
std::vector<cusparseHandle_t> handles;

} /* namespace LinAlg::CUDA::cuSPARSE */

} /* namespace LinAlg::CUDA */

} /* namespace LinAlg */

#endif /* HAVE_CUDA */
