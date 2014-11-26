/** \file
 *
 *  \brief            The global GPU handles
 *
 *  \date             Created:  Nov 26, 2014
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

namespace GPU {

// Status handle
bool _handles_are_initialized = false;

} /* namespace LinAlg::GPU */

} /* namespace LinAlg */

#endif /* HAVE_CUDA */
