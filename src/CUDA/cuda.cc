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
#include <vector>
#include <cublas_v2.h>

#include "CUDA/cuda.h"

#include "streams.h"

namespace LinAlg {

#ifdef USE_GLOBAL_TRANSFER_STREAMS
namespace CUDA {

//std::vector<CUDAStream> in_stream;
//std::vector<CUDAStream> out_stream;
//std::vector<CUDAStream> on_stream;
//std::vector<CUDAStream> compute_stream;
CUDAStream in_stream;
CUDAStream out_stream;
CUDAStream on_stream;

}
#endif

namespace GPU {

// GPU status
bool _GPU_structures_initialized = false;

} /* namespace LinAlg::GPU */

} /* namespace LinAlg */

#endif /* HAVE_CUDA */
