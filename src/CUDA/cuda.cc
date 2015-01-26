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

# ifndef USE_LOCAL_TRANSFER_STREAMS
#   include <cublas_v2.h>
#   include "CUDA/cuda.h"
#   include "streams.h"
# endif

namespace LinAlg {

namespace GPU {

// GPU status
bool GPU_structures_initialized = false;
int  n_devices = 0;

} /* namespace LinAlg::GPU */

# ifndef USE_LOCAL_TRANSFER_STREAMS
namespace CUDA {

Stream* in_stream;
Stream* out_stream;
Stream* on_stream;
Stream* compute_stream;

} /* namespace LinAlg::CUDA */
# endif

} /* namespace LinAlg */

#endif /* HAVE_CUDA */
