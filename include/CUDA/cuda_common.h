
/** \file
 *
 *  \brief            Shared structures/codes for CUDA
 *
 *  \date             Created:  Jan 1, 2015
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_CUDA_CUDA_COMMON_H_
#define LINALG_CUDA_CUDA_COMMON_H_

namespace LinAlg {

namespace GPU {

/// Global variable to signal the initialization status of the / structures for 
/// GPU handling
//"Defined" in src/CUDA/cuda.cc
extern bool GPU_structures_initialized;
extern int  n_devices;

} /* namespace LinAlg::GPU */

} /* namespace LinAlg */

#ifndef USE_LOCAL_STREAMS

# include <cuda_runtime.h>

# include "../streams.h"
# include "../profiling.h"

namespace LinAlg {

namespace CUDA {

// "Defined" in src/CUDA/cuda.cc
/// Global vector of streams for transfers into the GPU
extern Stream* in_stream;
/// Global vector of streams for transfers out of the GPU
extern Stream* out_stream;
/// Global vector of streams for transfers within the GPU
extern Stream* on_stream;
/// Global vector of streams for computations on the GPU
extern Stream* compute_stream;

/** \brief            Routine to intialize the shared streams
 */
inline void init() {

  PROFILING_FUNCTION_HEADER

  using LinAlg::GPU::GPU_structures_initialized;
  using LinAlg::GPU::n_devices;

  if (GPU_structures_initialized == true) return;

  in_stream      = new Stream[n_devices];
  out_stream     = new Stream[n_devices];
  on_stream      = new Stream[n_devices];
  compute_stream = new Stream[n_devices];

  for (int device_id = 0; device_id < n_devices; ++device_id) {

    in_stream[device_id].set(device_id, true);
    out_stream[device_id].set(device_id, true);
    on_stream[device_id].set(device_id, true);
    compute_stream[device_id].set(device_id, true);
  
  }

}

/** \brief            Routine to destroy the shared streams
 */
inline void destroy() {

  PROFILING_FUNCTION_HEADER

  using LinAlg::GPU::GPU_structures_initialized;

  if (GPU_structures_initialized == false) return;

  delete[] in_stream;
  delete[] out_stream;
  delete[] on_stream;
  delete[] compute_stream;

}

} /* namespace LinAlg::CUDA */

} /* namespace LinAlg */

#endif /* not USE_LOCAL_STREAMS */

#endif /* LINALG_CUDA_CUDA_COMON_H_ */
