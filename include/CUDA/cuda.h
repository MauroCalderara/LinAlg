/** \file
 *
 *  \brief            Inclusion of all CUDA function headers
 *
 *  \date             Created:  Jul 12, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_CUDA_CUDA_H_
#define LINALG_CUDA_CUDA_H_

#include <cuda_runtime.h>

#include "../preprocessor.h"

#ifdef HAVE_MAGMA
# include <magma.h>
#endif

// Keep this in alphabetical order
#include "cuda_checks.h"
#include "cuda_cublas.h"
#include "cuda_cusparse.h"
#include "cuda_memory_allocation.h"

#include "../streams.h"

namespace LinAlg {

#ifdef USE_GLOBAL_TRANSFER_STREAMS
namespace CUDA {

// "Defined" in src/CUDA/cuda.cc
/// Global vector of streams for transfers into the GPU
//extern std::vector<Stream> in_stream;
extern Stream in_stream;
/// Global vector of streams for transfers out of the GPU
//extern std::vector<Stream> out_stream;
extern Stream out_stream;
/// Global vector of streams for transfers within the GPU
//extern std::vector<Stream> on_stream;
extern Stream on_stream;
/// Global vector of streams for computations on the GPU
//extern std::vector<tream> compute_stream;

/** \brief            Routine to intialize the shared streams
 */
inline void init() {

  /*
  int n_devices = 0;
  checkCUDA(cudaGetDeviceCount(&n_devices));

  in_stream.resize(n_devices);
  out_stream.resize(n_devices);
  on_stream.resize(n_devices);
  compute_stream.resize(n_devices);

  for (int device_id = 0; device_id < n_devices; ++device_id) {

    in_stream[device_id].set(device_id, false);
    out_stream[device_id].set(device_id, false);
    on_stream[device_id].set(device_id, false);
    compute_stream[device_id].set(device_id, false);
  
  }
  */

  in_stream.set(0, false);
  out_stream.set(0, false);
  on_stream.set(0, false);

}

} /* namespace LinAlg::CUDA */
#endif

namespace GPU {

/// Global variable to signal the initialization status of the / structures for 
/// GPU handling
//"Defined" in src/CUDA/cuda.cc
extern bool _GPU_structures_initialized;

/** \brief            A wrapper to initialize all GPU related handles
 */
inline void init() {

#ifdef USE_GLOBAL_TRANSFER_STREAMS
  // Initialize the global streams
  LinAlg::CUDA::init();
#endif

  // Initialize all cuBLAS handles
  LinAlg::CUDA::cuBLAS::init();

  // Initialize all cuSPARSE handles
  LinAlg::CUDA::cuSPARSE::init();

#ifdef HAVE_MAGMA
  // Initialize MAGMA
  magma_init();
#endif

  _GPU_structures_initialized = true;

}

} /* namespace LinAlg::GPU */

} /* namespace LinAlg */

#endif /* LINALG_CUDA_CUDA_H_ */
