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

#include "../preprocessor.h"

#ifdef HAVE_MAGMA
# include <magma.h>
#endif

#include "cuda_common.h"
#include "cuda_checks.h"
#include "cuda_cublas.h"
#include "cuda_cusparse.h"
#include "cuda_memory_allocation.h"

namespace LinAlg {

namespace GPU {

/** \brief            A wrapper to initialize GPU related structures
 */
inline void init() {

  if (GPU_structures_initialized == true) return;

  checkCUDA(cudaGetDeviceCount(&n_devices));

#ifndef USE_LOCAL_STREAMS
  // Initialize the global streams
  LinAlg::CUDA::init();
#endif

#ifndef USE_LOCAL_CUDA_HANDLES
  // Initialize all cuBLAS handles
  LinAlg::CUDA::cuBLAS::init();

  // Initialize all cuSPARSE handles
  LinAlg::CUDA::cuSPARSE::init();
#endif

#ifdef HAVE_MAGMA
  // Initialize MAGMA
  magma_init();
#endif

  GPU_structures_initialized = true;

}

/** \brief            A wrapper to destroy the GPU related structures
 */
inline void destroy() {

  PROFILING_FUNCTION_HEADER

  if (GPU_structures_initialized == false) return;

#ifndef USE_LOCAL_STREAMS
  // Initialize the global streams
  LinAlg::CUDA::destroy();
#endif

#ifndef USE_LOCAL_CUDA_HANDLES
  // Initialize all cuBLAS handles
  LinAlg::CUDA::cuBLAS::destroy();

  // Initialize all cuSPARSE handles
  LinAlg::CUDA::cuSPARSE::destroy();
#endif

#ifdef HAVE_MAGMA
  // Initialize MAGMA
  magma_init();
#endif

  n_devices = 0;
  GPU_structures_initialized = false;

}

} /* namespace LinAlg::GPU */

} /* namespace LinAlg */

#endif /* LINALG_CUDA_CUDA_H_ */
