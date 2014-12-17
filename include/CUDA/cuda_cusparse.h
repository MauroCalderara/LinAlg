/** \file
 *
 *  \brief            cuSPARSE handles
 *
 *  \date             Created:  Jul 12, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_CUDA_CUDA_CUSPARSE_H_
#define LINALG_CUDA_CUDA_CUSPARSE_H_

#include "../preprocessor.h"

#ifdef HAVE_CUDA

# include <vector>         // std::vector for cuSPARSE handles
# include <cuda_runtime.h> // various CUDA routines
# include <cusparse_v2.h>

# include "../types.h"
# include "../profiling.h"
# include "cuda_checks.h"

namespace LinAlg {

namespace CUDA {

namespace cuSPARSE {

/// Global vector of cuSPARSE handles (one for each GPU).
// "Defined" in src/CUDA/cuda_cusparse.cc
extern std::vector<cusparseHandle_t> handles;

/** \brief            A wrapper to initialize the global cuSPARSE handles
 */
inline void init() {

  PROFILING_FUNCTION_HEADER

  int device_count;
  checkCUDA(cudaGetDeviceCount(&device_count));

  handles.resize(device_count);

  int prev_device;
  checkCUDA(cudaGetDevice(&prev_device));

  for (int device_id = 0; device_id < device_count; ++device_id) {

    checkCUDA(cudaSetDevice(device_id));
    checkCUSPARSE(cusparseCreate(&(handles[device_id])));

  }

  checkCUDA(cudaSetDevice(prev_device));

}


/** \brief            Destroy the global cuSPARSE handles
 */
inline void destroy() {

  PROFILING_FUNCTION_HEADER

  for (unsigned int device = 0; device < handles.size(); ++device) {

    checkCUSPARSE(cusparseDestroy(handles[device]));

  }

  handles.clear();

}

} /* namespace LinAlg::CUDA::cuSPARSE */

} /* namespace LinAlg::CUDA */

} /* namespace LinAlg */

# endif /* HAVE_CUDA */

#endif /* LINALG_CUDA_CUDA_CUSPARSE_H_ */
