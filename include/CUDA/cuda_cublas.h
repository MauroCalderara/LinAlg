/** \file
 *
 *  \brief            CUBLAS handles
 *
 *  \date             Created:  Jul 12, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_CUDA_CUDA_CUBLAS_H_
#define LINALG_CUDA_CUDA_CUBLAS_H_

#ifdef HAVE_CUDA

#include <vector>         // std::vector for cublas handles
#include <cuda_runtime.h> // various CUDA routines
#include <cublas_v2.h>

#include "../types.h"
#include "cuda_checks.h"

namespace LinAlg {

namespace CUDA {

namespace CUBLAS {

/// Global vector of cublas handles (one for each GPU). "Defined" in
/// src/CUDA/cuda_cublas.cc
extern std::vector<cublasHandle_t> handles;

/// Global variable to signal the initializatin status of CUBLAS::handles
extern bool _handles_are_initialized;

/** \brief            A wrapper to initialize the global CUBLAS handles
 */
inline void init() {

  int device_count;
  checkCUDA(cudaGetDeviceCount(&device_count));

  handles.resize(device_count);

  int prev_device;
  checkCUDA(cudaGetDevice(&prev_device));

  for (int device_id = 0; device_id < device_count; ++device_id) {

    checkCUDA(cudaSetDevice(device_id));
    checkCUBLAS(cublasCreate(&(handles[device_id])));

  }

  checkCUDA(cudaSetDevice(prev_device));

};

/** \brief            Check if the handles are initialized.
 *
 *  \returns          true if they are initialized, false otherwise.
 */
inline bool is_initialized() {

  return ((handles.empty()) ? false : true);

};

/** \brief            Destroy the global CUBLAS handles
 */
inline void destroy() {

  for (unsigned int device = 0; device < handles.size(); ++device) {

    checkCUBLAS(cublasDestroy(handles[device]));

  }

  handles.clear();

};

} /* namespace LinAlg::CUDA::CUBLAS */

} /* namespace LinAlg::CUDA */

} /* namespace LinAlg */

#endif /* HAVE_CUDA */

#endif /* LINALG_CUDA_CUDA_CUBLAS_H_ */
