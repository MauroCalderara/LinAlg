/** \file
 *
 *  \brief            cuBLAS handles
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

#include "../preprocessor.h"

#ifdef HAVE_CUDA

#   include <cuda_runtime.h> // various CUDA routines
#   include <cublas_v2.h>

#   include "../profiling.h"
#   include "cuda_checks.h"
#   include "cuda_common.h"
#   include "../streams.h"

namespace LinAlg {

namespace CUDA {

namespace cuBLAS {

# ifndef USE_LOCAL_CUDA_HANDLES

/// Global array of cuBLAS handles (one for each GPU).
// "Defined" in src/CUDA/cuda_cublas.cc
extern cublasHandle_t* handles;

/** \brief            A wrapper to initialize the global cuBLAS handles
 */
inline void init() {

  PROFILING_FUNCTION_HEADER

  using LinAlg::GPU::GPU_structures_initialized;
  using LinAlg::GPU::n_devices;
  using LinAlg::CUDA::cuBLAS::handles;

  if (GPU_structures_initialized == true) return;

  handles = new cublasHandle_t[n_devices];

  int prev_device;
  checkCUDA(cudaGetDevice(&prev_device));

  for (int device_id = 0; device_id < n_devices; ++device_id) {

    checkCUDA(cudaSetDevice(device_id));
    checkCUBLAS(cublasCreate(&(handles[device_id])));

  }

  checkCUDA(cudaSetDevice(prev_device));

}

/** \brief            Destroy the global cuBLAS handles
 */
inline void destroy() {

  PROFILING_FUNCTION_HEADER

  using LinAlg::GPU::GPU_structures_initialized;
  using LinAlg::GPU::n_devices;
  using LinAlg::CUDA::cuBLAS::handles;

  if (GPU_structures_initialized == false) return;

  for (int device = 0; device < n_devices; ++device) {

    checkCUBLAS(cublasDestroy(handles[device]));

  }

  delete[] handles;

}

# endif /* not USE_LOCAL_CUDA_HANDLES */

/** \brief            Prepare a call to cuBLAS
 *
 *  \param[in]        stream
 *                    Stream to use for the cuBLAS call(s). Note that the 
 *                    device to which the stream is bound, will used for the 
 *                    subsequent call to cuBLAS.
 *
 *  \param[in,out]    previous_device
 *                    Integer storing the device that was active before the 
 *                    cuBLAS call(s)
 *
 *  \param[in,out]    previous_stream
 *                    Stream storing the stream that was active before the 
 *                    cuBLAS call(s)
 *
 *  \returns          Pointer to a handle for the cuBLAS call(s)
 *
 *  \note             This function calls cudaSetDevice and cublasSetStream 
 *                    without resetting it, so call the cuBLAS function 
 *                    immediately after it
 *
 *  \example
 *
 *    using LinAlg::CUDA::cuBLAS::prepare_cublas;
 *    using LinAlg::CUDA::cuBLAS::finish_cublas;
 *
 *    int          prev_device = 0;
 *    cudaStream_t prev_cuda_stream;
 *
 *    auto handle = prepare_cublas(some_stream, &prev_device, 
 *                                 &prev_cuda_stream);
 *    BLAS::cuBLAS::xFOO(*handle, some, arguments);
 *    BLAS::cuBLAS::xBAR(*handle, other, arguments);
 *    finish_cublas(some_stream, &prev_device, &prev_cuda_stream, handle);
 *
 *
 */
inline cublasHandle_t* prepare_cublas(Stream& stream, int* previous_device,
                                      cudaStream_t* previous_stream) {

  PROFILING_FUNCTION_HEADER

  checkCUDA(cudaGetDevice(previous_device));

  checkCUDA(cudaSetDevice(stream.device_id));

  cublasHandle_t* handle;

# ifndef USE_LOCAL_CUDA_HANDLES

  using LinAlg::CUDA::cuBLAS::handles;

  checkCUBLAS(cublasGetStream(handles[stream.device_id], previous_stream));

  checkCUBLAS(cublasSetStream(handles[stream.device_id], stream.cuda_stream));

  handle = &(handles[stream.device_id]);

# else

  handle = new cublasHandle_t[1];
  
  checkCUBLAS(cublasCreate(handle));

  // Here we don't care about storing the  previous_stream of the handle, 
  // obviously
  checkCUBLAS(cublasSetStream(*handle, stream.cuda_stream));

# endif

  return handle;

}
          
/** \brief            Finish a call to cuBLAS
 *
 *  \param[in]        stream
 *                    Stream used for the preceeding cuBLAS call(s).
 *
 *  \param[in]        previous_device
 *                    Integer storing the device that was active before the 
 *                    call
 *
 *  \param[in]        previous_stream
 *                    Previous stream
 *
 *  \param[in,out]    handle
 *                    Handle used for the preceeding cuBLAS call(s).
 *
 *  \note             This call will reset the CUDA device to the one that was 
 *                    active previous to the call to prepare() above. Call 
 *                    immediately after the last cuBLAS call using this 
 *                    handle. Don't use the handle after this call.
 *
 *  \example          See prepare();
 */
inline void finish_cublas(Stream& stream, int* previous_device,
                          cudaStream_t* previous_stream,
                          cublasHandle_t* handle) {

  PROFILING_FUNCTION_HEADER

  checkCUDA(cudaSetDevice(*previous_device));

  stream.cuda_synchronized = false;

# ifndef USE_LOCAL_CUDA_HANDLES

  checkCUBLAS(cublasSetStream(handles[stream.device_id], *previous_stream));

# else

  checkCUBLAS(cublasDestroy(*handle));

  delete[] handle;

# endif

}

} /* namespace LinAlg::CUDA::cuBLAS */

} /* namespace LinAlg::CUDA */

} /* namespace LinAlg */

#endif /* HAVE_CUDA */

#endif /* LINALG_CUDA_CUDA_CUBLAS_H_ */
