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

# ifndef USE_LOCAL_CUDA_HANDLES

#   include <cuda_runtime.h> // various CUDA routines
#   include <cusparse_v2.h>

#   include "../profiling.h"
#   include "cuda_checks.h"
#   include "cuda_common.h"
#   include "../streams.h"

namespace LinAlg {

namespace CUDA {

namespace cuSPARSE {

/// Global array of cuSPARSE handles (one for each GPU).
// "Defined" in src/CUDA/cuda_cusparse.cc
extern cusparseHandle_t* handles;

/** \brief            A wrapper to initialize the global cuSPARSE handles
 */
inline void init() {

  PROFILING_FUNCTION_HEADER

  using LinAlg::GPU::GPU_structures_initialized;
  using LinAlg::GPU::n_devices;
  using LinAlg::CUDA::cuSPARSE::handles;

  if (GPU_structures_initialized == true) return;

  handles = new cusparseHandle_t[n_devices];

  int prev_device;
  checkCUDA(cudaGetDevice(&prev_device));

  for (int device_id = 0; device_id < n_devices; ++device_id) {

    checkCUDA(cudaSetDevice(device_id));
    checkCUSPARSE(cusparseCreate(&(handles[device_id])));

  }

  checkCUDA(cudaSetDevice(prev_device));

}


/** \brief            Destroy the global cuSPARSE handles
 */
inline void destroy() {

  PROFILING_FUNCTION_HEADER

  using LinAlg::GPU::GPU_structures_initialized;
  using LinAlg::GPU::n_devices;
  using LinAlg::CUDA::cuSPARSE::handles;

  if (GPU_structures_initialized == false) return;

  for (int device_id = 0; device_id < n_devices; ++device_id) {

    checkCUSPARSE(cusparseDestroy(handles[device_id]));

  }

  delete[] handles;

}

/** \brief            Prepare a call to cuSPARSE
 *
 *  \param[in]        stream
 *                    Stream to use for the cuSPARSE call(s). Note that the 
 *                    device to which the stream is bound, will used for the 
 *                    subsequent call to cuSPARSE.
 *
 *  \param[in,out]    previous_device
 *                    Integer storing the device that was active before the 
 *                    cuSPARSE call(s)
 *
 *  \param[in,out]    previous_stream
 *                    Stream storing the stream that was active before the 
 *                    cuSPARSE call(s)
 *
 *  \returns          Pointer to a handle for the cuSPARSE call(s)
 *
 *  \note             This function calls cudaSetDevice and cusparseSetStream 
 *                    without resetting it, so call the cuSPARSE function 
 *                    immediately after it
 *
 *  \example
 *
 *    using LinAlg::CUDA::cuSPARSE::prepare_cusparse;
 *    using LinAlg::CUDA::cuSPARSE::finish_cusparse;
 *
 *    int          prev_device = 0;
 *    cudaStream_t prev_cuda_stream;
 *
 *    auto handle = prepare_cusparse(some_stream, &prev_device, 
 *                                   &prev_cuda_stream);
 *    BLAS::cuSPARSE::xFOO(*handle, some, arguments);
 *    BLAS::cuSPARSE::xBAR(*handle, other, arguments);
 *    finish_cusparse(some_stream, &prev_device, &prev_cuda_stream, handle);
 *
 *
 */
inline cusparseHandle_t* prepare_cusparse(Stream& stream, int* previous_device,
                                          cudaStream_t* previous_stream) {

  PROFILING_FUNCTION_HEADER

  checkCUDA(cudaGetDevice(previous_device));

  checkCUDA(cudaSetDevice(stream.device_id));

  cusparseHandle_t* handle;

# ifndef USE_LOCAL_CUDA_HANDLES

  using LinAlg::CUDA::cuSPARSE::handles;

  // Isn't available:
  /* checkCUSPARSE(cusparseGetStream(handles[stream.device_id], \
                                     previous_stream)); */

  checkCUSPARSE(cusparseSetStream(handles[stream.device_id], \
                                  stream.cuda_stream));

  handle = &(handles[stream.device_id]);

# else

  handle = new cusparseHandle_t[1];
  
  checkCUSPARSE(cusparseCreate(handle));

  // Here we don't care about storing the  previous_stream of the handle, 
  // obviously
  checkCUSPARSE(cusparseSetStream(*handle, stream.cuda_stream));

# endif

  return handle;

}
          
/** \brief            Finish a call to cuSPARSE
 *
 *  \param[in]        stream
 *                    Stream used for the preceeding cuSPARSE call(s).
 *
 *  \param[in]        previous_device
 *                    Integer storing the device that was active before the 
 *                    call
 *
 *  \param[in]        previous_stream
 *                    Previous stream
 *
 *  \param[in,out]    handle
 *                    Handle used for the preceeding cuSPARSE call(s).
 *
 *  \note             This call will reset the CUDA device to the one that was 
 *                    active previous to the call to prepare_cusparse() above.  
 *                    Call immediately after the last cuSPARSE call using this 
 *                    handle. Don't use the handle after this call.
 *
 *  \example          See prepare_cusparse();
 */
inline void finish_cusparse(Stream& stream, int* previous_device,
                            cudaStream_t* previous_stream,
                            cusparseHandle_t* handle) {

  PROFILING_FUNCTION_HEADER

  checkCUDA(cudaSetDevice(*previous_device));

  stream.cuda_synchronized = false;

# ifndef USE_LOCAL_CUDA_HANDLES

  checkCUSPARSE(cusparseSetStream(handles[stream.device_id], *previous_stream));

# else

  checkCUSPARSE(cusparseDestroy(*handle));

  delete[] handle;

# endif

}


} /* namespace LinAlg::CUDA::cuSPARSE */

} /* namespace LinAlg::CUDA */

} /* namespace LinAlg */

# endif /* not USE_LOCAL_CUDA_HANDLES */

#endif /* HAVE_CUDA */

#endif /* LINALG_CUDA_CUDA_CUSPARSE_H_ */
