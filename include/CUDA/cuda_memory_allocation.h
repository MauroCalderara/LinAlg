/** \file
 *
 *  \brief            CUDA memory management (allocation and deletion)
 *
 *  \date             Created:  Jul 12, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_CUDA_CUDA_MEMORY_ALLOCATION_H_
#define LINALG_CUDA_CUDA_MEMORY_ALLOCATION_H_

#ifdef HAVE_CUDA

#include <vector>         // std::vector for cublas handles
#include <memory>         // std::shared_ptr
#include <cuda_runtime.h> // various CUDA routines
#include <cublas_v2.h>
#include <cusparse_v2.h>

#include "../preprocessor.h"
#include "../types.h"
#include "../profiling.h"
#include "cuda_checks.h"

namespace LinAlg {

namespace CUDA {

/** \brief            Deallocate memory on the GPGPU
 *
 *  \param[in]        device_ptr
 *                    A pointer to previously allocated memory on the GPGPU.
 *
 *  \param[in]        device_id
 *                    The number of the GPU on which the memory has been
 *                    allocated.
 */
template <typename T>
inline void cuda_deallocate(T* device_ptr, int device_id) {

  PROFILING_FUNCTION_HEADER

  auto prev_device = device_id;
  checkCUDA(cudaGetDevice(&prev_device));

  checkCUDA(cudaSetDevice(device_id));
  checkCUDA(cudaFree((void*) device_ptr));

  checkCUDA(cudaSetDevice(prev_device));

}

/** \brief            Allocate memory on the GPGPU
 *
 *  \param[in]        size
 *                    Length of the memory array to allocate on the GPGPU.
 *
 *  \param[in]        device_id
 *                    The number of the GPU on which to allocate the memory.
 *
 *  \return           A shared_ptr to memory on the GPGPU.
 */
template <typename T>
inline std::shared_ptr<T> cuda_make_shared(I_t size, int device_id) {

  PROFILING_FUNCTION_HEADER

#ifndef LINALG_NO_CHECKS
  if (size < 1) {
    throw excBadArgument("cuda_make_shared(): size must be larger than 0");
  }
#endif

  T* device_ptr;

  auto prev_device = device_id;
  checkCUDA(cudaGetDevice(&prev_device));

  checkCUDA(cudaSetDevice(device_id));
  checkCUDA(cudaMalloc((void **) &device_ptr, size * sizeof(T)));

  checkCUDA(cudaSetDevice(prev_device));

  auto deleter = std::bind(cuda_deallocate<T>, std::placeholders::_1,
                           device_id);

  return std::shared_ptr<T>(device_ptr, deleter);

}

/** \brief            Allocate optimized 2D array on the GPGPU
 *
 *  \param[in]        rows
 *                    Number of rows in the 2D array
 *
 *  \param[in]        cols
 *                    Number of columns in the 2D array
 *
 *  \param[in]        device_id
 *                    The number of the GPU on which to allocate the 2D array
 *
 *  \param[in]        format
 *                    Format of the 2D array
 *
 *  \param[out]       leading_dimension
 *                    The leading dimension of the allocated array
 *
 *  \return           A shared_ptr to the 2D array on the GPGPU
 */
template <typename T>
inline std::shared_ptr<T> cuda_make_shared_2D(const I_t rows, const I_t cols, 
                                              const int device_id, 
                                              const Format format, 
                                              I_t& leading_dimension) {

  PROFILING_FUNCTION_HEADER

#ifndef LINALG_NO_CHECKS
  if (rows == 0 || cols == 0) {
    throw excBadArgument("cuda_make_shared_2D(): rows and columns must be "
                         "larger than 0");
  }
#endif

  T* device_ptr;

  auto width  = ((format == Format::ColMajor) ? cols : rows) * sizeof(T);
  auto height =  (format == Format::ColMajor) ? rows : cols;

  auto prev_device = device_id;
  checkCUDA(cudaGetDevice(&prev_device));

  checkCUDA(cudaSetDevice(device_id));
  checkCUDA(cudaMallocPitch((void**) &device_ptr, \
                            (size_t*) &leading_dimension, \
                            width, (size_t) height));

  checkCUDA(cudaSetDevice(prev_device));

  auto deleter = std::bind(cuda_deallocate<T>, std::placeholders::_1,
                           device_id);

  return std::shared_ptr<T>(device_ptr, deleter);

}

} /* namespace LinAlg::CUDA */

} /* namespace LinAlg */

#endif /* HAVE_CUDA */

#endif /* LINALG_CUDA_CUDA_MEMORY_ALLOCATION_H_ */
