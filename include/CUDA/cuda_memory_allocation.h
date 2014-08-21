/** \file             cuda_memory_allocation.h
 *
 *  \brief            Routines to check for CUDA/CUBLAS/CUSPARSE error messages
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

#include "types.h"
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

  auto prev_device = device_id;
  checkCUDA(cudaGetDevice(&prev_device));

  checkCUDA(cudaSetDevice(device_id));
  checkCUDA(cudaFree((void*) device_ptr));

  checkCUDA(cudaSetDevice(prev_device));

};

/** \brief            Allocate memory on the GPGPU
 *
 *  \param[in]        size
 *                    Length of the memory array to allocate on the GPGPU.
 *
 *  \param[in]        device_id
 *                    The number of the GPU on which to allocate the memory.
 *
 *  \return           device_ptr
 *                    A pointer to memory on the GPGPU.
 */
template <typename T>
inline std::shared_ptr<T> cuda_make_shared(I_t size, int device_id) {

  T* device_ptr;

  auto prev_device = device_id;
  checkCUDA(cudaGetDevice(&prev_device));

  checkCUDA(cudaSetDevice(device_id));
  checkCUDA(cudaMalloc((void **) &device_ptr, size * sizeof(T)));

  checkCUDA(cudaSetDevice(prev_device));

  auto deleter = std::bind(cuda_deallocate<T>, std::placeholders::_1,
                           device_id);

  return std::shared_ptr<T>(device_ptr, deleter);

};

} /* namespace LinAlg::CUDA */

} /* namespace LinAlg */

#endif /* HAVE_CUDA */

#endif /* LINALG_CUDA_CUDA_MEMORY_ALLOCATION_H_ */
