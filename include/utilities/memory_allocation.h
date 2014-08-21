/** \file
 *
 *  \brief            Memory allocation using smart pointers
 *
 *  \date             Created:  Jul 12, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_UTILITIES_MEMORY_ALLOCATION_H_
#define LINALG_UTILITIES_MEMORY_ALLOCATION_H_

#include <memory>   // std::shared_ptr

#ifdef HAVE_CUDA
#include <cuda_runtime.h> // various CUDA routines
#include "CUDA/cuda_checks.h"  // checkCUDA, checkCUBLAS, checkCUSPARSE
#endif

#ifdef HAVE_MIC
#include "mic_helper.h"
#endif

#include "types.h"

namespace LinAlg {

namespace Utilities {

/** \brief            Routine to allocate an array for shared ownership
 */
template <typename T>
inline std::shared_ptr<T> host_make_shared(I_t size) {

#ifdef HAVE_CUDA
  T* host_ptr;
  checkCUDA(cudaMallocHost(&host_ptr, size * sizeof(T)));
  return std::shared_ptr<T>(host_ptr, [](T* p){ checkCUDA(cudaFreeHost(p)); });
#else
  T* host_ptr = new T[size];
  return std::shared_ptr<T>(host_ptr, [](T* p){ delete[] p; });
#endif

};
} /* namespace LinAlg::Utilities */

} /* namespace LinAlg */


#endif /* LINALG_UTILITIES_MEMORY_ALLOCATION_H_ */
