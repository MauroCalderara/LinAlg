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

#ifdef HAVE_MAGMA
#include <magma.h>
#endif

// Keep this in alphabetical order
#include "cuda_checks.h"
#include "cuda_cublas.h"
#include "cuda_memory_allocation.h"

namespace LinAlg {

namespace GPU {

/// Global variable to signal the initializatin status of CUBLAS::handles.
/// "Defined" in src/CUDA/cuda.cc
extern bool _handles_are_initialized;

/** \brief            A wrapper to initialize all GPU related handles
 */
inline void init() {

  // Initialize all CUBLAS handles
  LinAlg::CUDA::CUBLAS::init();

  // Same for CUSPARSE:
  //LinAlg::CUDA::CUSPARSE::init();

#ifdef HAVE_MAGMA
  // Initialize MAGMA
  magma_init();
#endif

  _handles_are_initialized = true;

}


} /* namespace LinAlg::CUDA */

} /* namespace LinAlg */

#endif /* LINALG_CUDA_CUDA_H_ */
