/** \file
 *
 *  \brief            Set (2D) array to a given value
 *
 *  \date             Created:  Jul 12, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_UTILITIES_SET_ARRAY_H_
#define LINALG_UTILITIES_SET_ARRAY_H_

#include <memory>   // std::shared_ptr

#include "../preprocessor.h"

#ifdef HAVE_CUDA
# include <cuda_runtime.h> // various CUDA routines
# include "../CUDA/cuda_checks.h"  // checkCUDA, checkCUBLAS, checkCUSPARSE
#endif

#ifdef HAVE_MIC
# include "../MIC/mic_helper.h"
#endif

#include "../types.h"
#include "../profiling.h"
#include "../LAPACK/laset.h"

namespace LinAlg {

namespace Utilities {

/** \brief            Set a 2D array to a given value
 *
 *  \param[in]        array
 *                    Pointer to the first element of the array
 *
 *  \param[in]        leading_dimension
 *                    Leading dimension of the array
 *
 *  \param[in]        rows
 *                    Number of rows in the array
 *
 *  \param[in]        cols
 *                    Number of columns in the array
 *
 *  \param[in]        value
 *                    Value to set the array to
 *
 *  \param[in]        format
 *                    The format of the array
 */
template <typename T>
inline void set_2Darray(T* array, Location location, int device_id, 
                        I_t leading_dimension, I_t rows, I_t cols, T value, 
                        Format format) {

  PROFILING_FUNCTION_HEADER

  if (rows == 0 || cols == 0) return;

#ifndef LINALG_NO_CHECKS
  if ((format != Format::ColMajor && format != Format::RowMajor) ||
      (format != Format::ColMajor && format != Format::RowMajor)   ) {

    throw excBadArgument("set_2Darray(): format must be one of "
                         "Format::ColMajor and Format::RowMajor");

  }
#endif
  if (location == Location::host) {
  
    // Use LAPACK
    auto uplo  = "A";
    auto m = (format == Format::ColMajor) ? rows : cols;
    auto n = (format == Format::ColMajor) ? cols : rows;
    
    LAPACK::FORTRAN::xLASET(uplo, m, n, value, value, array, leading_dimension);
  
  }
#ifdef HAVE_CUDA
  else if (location == Location::GPU) {
  
    if (value == cast<T>(0.0)) {

      // Use cudaMemset2D
      // Since the binary representation of a S_t, C_t, D_t, or Z_t 0 is the 
      // same as the concatenation of the binary representation of integer 
      // zeros, this use of cudaMemset2D is valid. See C99 standard, Annex F.
      I_t line_length = (format == Format::ColMajor) ? rows : cols;
      I_t lines       = (format == Format::ColMajor) ? cols : rows;

      LinAlg::Stream my_stream(device_id);

      checkCUDA(cudaMemset2DAsync(array, leading_dimension * sizeof(T), \
                                  0, line_length * sizeof(T), lines, \
                                  my_stream.cuda_stream));

      my_stream.sync();

# ifdef HAVE_MAGMA
    } else {
    
      using LAPACK::MAGMA::xLASET;

      auto m = (format == Format::ColMajor) ? rows : cols;
      auto n = (format == Format::ColMajor) ? cols : rows;
      xLASET(MagmaUpper, m, n, value, value, array, leading_dimension);
      xLASET(MagmaLower, m, n, value, value, array, leading_dimension);
    
    }
# else
    } else {
    
      throw excUnimplemented("set_2Darray(): without Magma enabled, currently "
                             "only zeroing of 2D arrays is supported on the "
                             "GPU");
    
    }
# endif
  
  }
#endif /* HAVE_CUDA */

#ifndef LINALG_NO_CHECKS
  else {

    throw excUnimplemented("copy_2Darray(): unsupported matrix location");

  }
#endif

}

} /* namespace LinAlg::Utilities */

} /* namespace LinAlg */

#endif /* LINALG_UTILITIES_SET_ARRAY_H_ */
