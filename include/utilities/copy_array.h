/** \file
 *
 *  \brief            General 1D and 2D array copy
 *
 *  \date             Created:  Jul 12, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_UTILITIES_COPY_ARRAY_H_
#define LINALG_UTILITIES_COPY_ARRAY_H_

#include "../preprocessor.h"

#ifdef HAVE_CUDA
# include <cuda_runtime.h> // various CUDA routines
# include "../CUDA/cuda_checks.h"  // checkCUDA, checkCUBLAS, checkCUSPARSE
#endif

#include "../types.h"
#include "../profiling.h"
#include "../exceptions.h"
#include "../streams.h"
#include "../BLAS/blas.h"

namespace LinAlg {

namespace Utilities {

#ifndef DOXYGEN_SKIP

/* \brief              Helper function specialized with to use xCOPY for
 *                      supported datatypes
 */
template <typename T>
inline void copy_1Darray_host(T* src_array, I_t length, T* dst_array) {

  PROFILING_FUNCTION_HEADER

  for (I_t i = 0; i < length; ++i) dst_array[i] = src_array[i];

}
template <>
inline void copy_1Darray_host(S_t* src_array, I_t length, S_t* dst_array){

  PROFILING_FUNCTION_HEADER

  LinAlg::BLAS::FORTRAN::xCOPY(length, src_array, 1, dst_array, 1);

}
template <>
inline void copy_1Darray_host(D_t* src_array, I_t length, D_t* dst_array){

  PROFILING_FUNCTION_HEADER

  LinAlg::BLAS::FORTRAN::xCOPY(length, src_array, 1, dst_array, 1);

}
template <>
inline void copy_1Darray_host(C_t* src_array, I_t length, C_t* dst_array){

  PROFILING_FUNCTION_HEADER

  LinAlg::BLAS::FORTRAN::xCOPY(length, src_array, 1, dst_array, 1);

}
template <>
inline void copy_1Darray_host(Z_t* src_array, I_t length, Z_t* dst_array){

  PROFILING_FUNCTION_HEADER

  LinAlg::BLAS::FORTRAN::xCOPY(length, src_array, 1, dst_array, 1);

}

#endif /* DOXYGEN_SKIP */


/** \brief              1D array copy
 *
 *  This function copies vectors from raw pointers. This is basically a
 *  wrapper around xCOPY and cudaMemcopy.
 *
 *  \param[in]          src_array
 *                      Source array for the 1D copy.
 *
 *  \param[in]          length
 *                      The length of the 1D array.
 *
 *  \param[in,out]      dst_array
 *                      Destination array for the 1D copy.
 *
 *  \param[in]          src_location
 *                      Location of the source array.
 *
 *  \param[in]          src_device_id
 *                      Device id for the source array.
 *
 *  \param[in]          dst_location
 *                      Location of the array.
 *
 *  \param[in]          dst_device_id
 *                      Id of the device containing
 *
 *  \todo               Investigate whether a specialization using xCOPY would
 *                      be faster
 */
template <typename T>
void copy_1Darray(T* src_array, I_t length, T* dst_array,
                  Location src_location, int src_device_id,
                  Location dst_location, int dst_device_id) {

  PROFILING_FUNCTION_HEADER

  if (src_location == Location::host && dst_location == Location::host) {

    copy_1Darray_host(src_array, length, dst_array);

  }

#ifdef HAVE_CUDA
  else if (src_location == Location::GPU || dst_location == Location::GPU) {

    if (src_location == Location::GPU && dst_location == Location::GPU) {

      if (src_device_id == dst_device_id) {

        int prev_device;

        checkCUDA(cudaGetDevice(&prev_device));
        checkCUDA(cudaSetDevice(dst_device_id));

        checkCUDA(cudaMemcpy(dst_array, src_array, length * sizeof(T),
                             cudaMemcpyDeviceToDevice));

        checkCUDA(cudaSetDevice(prev_device));

      }
# ifndef LINALG_NO_CHECKS
      else {

        throw excUnimplemented("copy_1Darray(): Copy between GPUs not yet "
                               "implemented.");

      }
# endif

    } else if (src_location == Location::GPU && dst_location == Location::host){

      int prev_device;

      checkCUDA(cudaGetDevice(&prev_device));
      checkCUDA(cudaSetDevice(src_device_id));

      checkCUDA(cudaMemcpy(dst_array, src_array, length * sizeof(T),
                           cudaMemcpyDeviceToHost));

      checkCUDA(cudaSetDevice(prev_device));

    } else if (src_location == Location::host && dst_location == Location::GPU){

      int prev_device;

      checkCUDA(cudaGetDevice(&prev_device));
      checkCUDA(cudaSetDevice(dst_device_id));

      checkCUDA(cudaMemcpy(dst_array, src_array, length * sizeof(T),
                           cudaMemcpyHostToDevice));

      checkCUDA(cudaSetDevice(prev_device));

    }
# ifndef LINALG_NO_CHECKS
    else {

      throw excUnimplemented("copy_1Darray(): Copy from GPU to other "
                             "accelerators not implemented. Transfer to host "
                             "first.");

    }
# endif

  }
#endif /* HAVE_CUDA */

#ifndef LINALG_NO_CHECKS
  else {

    throw excUnimplemented("copy_1Darray(): Requested copy not implemented.");

  }
#endif

}


/** \brief              2D array copy
 *
 *  This function copies matrices from raw pointers.
 *
 *  \param[in]          transpose
 *                      Whether to transpose the source matrix while copying.
 *
 *  \param[in]          src_format
 *                      Format of the input array
 *
 *  \param[in]          src_array
 *                      Source array for the 2D copy
 *
 *  \param[in]          src_ld
 *                      Leading dimension of input array (distance in memory of
 *                      the first elements of two consecutive columns for
 *                      ColMajor arrays or first elements of two consecutive
 *                      rows for RowMajor).
 *
 *  \param[in]          src_location
 *                      Location of the input array.
 *
 *  \param[in]          src_device_id
 *                      Device id of the input array.
 *
 *  \param[in]          rows
 *                      The number of rows in the source 2D array.
 *
 *  \param[in]          cols
 *                      The number of columns in the source 2D array.
 *
 *  \param[in]          dst_format
 *                      The format of the output array.
 *
 *  \param[in,out]      dst_array
 *                      Destination array for the 2D copy.
 *
 *  \param[in]          dst_ld
 *                      Leading dimension of the output array (see src_ld for
 *                      further details)
 *
 *  \param[in]          dst_location
 *                      Location of the output array.
 *
 *  \param[in]          dst_device_id
 *                      Device id of the output array.
 *
 *  \note               While calls to copy_2Darray are synchronous, each call
 *                      involving cudaMemcpy or xGEAM creates its own stream 
 *                      such that if there are multiple threads calling
 *                      functions on the GPU there is some parallelism (i.e.
 *                      another thread calling a cuBLAS routine will execute in
 *                      parallel, while other transfers in the same direction
 *                      are scheduled by the CUDA runtime). 
 */
template <typename T>
void copy_2Darray(bool transpose, Format src_format, const T* src_array,
                  I_t src_ld, Location src_location, int src_device_id,
                  I_t rows, I_t cols, Format dst_format, T* dst_array,
                  I_t dst_ld, Location dst_location, int dst_device_id) {

  PROFILING_FUNCTION_HEADER

  if (rows == 0 || cols == 0) return;

#ifndef LINALG_NO_CHECKS
  if ((src_format != Format::ColMajor && src_format != Format::RowMajor) ||
      (src_format != Format::ColMajor && src_format != Format::RowMajor)   ) {

    throw excBadArgument("copy_2Darray(): both src_format and dst_format "
                         "must be one of Format::ColMajor and "
                         "Format::RowMajor");

  }
#endif

  // Copy in main memory: here we support all variants
#ifdef HAVE_MKL
  using LinAlg::BLAS::MKL::xomatcopy;
#endif
  if (src_location == Location::host && dst_location == Location::host) {

    if (src_format == Format::ColMajor && dst_format == Format::ColMajor) {

      if (transpose) {
#ifndef HAVE_MKL
        for (I_t col = 0; col < cols; ++col) {
          for (I_t row = 0; row < rows; ++row) {
            dst_array[col * dst_ld + row] = src_array[row * src_ld + col];
          }
        }
#else
        xomatcopy('C', 'T', rows, cols, 1.0, src_array, src_ld, dst_array,
                  dst_ld);
#endif
      } else {
#ifndef HAVE_MKL
        for (I_t col = 0; col < cols; ++col) {
          for (I_t row = 0; row < rows; ++row) {
            dst_array[col * dst_ld + row] = src_array[col * src_ld + row];
          }
        }
#else
        xomatcopy('C', 'N', rows, cols, 1.0, src_array, src_ld, dst_array,
                   dst_ld);
#endif
      }

    } else if (src_format == Format::RowMajor && 
               dst_format == Format::ColMajor) {

      if (transpose) {
        for (I_t row = 0; row < rows; ++row) {
          for (I_t col = 0; col < cols; ++col) {
            dst_array[col * dst_ld + row] = src_array[col * src_ld + row];
          }
        }
      } else {
        for (I_t row = 0; row < rows; ++row) {
          for (I_t col = 0; col < cols; ++col) {
            dst_array[col * dst_ld + row] = src_array[row * src_ld + col];
          }
        }
      }

    } else if (src_format == Format::ColMajor && 
               dst_format == Format::RowMajor) {

      if (transpose) {
        for (I_t col = 0; col < cols; ++col) {
          for (I_t row = 0; row < rows; ++row) {
            dst_array[row * dst_ld + col] = src_array[row * src_ld + col];
          }
        }
      } else {
        for (I_t col = 0; col < cols; ++col) {
          for (I_t row = 0; row < rows; ++row) {
            dst_array[row * dst_ld + col] = src_array[col * src_ld + row];
          }
        }
      }

    } else if (src_format == Format::RowMajor && 
               dst_format == Format::RowMajor) {

      if (transpose) {
#ifndef HAVE_MKL
        for (I_t row = 0; row < rows; ++row) {
          for (I_t col = 0; col < cols; ++col) {
            dst_array[row * dst_ld + col] = src_array[col * src_ld + row];
          }
        }
#else
        xomatcopy('R', 'T', rows, cols, 1.0, src_array, src_ld, dst_array,
                  dst_ld);
#endif
      } else {
#ifndef HAVE_MKL
        for (I_t row = 0; row < rows; ++row) {
          for (I_t col = 0; col < cols; ++col) {
            dst_array[row * dst_ld + col] = src_array[row * src_ld + col];
          }
        }
#else
        xomatcopy('R', 'N', rows, cols, 1.0, src_array, src_ld, dst_array,
                  dst_ld);
#endif
      }
    }
  }

#ifdef HAVE_CUDA
  else if (src_location == Location::GPU && dst_location == Location::host) {

    // Transfer out of GPU

    CUDAStream* my_stream;

# ifndef USE_GLOBAL_TRANSFER_STREAMS
    my_stream = new CUDAStream(src_device_id);
# else
#   ifndef LINALG_NO_CHECKS
    check_gpu_handles("copy_2Darray()");
#   endif
    //my_stream = &(CUDA::out_stream[src_device_id]);
    my_stream = &CUDA::out_stream;
# endif

    // Supported: straight transfer (no transposition or format change)
    if (transpose == false && src_format == dst_format) {

      auto line_length = (src_format == Format::ColMajor) ? rows : cols;
      auto lines       = (src_format == Format::ColMajor) ? cols : rows;

      checkCUDA(cudaMemcpy2DAsync(dst_array, dst_ld * sizeof(T), \
                                  src_array, src_ld * sizeof(T), \
                                  line_length * sizeof(T), lines, \
                                  cudaMemcpyDeviceToHost, 
                                  my_stream->cuda_stream));

    }
# ifndef LINALG_NO_CHECKS
    // Unsupported: everything else
    else {
    
      throw excUnimplemented("copy_2Darray(): transfers out of the GPU "
                             "currently don't yet support transposition and "
                             "different source and destination formats");
      // With a temporary, everything is possible though ...
    
    }
# endif

    my_stream->sync();

# ifndef USE_GLOBAL_TRANSFER_STREAMS
    delete my_stream;
#endif

  }

  else if (src_location == Location::host && dst_location == Location::GPU) {
  
    // Transfer out

    CUDAStream* my_stream;

# ifndef USE_GLOBAL_TRANSFER_STREAMS
    my_stream = new CUDAStream(dst_device_id);
# else
#   ifndef LINALG_NO_CHECKS
    check_gpu_handles("copy_2Darray()");
#   endif
    //my_stream = &(CUDA::in_stream[dst_device_id]);
    my_stream = &CUDA::in_stream;
# endif

    // Supported: straight transfer (no transposition or format change)
    if (transpose == false && src_format == dst_format) {

      auto line_length = (src_format == Format::ColMajor) ? rows : cols;
      auto lines       = (src_format == Format::ColMajor) ? cols : rows;

      checkCUDA(cudaMemcpy2DAsync(dst_array, dst_ld * sizeof(T), \
                                  src_array, src_ld * sizeof(T), \
                                  line_length * sizeof(T), lines, \
                                  cudaMemcpyHostToDevice, \
                                  my_stream->cuda_stream));

    }
# ifndef LINALG_NO_CHECKS
    // Unsupported: everything else
    else {
    
      throw excUnimplemented("copy_2Darray(): transfers into the GPU "
                             "currently don't yet support transposition and "
                             "different source and destination formats");
      // With a temporary, everything is possible though ...
    
    }
# endif

    my_stream->sync();

# ifndef USE_GLOBAL_TRANSFER_STREAMS
    delete my_stream;
# endif
  
  }

  else if (src_location == Location::GPU && dst_location == Location::GPU) {

    using LinAlg::BLAS::CUBLAS::xGEAM;
  
    // Transfer within GPU

    CUDAStream* my_stream;

# ifndef USE_GLOBAL_TRANSFER_STREAMS
    my_stream = new CUDAStream(src_device_id);
# else
#   ifndef LINALG_NO_CHECKS
    check_gpu_handles("copy_2Darray()");
#   endif
    //my_stream = &(CUDA::on_stream[dst_device_id]);
    my_stream = &CUDA::on_stream;
# endif

    // Supported: straight transfer (no transposition or format change)
    if (transpose == false && src_format == dst_format) {

      auto line_length = (src_format == Format::ColMajor) ? rows : cols;
      auto lines       = (src_format == Format::ColMajor) ? cols : rows;

      checkCUDA(cudaMemcpy2DAsync(dst_array, dst_ld * sizeof(T), \
                                  src_array, src_ld * sizeof(T), \
                                  line_length * sizeof(T), lines, \
                                  cudaMemcpyDeviceToDevice, 
                                  my_stream->cuda_stream));

    }

    // Supported: transposition if both matrices are ColMajor
    else if (transpose == true && src_format == Format::ColMajor && 
                             dst_format == Format::ColMajor   ) {
    
      auto line_length = rows;
      auto lines       = cols;

      // Since xGEAM doesn't support B = nullptr, we use B = A and beta = 0.0;
      xGEAM(my_stream->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, line_length,
            lines, cast<T>(1.0), src_array, src_ld, cast<T>(0.0), src_array,
            src_ld, dst_array, dst_ld);
    
    }

    // Supported: change of format if source is ColMajor and destination is not 
    // a submatrix
    else if (transpose == false && src_format == Format::ColMajor &&
             dst_ld == cols) {

      // GEAM assumes both arrays are in ColMajor and would thus mess up
      // with a non-consecutive RowMajor array. However, since our destination 
      // array is consecutive, we transpose the ColMajor array into what xGEAM 
      // believes to be ColMajor array with tight leading dimension. This 
      // effectively makes the destination a RowMajor array with the same 
      // content as the source.

      auto line_length = rows;
      auto lines       = cols;

      // Since xGEAM doesn't support B = nullptr, we use B = A and beta = 0.0;
      xGEAM(my_stream->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, line_length,
            lines, cast<T>(1.0), src_array, src_ld, cast<T>(0.0), src_array, 
            src_ld, dst_array, dst_ld);
    
    }
    // Supported: change of format if source is RowMajor and not a submatrix  
    // and destination is ColMajor
    else if (transpose == false && src_format == Format::RowMajor &&
             src_ld == cols) {
    
      // GEAM assumes both arrays are in ColMajor and would thus mess up
      // with a non-consecutive RowMajor array. However, since our source array 
      // is consecutive we transpose the what xGEAM believes to be a ColMajor 
      // array with tight leading dimension into a ColMajor array with  
      // arbitrary properties. This effectively makes the destination a  
      // ColMajor array with the same content as the source RowMajor array.

      auto line_length = cols;
      auto lines       = rows;

      // Since xGEAM doesn't support B = nullptr, we use B = A and beta = 0.0;
      xGEAM(my_stream->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, line_length,
            lines, cast<T>(1.0), src_array, src_ld, cast<T>(0.0), src_array, 
            src_ld, dst_array, dst_ld);
    
    }

# ifndef LINALG_NO_CHECKS
    // Unsupported: everything else
    else {
    
      throw excUnimplemented("copy_2Darray(): the requested combination "
                             "of arguments has not yet been implemented for "
                             "transfers within the GPU");
      // With a temporary, everything is possible though ...

    }
# endif

    my_stream->sync();

# ifndef USE_GLOBAL_TRANSFER_STREAMS
    delete my_stream;
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

#endif /* LINALG_UTILITIES_COPY_ARRAY_H_ */
