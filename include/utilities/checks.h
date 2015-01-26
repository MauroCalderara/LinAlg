/** \file
 *
 *  \brief            Argument checking
 *
 *  \date             Created:  Jul 16, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_UTILITIES_CHECKS_H_
#define LINALG_UTILITIES_CHECKS_H_

#include "../preprocessor.h"

#include "../types.h"
#include "../exceptions.h"
#include "../streams.h"
#include "../dense.h"
#include "../sparse.h"
#include "stringformat.h"

#ifdef HAVE_CUDA
# include "../CUDA/cuda.h"    // GPU::
#endif

namespace LinAlg {

namespace Utilities {

#ifndef DOXYGEN_SKIP

# ifndef LINALG_NO_CHECKS
/** \brief            Checks if two matrices are on the same device, throws an
 *                    exception if not.
 *
 *  \param[in]        A
 *                    First matrix
 *
 *  \param[in]        B
 *                    Second matrix
 *
 *  \param[in]        caller_name
 *                    Name of the calling routine
 */
template <typename T, typename U>
inline void check_device(const Dense<T>& A, const Dense<U>& B,
                         const char* caller_name) {

  if (A._location != B._location || A._device_id != B._device_id) {

    std::string message = caller_name;
    message = message + ": argument matrices not on the same device";
    throw excBadArgument(message);

  }

}
/*  \overload
 */
template <typename T, typename U>
inline void check_device(const Sparse<T>& A, const Dense<U>& B,
                         const char* caller_name) {

  if (A._location != B._location || A._device_id != B._device_id) {

    std::string message = caller_name;
    message = message + ": argument matrices not on the same device";
    throw excBadArgument(message);

  }

}

/** \brief            Checks if three matrices are on the same device, throws 
 *                    an exception if not.
 *
 *  \param[in]        A
 *                    First matrix
 *
 *  \param[in]        B
 *                    Second matrix
 *
 *  \param[in]        C
 *                    Third matrix
 *
 *  \param[in]        caller_name
 *                    Name of the calling routine
 */
template <typename T, typename U, typename V>
inline void check_device(const Dense<T>& A, const Dense<U>& B,
                         const Dense<V>& C, const char* caller_name) {

  if (A._location != B._location || A._location != C._location ||
      A._device_id != B._device_id || A._device_id != C._device_id) {

    std::string message = caller_name;
    message = message + ": argument matrices not on the same device";
    throw excBadArgument(message);

  }

}

/** \brief            Checks if a matrix is transposed. Raises an exception if 
 *                    it is.
 *
 *  This helper is used for routines that do not support transposed input.
 *
 *  \param[in]        A
 *                    Matrix to check for transposedness.
 *
 *  \param[in]        caller_name
 *                    Name of the calling routine
 */
template <typename T>
inline void check_input_transposed(const Dense<T>& A, const char* caller_name) {

  if (A._transposed) {

    std::string message = caller_name;
    message = message + ": transposed matrices are not supported as input";
    throw excBadArgument(message);

  }

}
/*  \overload
 */
template <typename T>
inline void check_input_transposed(const Sparse<T>& A,
                                   const char* caller_name) {

  if (A._transposed) {

    std::string message = caller_name;
    message = message + ": transposed matrices are not supported as input";
    throw excBadArgument(message);

  }

}

/** \brief            Checks if a matrix that is supposed to be written to is
 *                    transposed.  Raises an exception if it is.
 *
 *  \param[in]        A
 *                    Matrix to check for transposedness.
 *
 *  \param[in]        caller_name
 *                    Name of the calling routine
 */
template <typename T>
inline void check_output_transposed(const Dense<T>& A,
                                    const char* caller_name) {

  if (A._transposed) {

    std::string message = caller_name;
    message = message + ": transposed matrices are not supported as output";
    throw excBadArgument(message);

  }

}

/** \brief            Checks if a Stream has the same device id as a matrix.
 *                    Raises an exception if the devices are different.
 *
 *  \param[in]        A
 *                    Matrix to check for the device_id.
 *
 *  \param[in]        stream
 *                    Stream to check for the device_id.
 *
 *  \param[in]        caller_name
 *                    Name of the calling routine
 */
template <typename T>
inline void check_stream(const Dense<T>& A, const Stream& stream,
                         const char* caller_name) {

  if (A._device_id != stream.device_id) {

    std::string message = caller_name;
    message = message + ": stream and matrices are associated with different "
                        "devices";
    throw excBadArgument(message);

  }

}

#ifdef HAVE_CUDA
/** \brief            Checks if the global GPU structures are initialized, 
 *                    raises an exception if not.
 *
 *  \param[in]        caller_name
 *                    Name of the calling routine
 */
inline void check_gpu_structures(const char* caller_name) {

  if (!GPU::GPU_structures_initialized) {

    std::string message = caller_name;
    message = message + ": GPU structures are not initialized, call " 
              "LinAlg::GPU::init() before calling any cuBLAS/cuSPARSE/MAGMA "
              "routines";

    throw CUDA::excCUDAError(message);

  }

}
#   endif /* HAVE_CUDA */

/** \brief            Checks if a matrix is in a certain format. Raises an
 *                    exception if it is not.
 *
 *  \param[in]        format
 *                    Format to check for
 *
 *  \param[in]        A
 *                    Matrix to check
 *
 *  \param[in]        caller_name
 *                    Name of the calling routine
 */
template <typename T>
inline void check_format(const Format format, const Dense<T>& A,
                         const char* caller_name) {

  if (A._format != format) {

    std::string message = caller_name;

    if (format == Format::ColMajor) {

      message = message + ": input matrix not in ColMajor format.";

    } else if (format == Format::RowMajor) {

      message = message + ": input matrix not in RowMajor format.";

    }

    throw excBadArgument(message);

  }

}
/** \overload
 */
template <typename T>
inline void check_format(Format format, const Sparse<T>& A,
                         const char* caller_name) {

  if (A._format != format) {

    std::string message = caller_name;

    if (format == Format::CSR) {

      message = message + ": input matrix not in CSR format.";

    } else if (format == Format::CSC) {

      message = message + ": input matrix not in CSC format.";

    }

    throw excBadArgument(message);

  }

}

/** \brief            Checks if a matrix has certain dimensions. Throws an
 *                    exception if it does not.
 *
 *  \param[in]        rows
 *                    Number of rows.
 *
 *  \param[in]        cols
 *                    Number of columns.
 *
 *  \param[in]        A
 *                    Matrix to check.
 *
 *  \param[in]        caller_name
 *                    Name of the calling routine.
 */
template <typename T>
inline void check_dimensions(I_t rows, I_t columns, const Dense<T>& A,
                             const char* caller_name) {

  if (A.rows() != rows || A.cols() != columns) {

    auto message = stringformat("%s: matrix has wrong dimensions: is %dx%d, "
                                "should be %dx%d)", caller_name, A.rows(),
                                A.cols(), rows, columns);

    throw excBadArgument(message);

  }

}
/* \overload
 */
template <typename T>
inline void check_dimensions(I_t rows, I_t columns, const Sparse<T>& A,
                             const char* caller_name) {

  if (A._minimal_index == 0 && A._maximal_index == 0) {

    throw excBadArgument("check_dimensions(): sparse matrix has undefined "
                         "extent, call .update_extent() first");

  }

  auto A_rows = A.rows();
  auto A_cols = A.cols();

  if (A_rows != rows || A_cols != columns) {

    auto message = stringformat("%s: matrix has wrong dimensions: is %dx%d, "
                                "should be %dx%d)", caller_name, A_rows,
                                A_cols, rows, columns);

    throw excBadArgument(message);

  }

}

/** \brief            Checks if a matrix has certain minimal dimensions.  
 *                    Throws an exception if it does not.
 *
 *  \param[in]        rows
 *                    Minimal number of rows.
 *
 *  \param[in]        cols
 *                    Minimal number of columns.
 *
 *  \param[in]        A
 *                    Matrix to check.
 *
 *  \param[in]        caller_name
 *                    Name of the calling routine.
 */
template <typename T>
inline void check_minimal_dimensions(I_t rows, I_t columns, const Dense<T>& A,
                                     const char* caller_name) {

  if (A.rows() < rows || A.cols() < columns) {

    auto message = stringformat("%s: matrix has too small dimensions: is "
                                "%dx%d, should be at least %dx%d)", 
                                caller_name, A.rows(), A.cols(), rows, columns);

    throw excBadArgument(message);

  }

}

/** \brief            Checks two matrices have the same dimensions. Throws an
 *                    exception if they do not.
 *
 *  \param[in]        A
 *                    Matrix to check.
 *
 *  \param[in]        B
 *                    Matrix to check.
 *
 *  \param[in]        caller_name
 *                    Name of the calling routine.
 */
template <typename T, typename U>
inline void check_same_dimensions(const Dense<T>& A, const Dense<U>& B,
                                  const char* caller_name) {

  if (A.rows() != B.rows() || A.cols() != B.cols()) {

    auto message = stringformat("%s: matrices have different dimensions (%dx%d "
                                "and %dx%d)", caller_name, A.rows(), A.cols(), 
                                B.rows(), B.cols());

    throw excBadArgument(message);

  }

}
/** \overload
 */
template <typename T, typename U>
inline void check_same_dimensions(const Sparse<T>& A, const Sparse<U>& B,
                                  const char* caller_name) {

  if (A._n_nonzeros != B._n_nonzeros || A._size != B._size) {

    auto message = stringformat("%s: matrices have different dimensions (A: "
                                "nnz=%d, size=%d / B: nnz=%d, size=%d)", 
                                caller_name, A._n_nonzeros, A._size, 
                                B._n_nonzeros, B._size);

    throw excBadArgument(message);

  }

}

/** \brief            Check if the thread of a stream is alive
 *
 *  \param[in]        stream
 *                    Stream to check.
 *
 *  \param[in]        caller_name
 *                    Name of the calling routine.
 */
inline void check_stream_alive(Stream& stream, const char* caller_name) {

  if (!stream.thread_alive) {

    throw excBadArgument("%s: stream has no active thread (call "
                         "Stream.start_thread() first", caller_name);

  }

}

/** \brief            Check that a stream has prefer_native set
 *
 *  \param[in]        stream
 *                    Stream to check.
 *
 *  \param[in]        caller_name
 *                    Name of the calling routine.
 */
inline void check_stream_prefer_native(Stream& stream,
                                       const char* caller_name) {

  if (!stream.prefer_native) {

    throw excUnimplemented("%s: non native stream support not yet "
                           "implemented", caller_name);

  }

}

/** \brief            Check that a stream has prefer_native not set
 *
 *  \param[in]        stream
 *                    Stream to check.
 *
 *  \param[in]        caller_name
 *                    Name of the calling routine.
 */
inline void check_stream_no_prefer_native(Stream& stream,
                                          const char* caller_name) {

  if (stream.prefer_native) {

    throw excUnimplemented("%s: native stream support not yet implemented", 
                           caller_name);

  }

}

/** \brief            Checks the stream device_id
 *
 *  \param[in]        stream
 *                    Stream to check.
 *
 *  \param[in]        device_id
 *                    Device id to compare against.
 *
 *  \param[in]        caller_name
 *                    Name of the calling routine.
 */
inline void check_stream_device_id(Stream& stream, int device_id,
                                   const char* caller_name) {

  if (stream.device_id != device_id) {

    throw excBadArgument("%s: stream's device id (%d) doesn't match data "
                         "device id (%d)", caller_name, stream.device_id,
                         device_id);

  }

}

#   ifdef HAVE_MPI
/** \brief            Checks whether a rank can be in a given communicator
 * 
 *  \param[in]        rank
 *                    MPI rank.
 *
 *  \param[in]        communicator
 *                    MPI communicator.
 *
 *  \param[in]        caller_name
 *                    Name of the calling routine.
 */
inline void check_rank(int rank, MPI_Comm communicator,
                       const char* caller_name) {

  int comm_size = 0;
  MPI_Comm_size(communicator, &comm_size);

  if (rank >= comm_size) {

    throw excBadArgument("%s: specified rank larger than communicator size",
                         caller_name);

  }

}
#   endif /* HAVE_MPI */

# endif /* LINALG_NO_CHECKS */

#endif /* DOXYGEN_SKIP */

} /* namespace Utilities */

} /* namespace LinAlg */

#endif /* LINALG_UTILITIES_CHECKS_H_ */
