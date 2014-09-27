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

#include "../types.h"
#include "../exceptions.h"
#include "../streams.h"
#include "stringformat.h"

namespace LinAlg {

namespace Utilities {

#ifndef DOXYGEN_SKIP

#ifndef LINALG_NO_CHECKS
/*  \brief            Checks if two matrices are on the same device, throws an
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
inline void check_device(Dense<T>& A, Dense<U>& B, const char* caller_name) {

  if (A._location != B._location || A._device_id != B._device_id) {

    std::string message = caller_name;
    message = message + ": argument matrices not on the same device";
    throw excBadArgument(message);

  }

}

/*  \brief            Checks if three matrices are on the same device, throws an
 *                    exception if not.
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
inline void check_device(const Dense<T>& A, const Dense<U>& B, Dense<V>& C,
                         const char* caller_name) {

  if (A._location != B._location || A._location != C._location ||
      A._device_id != B._device_id || A._device_id != C._device_id) {

    std::string message = caller_name;
    message = message + ": argument matrices not on the same device";
    throw excBadArgument(message);

  }

}

/*  \brief            Checks if a matrix is transposed.  Raises an exception if
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
inline void check_input_transposed(Dense<T>& A, const char* caller_name) {

  if (A._transposed) {

    std::string message = caller_name;
    message = message + ": transposed matrices are not supported as input";
    throw excBadArgument(message);

  }

}

/*  \brief            Checks if a matrix that is supposed to be written to is
 *                    transposed.  Raises an exception if it is.
 *
 *  \param[in]        A
 *                    Matrix to check for transposedness.
 *
 *  \param[in]        caller_name
 *                    Name of the calling routine
 */
template <typename T>
inline void check_output_transposed(Dense<T>& A, const char* caller_name) {

  if (A._transposed) {

    std::string message = caller_name;
    message = message + ": transposed matrices are not supported as output";
    throw excBadArgument(message);

  }

}

#ifdef HAVE_CUDA
/*  \brief            Checks if a CUDAstream has the same device id as a matrix.
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
inline void check_stream(Dense<T>& A, CUDAStream stream,
                         const char* caller_name) {

  if (A._device_id != stream.device_id) {

    std::string message = caller_name;
    message = message + ": stream and matrices are associated with different "
                        "devices";
    throw excBadArgument(message);

  }
};
#endif /* HAVE_CUDA */

/*  \brief            Checks if a matrix is in a certain format. Raises an
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
inline void check_format(Format format, Dense<T>& A, const char* caller_name) {

  if (A._format != format) {

    std::string message = caller_name;

    if (format == Format::ColMajor) {

      message = message + ": input matrix not in ColMajor format.";

    } else if (format == Format::RowMajor) {

      message = message + ": input matrix not in RowMajor format.";

    }

    throw excBadArgument(message);

  }

};

/*  \brief            Checks if a matrix has certain dimensions. Throws an
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
inline void check_dimensions(I_t rows, I_t columns, Dense<T>& A,
                             const char* caller_name) {

  if (A.rows() != rows || A.cols() != columns) {

    auto message = stringformat("%s: matrix has wrong dimensions is %dx%d, "
                                "should be %dx%d)", caller_name, A.rows(),
                                A.cols(), rows, columns);

    throw excBadArgument(message);

  }

};

/*  \brief            Checks two matrices have the same dimensions. Throws an
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
inline void check_same_dimensions(Dense<T>& A, Dense<U>& B,
                                  const char* caller_name) {

  if (A.rows() != B.rows() || A.cols() != B.cols()) {

    auto message = stringformat("%s: matrices have different dimensions (%dx%d "
                                "and %dx%d)", caller_name, A.rows(), A.cols(), 
                                B.rows(), B.cols());

    throw excBadArgument(message);

  }

};

#endif /* LINALG_NO_CHECKS */

#endif /* DOXYGEN_SKIP */

} /* namespace Utilities */

} /* namespace LinAlg */

#endif /* LINALG_UTILITIES_CHECKS_H_ */
