/** \file             checks.h
 *
 *  \brief            Routines to facilitate argument checking.
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

#include "types.h"
#include "exceptions.h"

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
    throw excBadArgument(message.c_str());

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
    throw excBadArgument(message.c_str());

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
    throw excBadArgument(message.c_str());

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
    throw excBadArgument(message.c_str());

  }

}

/*  \brief            Checks if a stream has the same device id as a matrix.
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
    throw excBadArgument(message.c_str());

  }
};

#endif /* LINALG_NO_CHECKS */

#endif /* DOXYGEN_SKIP */

} /* namespace Utilities */

} /* namespace LinAlg */

#endif /* LINALG_UTILITIES_CHECKS_H_ */
