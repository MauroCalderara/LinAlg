/** \file
 *
 *  \brief            Reading/writing IJV files
 *
 *  \date             Created:  Jul 12, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_UTILITIES_IJV_H_
#define LINALG_UTILITIES_IJV_H_

#include <string>     // std::string
#include <fstream>    // std::fstream
#include <sstream>    // std::istringstream

#include "../types.h"
#include "../profiling.h"
#include "../exceptions.h"
#include "stringformat.h"
#include "../dense.h"
#include "../sparse.h"
#include "../fills.h"

namespace LinAlg {

namespace Utilities {

#ifndef DOXYGEN_SKIP
/*  \brief              Support routine for writing matrix data (not to be
 *                      called by the user)
 *
 *  \param[in]          matrix
 *                      Matrix to read data from.
 *
 *  \param[in]          filename
 *                      Name of the file to write data to.
 *
 *  \note               Dense matrices are written in Fortran-style indexing
 *                      with zero values removed, Sparse matrices are written
 *                      with the indexing of the input matrix and in sparsity
 *                      preserving fashion (that is zero elements are written
 *                      explicitly).
 */
template <typename T>
void write_IJV_data(LinAlg::Dense<T>& matrix, std::string filename) {

  PROFILING_FUNCTION_HEADER

  auto rows              = matrix._rows;
  auto columns           = matrix._cols;
  auto leading_dimension = matrix._leading_dimension;
  auto begin             = matrix._begin();
  auto matrix_is_complex = matrix._is_complex();

  std::ofstream file_to_write(filename, std::ios_base::app);

  if (file_to_write.is_open()) {

    using LinAlg::real;
    using LinAlg::imag;

    file_to_write.precision(15);

    // Write the values in row major. Exceptions are caugth by the calling
    // routine
    if (matrix._transposed == false) {
      for (I_t row = 0; row < rows; ++row) {
        for (I_t col = 0; col < columns; ++col) {

          auto value = begin[col * leading_dimension + row];
          if (value != cast<T>(0.0)) {
            file_to_write << row + 1 << " " << col + 1 << " ";
            if (matrix_is_complex) {
              file_to_write << real(value) << imag(value) << "\n";
            } else {
              file_to_write << real(value) << "\n";
            }
          }

        }
      }
    } else {
      for (I_t col = 0; col < columns; ++col) {
        for (I_t row = 0; row < rows; ++row) {

          auto value = begin[col * leading_dimension + row];
          if (value != cast<T>(0.0)) {
            file_to_write << col + 1 << " " << row + 1 << " ";
            if (matrix_is_complex) {
              file_to_write << real(value) << imag(value) << "\n";
            } else {
              file_to_write << real(value) << "\n";
            }
          }

        }
      }
    }
  }

#ifndef LINALG_NO_CHECKS
  else {

    throw excBadArgument("write_IJV_data(): unable to open file (%s) for "
                         "writing.", filename.c_str());

  }
#endif

}
template <typename T>
void write_IJV_data(LinAlg::Sparse<T>& matrix, std::string filename) {

  PROFILING_FUNCTION_HEADER

  auto size              = matrix._size;
  auto first_index       = matrix._first_index;
  auto values            = matrix._values.get();
  auto indices           = matrix._indices.get();
  auto edges             = matrix._edges.get();
  auto matrix_is_complex = matrix._is_complex();

  std::ofstream file_to_write(filename, std::ios_base::app);

  if (file_to_write.is_open()) {

    using LinAlg::real;
    using LinAlg::imag;

    file_to_write.precision(15);

    // Write the values in row major. If there's an exception, the calling
    // routine will catch it.
    for (I_t row = 0; row < size; ++row) {
      for (I_t element = edges[row] - first_index;
           element < edges[row + 1] - first_index;
           ++element) {

        auto col   = indices[element];
        auto value = values[element];

        file_to_write << row + first_index << " " << col << " ";

        if (matrix_is_complex) {
          file_to_write << real(value) << imag(value) << "\n";
        } else {
          file_to_write << value << "\n";
        }

      }
    }
  }

#ifndef LINALG_NO_CHECKS
  else {

    throw excBadArgument("write_IJV_data(): unable to open file (%s) for "
                         "writing.", filename.c_str());

  }
#endif

}

#endif /* DOXYGEN_SKIP */

/** \brief              Write a matrix to a file in IJV format.
 *
 *  \param[in]          matrix
 *                      Matrix to read data from.
 *
 *  \param[in]          filename
 *                      Name of the file to write to (file is created if it
 *                      doesn't exist and overwritten if it exists).
 *
 *  \note               IJV format is
 *
 *                        row col real [imag]
 *
 *                      sorted in row major (basically the same as CSR but
 *                      without the header).
 */
template <typename T>
inline void write_IJV(LinAlg::Dense<T>& matrix, std::string filename) {

  PROFILING_FUNCTION_HEADER

  if (matrix._location != Location::host) {

    // Create a temporary matrix located in main memory and try again
    Dense<T> temporary;
    temporary.clone_from(matrix);
    temporary.location(Location::host);
    write_IJV(temporary, filename);

    return;

  }

  if (matrix._transposed) {

    // Create a temporary matrix with the transposed contents and try again
    Dense<T> temporary(matrix.rows(), matrix.cols());
    temporary << matrix;
    write_IJV(temporary, filename); 

    return;

  }

  std::ofstream file_to_write(filename, std::ios_base::trunc);

  if (file_to_write.is_open()) {

    file_to_write.close();

  }

#ifndef LINALG_NO_CHECKS
  else {

    throw excBadArgument("write_IJV(): unable to open file (%s) for writing.",
                         filename.c_str());

  }

  try {
#endif

    // Write the data
    write_IJV_data(matrix, filename);

#ifndef LINALG_NO_CHECKS
  } catch(std::ofstream::failure err) {

    throw excBadFile("write_IJV(): Output file (%s:%d): write error.",
                     filename.c_str());

  }
#endif

}
/** \overload
 *
 *  \param[in]          matrix
 *                      Matrix to read data from.
 *
 *  \param[in]          filename
 *                      Name of the file to write to (file is created if it
 *                      doesn't exist and overwritten if it exists).
 *
 */
template <typename T>
inline void write_IJV(LinAlg::Sparse<T>& matrix, std::string filename) {

  PROFILING_FUNCTION_HEADER

  std::ofstream file_to_write(filename, std::ios_base::trunc);

  if (file_to_write.is_open()) {

    file_to_write.close();

  }

#ifndef LINALG_NO_CHECKS
  else {

    throw excBadArgument("write_IJV(): unable to open file (%s) for writing.",
                         filename.c_str());

  }

  try {
#endif

    // Write the data
    write_IJV_data(matrix, filename);

#ifndef LINALG_NO_CHECKS
  } catch(std::ofstream::failure err) {

    throw excBadFile("write_IJV(): Output file (%s:%d): write error.",
                     filename.c_str());

  }
#endif

}

/** \overload
 *
 *  \param[in]          matrix
 *                      Matrix to read data from.
 *
 *  \param[in]          formatstring
 *                      Formatstring for the filename to write to (file is 
 *                      created if it doesn't exist and overwritten if it 
 *                      exists).
 *
 *  \param[in]          formatargs
 *                      Formatargs for the file to write to.
 */
template <typename T, typename... Us>
inline void write_IJV(LinAlg::Dense<T>& matrix, const char* formatstring,
                      Us... formatargs) {
  std::string filename_str = stringformat(formatstring, formatargs...);
  write_IJV(matrix, filename_str);
}

/** \overload
 *
 *  \param[in]          matrix
 *                      Matrix to read data from.
 *
 *  \param[in]          filename
 *                      Name of the file to write to (file is created if it
 *                      doesn't exist and overwritten if it exists).
 */
template <typename T>
inline void write_IJV(LinAlg::Dense<T>& matrix, const char* filename) {
  std::string filename_str = filename;
  write_IJV(matrix, filename_str);
}

/** \overload
 *
 *  \param[in]          matrix
 *                      Matrix to read data from.
 *
 *  \param[in]          formatstring
 *                      Formatstring for the filename to write to (file is 
 *                      created if it doesn't exist and overwritten if it 
 *                      exists).
 *
 *  \param[in]          formatargs
 *                      Formatargs for the file to write to.
 */
template <typename T, typename... Us>
inline void write_IJV(LinAlg::Sparse<T>& matrix, const char* formatstring,
                      Us... formatargs) {
  std::string filename_str = stringformat(formatstring, formatargs...);
  write_IJV(matrix, filename_str);
}

/** \overload
 *
 *  \param[in]          matrix
 *                      Matrix to read data from.
 *
 *  \param[in]          filename
 *                      Name of the file to write to (file is created if it
 *                      doesn't exist and overwritten if it exists).
 */
template <typename T>
inline void write_IJV(LinAlg::Sparse<T>& matrix, const char* filename) {
  std::string filename_str = filename;
  write_IJV(matrix, filename_str);
}

} /* namespace LinAlg::Utilities */

} /* namespace LinAlg */


#endif /* LINALG_UTILITIES_IJV_H_ */
