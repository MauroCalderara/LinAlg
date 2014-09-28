/** \file
 *
 *  \brief            Reading/writing CSR files
 *
 *  \date             Created:  Jul 12, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_UTILITIES_CSR_H_
#define LINALG_UTILITIES_CSR_H_

#include <iostream>   // std::cout
#include <string>     // std::string
#include <tuple>      // std::tuple, std::tie
#include <fstream>    // std::fstream
#include <sstream>    // std::istringstream
#include <utility>    // std::move

#include "../types.h"
#include "../exceptions.h"
#include "stringformat.h"
#include "misc.h"           // Utilities::goto_line
#include "../dense.h"
#include "../sparse.h"
#include "../fills.h"

#include "IJV.h"

namespace LinAlg {

namespace Utilities {

// See src/utilities/CSR.cc
std::tuple<I_t, I_t, I_t, bool> parse_CSR_header(std::string filename);

// See src/utilities/CSR.cc
std::tuple<I_t, I_t, I_t, bool> parse_CSR_body(std::string filename);

/** \brief              Read from a CSR file into a matrix.
 *
 *  This function parses the CSR file and checks if the dimensions of the matrix
 *  in the file and the target matrix match. If the target matrix is empty (rows
 *  and cols are set to 0), memory is allocated as required for the matrix in
 *  the file.
 *
 *  \param[out]         matrix
 *                      Matrix to read data into.
 *
 *  \param[in]          filename
 *                      Name of the file to read from.
 *
 *  \note               1 - The CSR format is not very convenient to load into
 *                          dense matrices as the whole file needs to be parsed
 *                          twice.
 *                      2 - For performance reasons the routine checks the file
 *                          content not as closely for reading into sparse
 *                          matrices as it does for dense since only one pass
 *                          through the file is required. In particular the
 *                          lines are not checked against a regexp for proper
 *                          formatting. In case of invalid input the routine
 *                          may throw at your inconvenience. To check the
 *                          input, call parse_CSR_body(filename) before
 *                          loading into a sparse matrix.
 */
template <typename T>
void read_CSR(LinAlg::Dense<T>& matrix, std::string filename) {

#ifndef LINALG_NO_CHECKS
  if (matrix._location != Location::host) {

    throw excUnimplemented("read_CSR(): can only read to matrices in main "
                           "memory.");

  }

  // TODO: check for empty file

#endif

  I_t rows, columns, n_nonzeros;
  bool file_is_complex;

  std::tie(rows, columns, n_nonzeros, file_is_complex) =
                                                  parse_CSR_body(filename);

  if (matrix._rows == 0) {

    matrix.reallocate(rows, columns);
    Fills::zero(matrix);

  }
#ifndef LINALG_NO_CHECKS
  else if (matrix._rows != rows || matrix._cols != columns) {

    throw excBadArgument("read_CSR(): output matrix size (%dx%d) doesn't " \
                         "match matrix in file (%dx%d)\n", matrix._rows,
                         matrix._cols, rows, columns);

  } else if (matrix._is_complex() != file_is_complex) {

    throw excBadArgument("read_CSR(): output matrix has (complex=%s) and " \
                         "matrix in file has (complex=%s)\n",
                         matrix._is_complex() ? "true" : "false",
                         file_is_complex ? "true" : "false");

  }
#endif

  // Read the file content. Since we already parsed the file once, we can be a
  // bit more agressive here:
  using std::ifstream;
  using std::string;
  using std::getline;
  using std::istringstream;

  ifstream file_to_read(filename);

  if (file_to_read.is_open()) {

    I_t line_num = 0;

#ifndef LINALG_NO_CHECKS
    file_to_read.exceptions(ifstream::failbit | ifstream::badbit |
                            ifstream::eofbit);

    try {
#endif

      string line;
      istringstream linestream;

      auto m_data = matrix._begin();
      auto ld = matrix._leading_dimension;
      I_t i, j;

      // Skip the first three lines
      line_num = 4;
      goto_line(file_to_read, line_num);

      // Read the data
      if (file_is_complex) {

        double real, imag;

        for (I_t element = 0; element < n_nonzeros; ++element) {

          getline(file_to_read, line);

          // Input has already been checked by parse_CSR_body()
          linestream.str(line); linestream.clear();
          linestream >> i >> j >> real >> imag;
          m_data[j * ld + i] = cast<T>(real, imag);

          ++line_num;

        }

      } else {

        double real;

        for (I_t element = 0; element < n_nonzeros; ++element) {

          getline(file_to_read, line);

          linestream.str(line); linestream.clear();
          linestream >> i >> j >> real;
          m_data[j * ld + i] = cast<T>(real);

          ++line_num;

        }

      }

#ifndef LINALG_NO_CHECKS
    } catch(ifstream::failure err) {

      throw excBadFile("read_CSR(): Input file (%s:%d): premature end or read "
                       "error.", filename.c_str(), line_num);

    }
#endif

    file_to_read.close();

  }

#ifndef LINALG_NO_CHECKS
  else {

    throw excBadArgument("read_CSR(): unable to open file (%s) for reading.",
                         filename.c_str());

  }
#endif

};
/** \overload
 *
 *  \param[out]         matrix
 *                      Matrix to read data into.
 *
 *  \param[in]          filename
 *                      Name of the file to read from.
 */
template <typename T>
void read_CSR(LinAlg::Sparse<T>& matrix, std::string filename) {

#ifndef LINALG_NO_CHECKS
  if (matrix._location != Location::host) {

    throw excUnimplemented("read_CSR(): can only read to matrices in main "
                           "memory.");

  }

  // TODO: check for empty file

#endif

  I_t size, n_nonzeros, first_index;
  bool file_is_complex;
  std::tie(size, n_nonzeros, first_index, file_is_complex) =
                                                     parse_CSR_header(filename);

#ifndef LINALG_NO_CHECKS
  if (matrix._is_complex() != file_is_complex) {

    if (file_is_complex) {

      throw excBadArgument("read_CSR(): input matrix is real, file (%s) is "
                           "complex.", filename.c_str());

    } else {

      throw excBadArgument("read_CSR(): input matrix is complex, file (%s) is "
                           "real.", filename.c_str());

    }

  }
#endif

  if (matrix._size == 0) {

    matrix.reallocate(size, n_nonzeros, matrix._location, matrix._device_id);

  }

#ifndef LINALG_NO_CHECKS
  else if (matrix._size != size) {

    throw excBadArgument("read_CSR(): matrix in input file and matrix to write "
                         "to have different sizes");

  } else if (matrix._n_nonzeros != n_nonzeros) {

    throw excBadArgument("read_CSR(): matrix in input file and matrix to write "
                         "to have different number of non zero entries");

  }
#endif

  // Read the data. For efficiency reasons we assume and check for row major
  // order in the entries.
  using std::ifstream;

  ifstream file_to_read(filename, std::ios_base::in);

  if (file_to_read.is_open()) {

    I_t element;

#ifndef LINALG_NO_CHECKS
    // Bumping into the end of the file within this loop is an error (as the
    // file should have n_nonzero + 3 lines).
    file_to_read.exceptions(ifstream::failbit | ifstream::badbit |
                            ifstream::eofbit);

    try {
#endif

      auto values  = matrix._values.get();
      auto indices = matrix._indices.get();
      auto edges   = matrix._edges.get();

      I_t i, j;
      I_t last_i = first_index;
      bool format_real_ok = true;
      bool format_cmpx_ok = true;

      double real;
      double imag = 0;

      // Skip header (already parsed).
      goto_line(file_to_read, 4);

      // Read data
      edges[0] = first_index;
      for (element = 0; element < n_nonzeros; ++element) {

        format_real_ok = file_to_read >> i >> j >> real;
        if (file_is_complex) { format_cmpx_ok = file_to_read >> imag; }

#ifndef LINALG_NO_CHECKS
        if (format_real_ok == false || format_cmpx_ok == false) {
          throw excBadFile("read_CSR(): Input file (%s:%d): invalid "
                           "formatting.", filename.c_str(), element + 4);
        }
#endif

        values[element] = cast<T>(real, imag);
        indices[element] = j;

        if (i > last_i) {
          for (I_t edge = last_i + 1; edge < i + 1; ++edge) {
            edges[edge - first_index] = element + first_index;
          }
        }

        last_i = i;
      }
      edges[size] = n_nonzeros + first_index;

      // TODO: Optionally check for consistency

#ifndef LINALG_NO_CHECKS
    } catch(ifstream::failure err) {

      std::cout << err.what() << "\n";

      throw excBadFile("read_CSR(): Input file (%s:%d): premature end or read "
                       "error.", filename.c_str(), element + 4);

    }
#endif

    matrix._first_index = first_index;

    file_to_read.close();

  }

#ifndef LINALG_NO_CHECKS
  else {

    throw excBadArgument("read_CSR(): unable to open file (%s) for reading.",
                         filename.c_str());

  }
#endif

};
/** \overload
 *
 *  \param[out]         matrix
 *                      Matrix to read data into.
 *
 *  \param[in]          formatstring
 *                      Formatstring for the file to read from.
 *
 *  \param[in]          formatargs
 *                      Formatargs for the file to read from.
 */
template <typename T, typename... Us>
inline void read_CSR(LinAlg::Dense<T>& matrix, const char* formatstring,
                     Us... formatargs) {
  std::string filename_str = stringformat(formatstring, formatargs...);
  read_CSR(matrix, filename_str);
};
/** \overload
 *
 *  \param[out]         matrix
 *                      Matrix to read data into.
 *
 *  \param[in]          formatstring
 *                      Formatstring for the file to read from.
 *
 *  \param[in]          formatargs
 *                      Formatargs for the file to read from.
 */
template <typename T, typename... Us>
inline void read_CSR(LinAlg::Sparse<T>& matrix, const char* formatstring,
                     Us... formatargs) {
  std::string filename_str = stringformat(formatstring, formatargs...);
  read_CSR(matrix, filename_str);
};
/** \overload
 *
 *  \param[out]         matrix
 *                      Matrix to read data into.
 *
 *  \param[in]          filename
 *                      Name of the file to read from.
 */
template <typename T>
inline void read_CSR(LinAlg::Dense<T>& matrix, const char* filename) {
  std::string filename_str = filename;
  read_CSR(matrix, filename_str);
};
/** \overload
 *
 *  \param[out]         matrix
 *                      Matrix to read data into.
 *
 *  \param[in]          filename
 *                      Name of the file to read from.
 */
template <typename T>
inline void read_CSR(LinAlg::Sparse<T>& matrix, const char* filename) {
  std::string filename_str = filename;
  read_CSR(matrix, filename_str);
};

/** \brief              Write a matrix to a file in CSR format.
 *
 *  \param[in]          matrix
 *                      Matrix to read data from.
 *
 *  \param[in]          filename
 *                      Name of the file to write to.
 *
 *  \note               Dense matrices are written in C-style indexing with zero
 *                      values removed, sparse matrices are written with the
 *                      indexing of the input matrix and in sparsity preserving
 *                      fashion (that is zero elements are written explicitly).
 */
template <typename T>
void write_CSR(LinAlg::Dense<T>& matrix, std::string filename) {

  if (matrix._location != Location::host) {

    // Create a temporary matrix located in main memory and try again
    Dense<T> temporary = matrix;
    temporary.location(Location::host);
    write_CSR(temporary, filename);

    return;

  }

  if (matrix._transposed) {

    // Create a temporary matrix with the transposed contents and try again
    Dense<T> temporary(matrix.rows(), matrix.cols());
    temporary << matrix;
    write_CSR(temporary, filename);

    return;

  }

  std::ofstream file_to_write(filename, std::ios_base::trunc);

  if (file_to_write.is_open()) {

    auto rows              = matrix._rows;
    auto columns           = matrix._cols;
    auto leading_dimension = matrix._leading_dimension;
    auto begin             = matrix._begin();

    // Count all non zero elements
    I_t n_nonzeros = 0;
    for (I_t col = 0; col < columns; ++col) {
      for (I_t row = 0; row < rows; ++row) {
        if (begin[col * leading_dimension + row] == cast<T>(0.0)) {
          ++n_nonzeros;
        }
      }
    }


#ifndef LINALG_NO_CHECKS
    try {
#endif

      // Write the header
      file_to_write << rows << "\n";
      file_to_write << n_nonzeros << "\n";
      file_to_write << 0 << "\n";
      file_to_write.close();

      // Write the data
      write_IJV_data(matrix, file_to_write);

#ifndef LINALG_NO_CHECKS
    } catch(std::ofstream::failure err) {

      throw excBadFile("write_CSR(): Output file (%s:%d): write error.",
                       filename.c_str());

    }
#endif
  }
#ifndef LINALG_NO_CHECKS
  else {

    throw excBadArgument("write_CSR(): unable to open file (%s) for writing.",
                         filename.c_str());

  }
#endif

}
/** \overload
 *
 *  \param[in]          matrix
 *                      Matrix to write data from.
 *
 *  \param[in]          filename
 *                      Name of the file to write to.
 */
template <typename T>
void write_CSR(LinAlg::Sparse<T>& matrix, std::string filename) {

#ifndef LINALG_NO_CHECKS
  if (matrix._location != Location::host) {

    throw excUnimplemented("write_CSR(): can only write out matrices located "
                           "in main memory.");

  } else if (matrix._format != Format::CSR) {

    throw excUnimplemented("write_CSR(): can only write sparse matrices if "
                           "they are in CSR format.");

  } else if (matrix._transposed) {

    throw excUnimplemented("write_CSR(): can only write sparse matrices if "
                           "they are not transposed.");

  }
#endif

  std::ofstream file_to_write(filename, std::ios_base::trunc);

  if (file_to_write.is_open()) {

    auto size              = matrix._size;
    auto n_nonzeros        = matrix._n_nonzeros;
    auto first_index       = matrix._first_index;

#ifndef LINALG_NO_CHECKS
    try {
#endif

      // Write the header
      file_to_write << size << "\n";
      file_to_write << n_nonzeros << "\n";
      file_to_write << first_index << "\n";
      file_to_write.close();

      // Write the data
      write_IJV_data(matrix, filename);

#ifndef LINALG_NO_CHECKS
    } catch(std::ofstream::failure err) {

      throw excBadFile("write_CSR(): Output file (%s:%d): write error.",
                       filename.c_str());

    }
#endif
  }
#ifndef LINALG_NO_CHECKS
  else {

    throw excBadArgument("write_CSR(): unable to open file (%s) for writing.",
                         filename.c_str());

  }
#endif

};

/** \overload
 *
 *  \param[out]         matrix
 *                      Matrix to read data into.
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
inline void write_CSR(LinAlg::Dense<T>& matrix, const char* formatstring,
                      Us... formatargs) {
  std::string filename_str = stringformat(formatstring, formatargs...);
  write_CSR(matrix, filename_str);
};

/** \overload
 *
 *  \param[out]         matrix
 *                      Matrix to read data into.
 *
 *  \param[in]          filename
 *                      Name of the file to read from.
 */
template <typename T>
inline void write_CSR(LinAlg::Dense<T>& matrix, const char* filename) {
  std::string filename_str = filename;
  write_CSR(matrix, filename_str);
};

/** \overload
 *
 *  \param[out]         matrix
 *                      Matrix to read data into.
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
inline void write_CSR(LinAlg::Sparse<T>& matrix, const char* formatstring,
                      Us... formatargs) {
  std::string filename_str = stringformat(formatstring, formatargs...);
  write_CSR(matrix, filename_str);
};

/** \overload
 *
 *  \param[out]         matrix
 *                      Matrix to read data into.
 *
 *  \param[in]          filename
 *                      Name of the file to read from.
 */
template <typename T>
inline void write_CSR(LinAlg::Sparse<T>& matrix, const char* filename) {
  std::string filename_str = filename;
  write_CSR(matrix, filename_str);
};

} /* namespace LinAlg::Utilities */

} /* namespace LinAlg */


#endif /* LINALG_UTILITIES_CSR_H_ */
