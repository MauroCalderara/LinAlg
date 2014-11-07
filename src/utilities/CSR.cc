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

//#include <iostream>
#include <fstream>      // std::ifstream
#include <string>       // std::stream
#include <tuple>        // std::tuple

#include "types.h"
#include "profiling.h"
#include "utilities/CSR.h"
#include "utilities/stringformat.h"

namespace LinAlg {

namespace Utilities {

/** \brief              Parses the header of a CSR file to determine the size, 
 *                      number of non zero elements, indexing and complexness  
 *                      of the matrix contained
 *
 *  \param[in]          filename
 *                      Path to the CSR file
 *
 *  \return             A std::tuple<I_t, I_t, I_t, bool> with the elements:
 *                      size, n_nonzeros, first_index, and is_complex. Those 
 *                      are the number of rows (for a CSR matrix) or columns 
 *                      (for a CSC matrix), the number of non zero elements, 
 *                      the index of the first element (0 for C style, 1 for 
 *                      FORTRAN style indexing), and 'true' if the matrix has 
 *                      complex data type (C_t or Z_t) and false otherwise
 *
 *  Example usage:
 *  \code
 *     using std::tie;
 *     tie(size, n_nonzeros, first_index, is_complex)= parse_CSR_header(file);
 *  \endcode
 */
std::tuple<I_t, I_t, I_t, bool> parse_CSR_header(std::string filename) {

  PROFILING_FUNCTION_HEADER

  using std::ifstream;
  using std::string;
  using std::getline;
  using std::istringstream;
  using std::tuple;

  I_t size = 0;
  I_t n_nonzeros = 0;
  I_t first_index = 0;
  bool matrix_is_cmpx;

  ifstream file_to_parse(filename);
#ifndef LINALG_NO_CHECKS
  if (file_to_parse.is_open()) {
#endif

    string line;
    istringstream linestream;
    bool line_is_ok;
    I_t line_num = 0;

#ifndef LINALG_NO_CHECKS
    try {

      // Throw if any of this fails or we reach the end of the file before 4
      // lines
      file_to_parse.exceptions(ifstream::failbit | ifstream::badbit |
                               ifstream::eofbit);
#endif

      // Parse the header

      // First line: size
      line_num = 1;
      getline(file_to_parse, line);
      linestream.str(line); linestream.clear();
      line_is_ok = linestream >> size;
#ifndef LINALG_NO_CHECKS
      if (!line_is_ok) {
        throw excBadFile("parse_CSR_header(): error in %s:%d", filename.c_str(),
                         line_num);
      }
#endif

      // Second line: number of non_zero elements
      line_num = 2;
      getline(file_to_parse, line);
      linestream.str(line); linestream.clear();
      line_is_ok = linestream >> n_nonzeros;
#ifndef LINALG_NO_CHECKS
      if (!line_is_ok) {
        throw excBadFile("parse_CSR_header(): error in %s:%d", filename.c_str(),
                         line_num);
      }
#endif

      // Third line: index of first element
      line_num = 3;
      getline(file_to_parse, line);
      linestream.str(line); linestream.clear();
      line_is_ok = linestream >> first_index;
#ifndef LINALG_NO_CHECKS
      if (!line_is_ok) {
        throw excBadFile("parse_CSR_header(): error in %s:%d", filename.c_str(),
                         line_num);
      }
#endif

      // The fourth line determines whether the matrix is real or complex
      line_num = 4;
      getline(file_to_parse, line);
      linestream.str(line); linestream.clear();
      I_t i, j;
      double real, imag;
      matrix_is_cmpx = linestream >> i >> j >> real >> imag;

#ifndef LINALG_NO_CHECKS
    } catch(ifstream::failure err) {

      throw excBadFile("parse_CSR_header(): Input file (%s:%d): premature end "
                       "or read error.", filename.c_str(), line_num);

    }
#endif

    file_to_parse.close();

#ifndef LINALG_NO_CHECKS
  } else {

    throw excBadArgument("parse_CSR_header(): unable to open file %s",
                         filename.c_str());

  }
#endif

  return tuple<I_t, I_t, I_t, bool>(size, n_nonzeros, first_index,
                                    matrix_is_cmpx);

}

/** \brief              Parses a CSR file to determine the size of the matrix
 *                      contained.
 *
 *  \param[in]          filename
 *                      Path to the CSR file.
 *
 *  \return             A std::tuple<I_t, I_t, bool> with the elements: rows,
 *                      colums, n_nonzeros, is_complex.
 *
 *  \todo               Have an example of how to use this using idiomatic C++
 *                      with std::tie:
 *
 *  Example usage:
 *  \code
 *       std::tie(rows, cols, n_nonzeros, is_complex) = parse_CSR_body(file);
 *  \endcode
 */
std::tuple<I_t, I_t, I_t, bool> parse_CSR_body(std::string filename) {

  PROFILING_FUNCTION_HEADER

  using std::ifstream;
  using std::string;
  using std::getline;
  using std::istringstream;
  //using std::regex_match;
  using std::tuple;

  I_t header_rows, header_n_nonzeros;
  I_t rows = 0;
  I_t columns = 0;
  bool matrix_is_cmpx;
  bool parse_ok;
  istringstream linestream;

  ifstream file_to_parse(filename);
  if (file_to_parse.is_open()) {

    int line_num;
    int first_index;    // fortran index
    string line;

#ifndef LINALG_NO_CHECKS
    try {

      // Throw if we can't read the first 4 lines
      file_to_parse.exceptions(ifstream::failbit | ifstream::badbit |
                               ifstream::eofbit);
#endif

      // Read header_rows
      line_num = 1;
      getline(file_to_parse, line);
      linestream.str(line); linestream.clear();
      parse_ok = linestream >> header_rows;
#ifndef LINALG_NO_CHECKS
      if (!parse_ok) {
        throw excBadFile("parse_CSR_body(): error in %s:%d", filename.c_str(),
                         line_num);
      }
#endif

      // Read header_n_nonzeros
      line_num = 2;
      getline(file_to_parse, line);
      linestream.str(line); linestream.clear();
      parse_ok = linestream >> header_n_nonzeros;
#ifndef LINALG_NO_CHECKS
      if (!parse_ok) {
        throw excBadFile("parse_CSR_body(): error in %s:%d", filename.c_str(),
                         line_num);
      }
#endif

      // Read fortran_index / first_index
      line_num = 3;
      getline(file_to_parse, line);
      linestream.str(line); linestream.clear();
      parse_ok = linestream >> first_index;
#ifndef LINALG_NO_CHECKS
      if (!parse_ok) {
        throw excBadFile("parse_CSR_body(): error in %s:%d", filename.c_str(),
                         line_num);
      }
#endif

      // The fourth line determines whether the matrix is real or complex
      line_num = 4;
      getline(file_to_parse, line);
      linestream.str(line); linestream.clear();
      I_t i, j;
      double real, imag;
      matrix_is_cmpx = linestream >> i >> j >> real >> imag;

      // Update the counted rows and columns
      rows = ((i - first_index + 1) > rows) ? (i - first_index + 1) : rows;
      columns = ((j - first_index + 1) > columns) ?
                j - first_index + 1 : columns;

      // Parse the rest of the file.
      line_num = 5;
      for (I_t element = 1; element < header_n_nonzeros; ++element) {

        getline(file_to_parse, line);
        linestream.str(line); linestream.clear();

        if (matrix_is_cmpx) {
          parse_ok = linestream >> i >> j >> real >> imag;
        } else {
          parse_ok = linestream >> i >> j >> real;
        }

#ifndef LINALG_NO_CHECKS
        if (!parse_ok) {
          throw excBadFile("parse_CSR_body(): error in %s:%d", filename.c_str(),
                           line_num);
        }
#endif

        rows = ((i - first_index + 1) > rows) ? (i - first_index + 1) : rows;
        columns = ((j - first_index + 1) > columns) ?
                  j - first_index + 1 : columns;

        ++line_num;

      }

#ifndef LINALG_NO_CHECKS
    } catch(ifstream::failure err) {

      throw excBadFile("parse_CSR_body(): Input file (%s:%d): premature end or "
                       "read error (%s).", filename.c_str(), line_num, err.what());

    }
#endif

    file_to_parse.close();

  }

#ifndef LINALG_NO_CHECKS
  else {

    throw excBadArgument("parse_CSR_body(): unable to open file %s",
                         filename.c_str());

  }
#endif

  return tuple<I_t, I_t, I_t, bool>(rows, columns, header_n_nonzeros,
                                    matrix_is_cmpx);

}

} /* namespace LinAlg::Utilities */

} /* namespace LinAlg */
