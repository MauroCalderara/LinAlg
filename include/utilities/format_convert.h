/** \file
 *
 *  \brief            Conversion of storage formats
 *
 *  \date             Created:  Jul 12, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_UTILITIES_FORMAT_CONVERT_H_
#define LINALG_UTILITIES_FORMAT_CONVERT_H_

#include <tuple>      // std::tie

#include "../preprocessor.h"
#include "../types.h"
#include "../profiling.h"
#include "../exceptions.h"
#include "../dense.h"
#include "../sparse.h"
#include "../fills.h"
#include "checks.h"

namespace LinAlg {

namespace Utilities {

/** \brief            Count non-zeros in a sub block of a matrix
 *
 *  \param[in]        matrix
 *                    Matrix from which to count the non-zeros
 *
 *  \param[in]        sub_block
 *                    Submatrix specification (C-style indexing).
 */
template <typename T>
inline I_t count_nonzeros(const Dense<T>& matrix, SubBlock sub_block) {

  PROFILING_FUNCTION_HEADER

#ifndef LINALG_NO_CHECKS
  if (matrix._location != Location::host) {
    throw excUnimplemented("count_nonzeros(): counting non_zero elements is "
                           "only implemented for matrices in main memory.");
  } else if (matrix._transposed) {
    throw excUnimplemented("count_nonzeros(): counting non_zero elements in "
                           "transposed matrices is not implemented.");
  } 

  check_format(Format::ColMajor, matrix, "count_nonzeros()");

  if ((sub_block.first_row > matrix.rows() - 1) || 
      (sub_block.last_row > matrix.rows())         ) {
    throw excBadArgument("count_nonzeros(): requested row range is not "
                         "contained in the source matrix.");
  } else if ((sub_block.first_col > matrix.cols() - 1) || 
             (sub_block.last_col > matrix.cols())         ) {
    throw excBadArgument("count_nonzeros(): requested col range is not "
                         "contained in the source matrix."); }
#endif

  I_t  n_nonzeros        = 0;
  auto data              = matrix._data.get();
  auto leading_dimension = matrix._leading_dimension;
  auto zero              = cast<T>(0.0);

  for (auto col = sub_block.first_col; col < sub_block.last_col; ++col) {
  
    for (auto row = sub_block.first_row; row < sub_block.last_row; ++row) {
    
      if (data[col * leading_dimension + row] != zero) ++n_nonzeros;
    
    }
  
  }

  return n_nonzeros;

}
/** \overload
 */
template <typename T>
inline I_t count_nonzeros(const Sparse<T>& matrix, SubBlock sub_block) {

  PROFILING_FUNCTION_HEADER

#ifndef LINALG_NO_CHECKS
  if (matrix._location != Location::host) {
    throw excUnimplemented("count_nonzeros(): counting non_zero elements is "
                           "only implemented for matrices in main memory.");
  } else if (matrix._transposed) {
    throw excUnimplemented("count_nonzeros(): counting non_zero elements in "
                           "transposed matrices is not implemented.");
  } 

  check_format(Format::CSR, matrix, "count_nonzeros()");

  if ((sub_block.first_row > matrix.rows() - 1) || 
      (sub_block.last_row > matrix.rows())         ) {
    throw excBadArgument("count_nonzeros(): requested row range is not "
                         "contained in the source matrix.");
  } else if ((sub_block.first_col > matrix.cols() - 1) || 
             (sub_block.last_col > matrix.cols())         ) {
    throw excBadArgument("count_nonzeros(): requested col range is not "
                         "contained in the source matrix.");
  }
#endif

  I_t n_nonzeros = 0;

  auto first_index = matrix._first_index;
  auto edges = matrix._edges.get();
  auto indices = matrix._indices.get();
  auto row_offset = sub_block.first_row;
  auto col_offset = sub_block.first_col;

  for (auto source_row = sub_block.first_row; source_row < sub_block.last_row; 
       ++source_row) {

    for (auto index = edges[source_row] - first_index;
         index < edges[source_row + 1] - first_index; ++index) {

      auto source_col = indices[index] - first_index;

      if (source_col < sub_block.first_col) {

        continue;

      } else if (source_col < sub_block.last_col) {

        auto col_out   = source_col - col_offset;
        auto row_out   = source_row - row_offset;

        ++n_nonzeros;

      } else {

        break;

      }

    }

  }

  return n_nonzeros;

}

/** \brief            Reallocate a matrix to be able to hold the contents of 
 *                    another matrix
 *
 *  \param[in, out]   matrix
 *                    Matrix which to reallocate
 *
 *  \param[in]        reference
 *                    Reference matrix for the reallocation
 *
 *  \param[in]        reference_sub_block
 *                    Optional: reference sub matrix specification. Default:  
 *                    use the whole of reference
 *
 *  \param[in]        location
 *                    Optional: Location on which to allocate the memory.  
 *                    Default: same as reference.
 *
 *  \param[in]        device_id
 *                    Optional: device id on which to allocate the memory.h
 *                    Default: same as reference.
 */
template <typename T, typename U>
void reallocate_like(Dense<T>& matrix, const Dense<U>& reference,
                     SubBlock sub_block, Location location, int device_id) {

  PROFILING_FUNCTION_HEADER

  matrix.reallocate(sub_block.rows(), sub_block.cols(), location, device_id);

}
/** \overload
 */
template <typename T, typename U>
void reallocate_like(Dense<T>& matrix, const Dense<U>& reference,
                     Location location, int device_id) {

  PROFILING_FUNCTION_HEADER

  matrix.reallocate(reference.rows(), reference.cols(), location, device_id);

}
/** \overload
 */
template <typename T, typename U>
void reallocate_like(Dense<T>& matrix, const Dense<U>& reference) {

  PROFILING_FUNCTION_HEADER

  matrix.reallocate(reference.rows(), reference.cols(), reference._location,
                    reference._device_id);

}
/** \overload
 */
template <typename T, typename U>
void reallocate_like(Dense<T>& matrix, const Sparse<U>& reference,
                     SubBlock sub_block, Location location, int device_id) {

  PROFILING_FUNCTION_HEADER

#ifndef LINALG_NO_CHECKS
  if ((reference._format == Format::CSR)   && 
      (sub_block.rows() > reference.rows())  ) {
    throw excBadArgument("reallocate_like(A, B, sub_block): sub_block has more "
                         "rows than B");
  } 
  if ((reference._format == Format::CSC)   &&
      (sub_block.cols() > reference.cols())  ) {
    throw excBadArgument("reallocate_like(A, B, sub_block): sub_block has more "
                         "columns than B");
  }
#endif
  matrix.reallocate(sub_block.rows(), sub_block.cols(), location, device_id);

}
/** \overload
 */
template <typename T, typename U>
void reallocate_like(Dense<T>& matrix, const Sparse<U>& reference,
                     Location location, int device_id) {

  reallocate_like(matrix, reference,
                  SubBlock(0, reference.rows(), 0, reference.cols()),
                  location, device_id);

}
/** \overload
 */
template <typename T, typename U>
void reallocate_like(Dense<T>& matrix, const Sparse<U>& reference) {

  PROFILING_FUNCTION_HEADER

  matrix.reallocate(reference.rows(), reference.cols(), reference._location, 
                    reference._device_id);

}
/** \overload
 */
template <typename T, typename U>
void reallocate_like(Sparse<T>& matrix, const Dense<U>& reference,
                     SubBlock sub_block, Location location, int device_id) {

  PROFILING_FUNCTION_HEADER

  auto n_nonzeros   = count_nonzeros(reference(sub_block));
  I_t  lines        = 0;
  if      (matrix._format == Format::CSR) lines = sub_block.rows();
  else if (matrix._format == Format::CSR) lines = sub_block.cols();
#ifndef LINALG_NO_CHECKS
  else {
    throw excUnimplemented("reallocate_like(): not implemented for given "
                           "format");
  }
#endif

  matrix.reallocate(lines, n_nonzeros, location, device_id);

}
/** \overload
 */
template <typename T, typename U>
void reallocate_like(Sparse<T>& matrix, const Dense<U>& reference,
                     Location location, int device_id) {

  reallocate_like(matrix, reference,
                  SubBlock(0, reference.rows(), 0, reference.cols()),
                  reference._location, reference._device_id);

}
/** \overload
 */
template <typename T, typename U>
void reallocate_like(Sparse<T>& matrix, const Dense<U>& reference) {

  reallocate_like(matrix, reference,
                  SubBlock(0, reference.rows(), 0, reference.cols()),
                  reference._location, reference._device_id);

}
/** \overload
 */
template <typename T, typename U>
void reallocate_like(Sparse<T>& matrix, const Sparse<U>& reference,
                     SubBlock sub_block, Location location, int device_id) {

  PROFILING_FUNCTION_HEADER

  auto n_nonzeros   = count_nonzeros(reference, sub_block);
  I_t  lines        = 0;
  if      (matrix._format == Format::CSR) lines = sub_block.rows();
  else if (matrix._format == Format::CSR) lines = sub_block.cols();
#ifndef LINALG_NO_CHECKS
  else {
    throw excUnimplemented("reallocate_like(): not implemented for given "
                           "format");
  }
#endif

  matrix.reallocate(lines, n_nonzeros, location, device_id);

}
/** \overload
 */
template <typename T, typename U>
void reallocate_like(Sparse<T>& matrix, const Sparse<U>& reference,
                     Location location, int device_id) {

  PROFILING_FUNCTION_HEADER

  matrix.reallocate(reference._size, reference._n_nonzeros, location, 
                    device_id);

}
/** \overload
 */
template <typename T, typename U>
void reallocate_like(Sparse<T>& matrix, const Sparse<U>& reference) {

  PROFILING_FUNCTION_HEADER

  matrix.reallocate(reference._size, reference._n_nonzeros, 
                    reference._location, reference._device_id);

}

/** \brief            Add a subblock of a sparse matrix to a dense matrix
 *
 *
 *  \param[in]        source
 *                    The sparse matrix to extract the subblock from.
 *
 *  \param[in]        sub_block
 *                    Submatrix specification (C-style indexing).
 *
 *  \param[out]       destination
 *                    Dense matrix to store the subblock.
 *
 *  \todo             If the full matrix is to be converted, mkl_?dnscsr and
 *                    cusparseXdense2csr could be used
 */
template <typename T>
inline void sparse2dense_host(const Sparse<T>& source, SubBlock sub_block, 
                              Dense<T>& destination) {

  PROFILING_FUNCTION_HEADER

#ifndef LINALG_NO_CHECKS
  if (destination._transposed) {
    throw excUnimplemented("sparse2dense_host(): assignment to transposed "
                           "dense matrices is not supported.");
  } else if (source._transposed) {
    throw excUnimplemented("sparse2dense_host(): assignment from transposed "
                           "sparse matrices is not supported.");
  }
  check_format(Format::CSR, source, "sparse2dense_host()");

  if ((sub_block.first_row > source._size - 1) || 
      (sub_block.last_row > source._size)         ) {
    throw excBadArgument("sparse2dense_host(): requested row range is not "
                         "contained in the source matrix.");
  }
#endif

  if (source._location == Location::host && 
      destination._location == Location::host) {

    auto rows = sub_block.last_row - sub_block.first_row;
    auto cols = sub_block.last_col - sub_block.first_col;

    // Check if size fits. If destination is empty, reallocate accordingly
    if (destination.is_empty()) {

      destination.reallocate(rows, cols);

      Fills::zero(destination);

    }
#ifndef LINALG_NO_CHECK
    else if (destination._rows != rows || destination._cols != cols) {

      throw excBadArgument("sparse2dense_host(): matrix dimension mismatch");

    }
#endif

    auto first_index = source._first_index;
    auto edges = source._edges.get();
    auto indices = source._indices.get();
    auto values = source._values.get();

    auto destination_data = destination._begin();
    auto destination_ld = destination._leading_dimension;
    auto row_offset = sub_block.first_row;
    auto col_offset = sub_block.first_col;

    for (auto source_row = sub_block.first_row; 
         source_row < sub_block.last_row; ++source_row) {

      for (auto index = edges[source_row] - first_index;
           index < edges[source_row + 1] - first_index; ++index) {

        auto source_col = indices[index] - first_index;

        if (source_col < sub_block.first_col) {

          continue;

        } else if (source_col < sub_block.last_col) {

          auto col_out   = source_col - col_offset;
          auto row_out   = source_row - row_offset;

          I_t array_pos;
          if (destination._format == Format::ColMajor) {
            array_pos = col_out * destination_ld + row_out;
          } else {
            array_pos = row_out * destination_ld + col_out;
          }

          destination_data[array_pos] += values[index];

        } else {

          break;

        }

      }

    }

  }

#ifndef LINALG_NO_CHECKS
  else {

    throw excUnimplemented("sparse2dense_host(): only matrices in main memory "
                           "are supported");

  }
#endif

}


/** \brief            Add a subblock of a sparse matrix to a dense matrix
 *
 *
 *  \param[in]        source
 *                    The sparse matrix to extract the subblock from.
 *
 *  \param[in]        first_row
 *                    Row to start extraction (included, C-style indexing).
 *
 *  \param[in]        last_row
 *                    Row to stop extraction (excluded, C-style indexing).
 *
 *  \param[in]        first_col
 *                    Colum to start extraction (included, C-style indexing).
 *
 *  \param[in]        last_col
 *                    Column to stop extraction (excluded, C-style indexing).
 *
 *  \param[out]       destination
 *                    Dense matrix to store the subblock.
 */
template <typename T>
inline void sparse2dense_host(const Sparse<T>& source, I_t first_row,
                              I_t last_row, I_t first_col, I_t last_col, 
                              Dense<T>& destination) {

  sparse2dense_host(source, SubBlock(first_row, last_row, first_col, last_col), 
                    destination);

}

} /* namespace LinAlg::Utilities */

} /* namespace LinAlg */

#endif /* LINALG_UTILITIES_FORMAT_CONVERT_H_ */
