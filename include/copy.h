/** \file
 *
 *  \brief            General matrix copy
 *
 *  \date             Created:  Dec 10, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_COPY_H_
#define LINALG_COPY_H_

#include <tuple>
#include <cassert>

#include "preprocessor.h"

#include "types.h"
#include "profiling.h"
#include "streams.h"
#include "utilities/checks.h"
#include "utilities/copy_array.h"
#include "utilities/format_convert.h"
#include "BLAS/cusparse/csr2dense.h"

namespace LinAlg {

/** \brief            General matrix copy
 *
 *  \param[in]        source
 *
 *  \param[in]        source_sub_block
 *                    Sub block of the source to be copied
 *
 *  \param[in,out]    destination
 */
template <typename T>
inline void copy(const Dense<T>& source, Dense<T>& destination) {

  PROFILING_FUNCTION_HEADER

  if (source.is_empty()) return;

  if (destination.is_empty()) destination.reallocate_like(source); 

  using Utilities::copy_2Darray;

  copy_2Darray(source._transposed, source._format, source._memory.get(),
               source._leading_dimension, source._location, source._device_id, 
               source._rows, source._cols, destination._format, 
               destination._memory.get(), destination._leading_dimension, 
               destination._location, destination._device_id);

}
/** \overload
 */
template <typename T>
inline void copy(const Dense<T>& source, Sparse<T>& destination) {

  throw excUnimplemented("copy(A, B): dense to sparse copy currently "
                         "unimplemented");

}
/** \overload
 */
template <typename T>
void copy(const Sparse<T>& source, Dense<T>& destination) {

  // This routine mostly handles the GPU case and delegates to the CPU 
  // sub_blocked version for the CPU case

  PROFILING_FUNCTION_HEADER

  if (source.is_empty())      return;

  // On the CPU, the sub_block variant is more general so we perform all 
  // checking there
  if (source._location == Location::host && 
      destination._location == Location::host) {

    using std::tie;

    I_t first_index, last_index;
    tie(first_index, last_index) = source.find_extent(0, source._size);

    if (source._format == Format::CSR) {

      SubBlock complete_matrix(0, source._size, first_index, last_index);
      copy(source, complete_matrix, destination);

    } else if (source._format == Format::CSC) {
    
      SubBlock complete_matrix(first_index, last_index, 0, source._size);
      copy(source, complete_matrix, destination);
    
    }
#ifndef LINALG_NO_CHECKS
    else {
      throw excUnimplemented("copy(A, B), A: unsupported sparse format");
    }
#endif

    return;
  
  }


#ifndef LINALG_NO_CHECKS
  // All sorts of transpositions could be done by interpreting the source as 
  // CSC, but most of that is currently unimplemented
  Utilities::check_input_transposed(source, "copy(A, B), A");
  Utilities::check_output_transposed(destination, "copy(A, B), B");
  Utilities::check_format(Format::CSR, source, "copy(A, B), A");
  Utilities::check_format(Format::ColMajor, destination, "copy(A, B), B");
#endif

  if (destination.is_empty()) destination.reallocate_like(source);

#ifdef LINALG_NO_CHECKS
  if (source._size != destination._rows) {
    throw excBadArgument("copy(A, B): number of rows don't match (A.rows = %d, "
                         "B.rows = %d)", source._size, destination._rows);
  }
# ifdef HAVE_CUDA
  if (source._location == Location::GPU      || 
      destination._location == Location::GPU   ) {

    Utilities::check_gpu_handles("copy(A, B)");

  }
# endif
#endif

  if (source._location == Location::host && 
      destination._location == Location::host) {
  
    assert(false);
    // This can't happen (we checked for that case above), this if-clause is 
    // just to allow a trailing else below

  }
#ifdef HAVE_CUDA
  else if (source._location == Location::GPU &&
           destination._location == Location::GPU) {
  
    using LinAlg::BLAS::cuSPARSE::xcsr2dense;
    xcsr2dense(source, destination);
  
  }
  else if ((source._location == Location::host    &&
            destination._location == Location::GPU  ) ||
           (source._location == Location::GPU      &&
            destination._location == Location::host  )  ) {
  
    // Create a sparse temporary on the destination, then convert to dense
    Sparse<T> tmp;
    tmp.reallocate_like(source, destination._location, destination._device_id);

    copy(source, tmp);        // changing the location: copy(sparse,sparse)
    copy(tmp, destination);   // changing the format: this function
  
  }
#endif /* HAVE_CUDA */
#ifndef LINALG_NO_CHECKS
  else {
  
    throw excUnimplemented("copy(A, B): not supported on selected location");

  }
#endif

}
/** \overload
 */
template <typename T>
inline void copy(const Sparse<T>& source, Sparse<T>& destination) {

  PROFILING_FUNCTION_HEADER

  if (source.is_empty()) return;

  if (destination.is_empty()) destination.reallocate_like(source);

#ifndef LINALG_NO_CHECKS
  Utilities::check_same_dimensions(source, destination, "copy(A, B)");
#endif

  // Copy the data
  using Utilities::copy_1Darray;

  copy_1Darray(source._values.get(), source._n_nonzeros, 
               destination._values.get(), 
               source._location, source._device_id, 
               destination._location, destination._device_id);
  copy_1Darray(source._indices.get(), source._n_nonzeros, 
               destination._indices.get(), 
               source._location, source._device_id, 
               destination._location, destination._device_id);
  copy_1Darray(source._edges.get(), source._size + 1, 
               destination._edges.get(), 
               source._location, source._device_id, 
               destination._location, destination._device_id);

  // Update the target matrix
  destination.first_index(source._first_index);
  destination._properties = source._properties;
  destination._minimal_index = source._minimal_index;
  destination._maximal_index = source._maximal_index;

}

/** \brief            General sub matrix copy
 *
 *  \param[in]        source
 *                    The matrix from which to copy the data.
 *
 *  \param[in]        source_sub_block
 *                    Source submatrix specification (C-style indexing).
 *
 *  \param[in,out]    destination
 *                    The matrix to which to copy the data.
 */
template <typename T>
inline void copy(const Dense<T>& source, SubBlock source_sub_block,
                 Dense<T>& destination) {

  if (source_sub_block.is_empty()) return;

  copy(source(source_sub_block), destination);

}
/** \overload
 */
template <typename T>
inline void copy(const Dense<T>& source, SubBlock source_sub_block,
                 Sparse<T>& destination) {
  throw excUnimplemented("copy(A, sub_block, B): subDense -> Sparse currently "
                         "not implemented");
}
/** \overload
 */
template <typename T>
inline void copy(const Sparse<T>& source, SubBlock source_sub_block,
                 Dense<T>& destination) {

  PROFILING_FUNCTION_HEADER

  if (source_sub_block.is_empty()) return;

  auto rows = source_sub_block.last_row - source_sub_block.first_row;
  auto cols = source_sub_block.last_col - source_sub_block.first_col;

#ifndef LINALG_NO_CHECKS

# ifdef HAVE_CUDA
  if (source._location == Location::GPU) {
    // Would need a submatrix extraction routine (sparse2dense_X) and a extent 
    // computation routine (find_extent) for the GPU
    throw excUnimplemented("copy(A, sub_block, B): submatrix extraction is "
                           "not supported on the GPU");
  } else if (destination._location == Location::GPU) {
    Utilities::check_gpu_handles("copy(A, sub_block, B)");
  }
# endif

  // All sorts of transpositions could be done by interpreting the source as 
  // CSC, but most of that is currently unimplemented
  Utilities::check_input_transposed(source, "copy(A, sub_block, B), A");
  Utilities::check_output_transposed(destination, "copy(A, sub_block, B), B");
  Utilities::check_format(Format::CSR, source, "copy(A, sub_block, B), A");
  Utilities::check_format(Format::ColMajor, destination, "copy(A, sub_block, "
                                                          "B), B");

  if (source_sub_block.last_row > source._size ||
      source_sub_block.first_row < 0             ) {
    throw excBadArgument("copy(A, sub_block, B): stop row not contained in "
                         "source matrix");
  }

  if (!destination.is_empty()) {
    if (rows != destination.rows() || cols != destination.cols()) {
      throw excBadArgument("copy(A, sub_block, B): dimensions of sub_block and "
                           "B don't match (sub_block = %dx%d, destination = "
                           "%dx%d)", rows, cols, destination.rows(), 
                           destination.cols());
    }
  }
#endif

  if (destination.is_empty()) destination.reallocate(rows, cols);

  if (source._location == Location::host && 
      destination._location == Location::host) {
  
    Fills::zero(destination);
    Utilities::sparse2dense_host(source, source_sub_block, destination);

  }

#ifdef HAVE_CUDA
  else if (source._location == Location::host &&
           destination._location == Location::GPU) {
  
    Sparse<T> tmp;
    copy(source, source_sub_block, tmp);    // copy(sparse, sub_block, sparse)
    copy(tmp, destination);                 // copy(sparse, dense)
  
  }
#endif

#ifndef LINALG_NO_CHECKS
  else {
    throw excUnimplemented("copy(A, sub_block, B): function not implemented "
                           "for requested location");
  }
#endif

}
/** \overload
 */
template <typename T>
inline void copy(const Sparse<T>& source, SubBlock source_sub_block,
                 Sparse<T>& destination) {

  PROFILING_FUNCTION_HEADER

  if (source_sub_block.is_empty()) return;

#ifndef LINALG_NO_CHECKS
  if (source._location != Location::host) {
    throw excUnimplemented("copy(A, sub_block, B): submatrix extraction "
                           "currently only implemented for source matrices in "
                           "main memory.");
  }

  Utilities::check_format(Format::CSR, source, "Sparse.copy_from()");

  if ((source_sub_block.first_row > source._size - 1) ||
      (source_sub_block.last_row > source._size)         ) {
    throw excBadArgument("copy(A, B): requested submatrix row range is not "
                         "contained in the source matrix.");
  }
#endif

  if (destination.is_empty()) {
  
    using Utilities::count_nonzeros;
  
    auto source_sub_size = source_sub_block.last_row - 
                           source_sub_block.first_row;
    auto n_nonzeros      = count_nonzeros(source, source_sub_block);

    destination.reallocate(source_sub_size, n_nonzeros, source._location, 
                           source._device_id);
  
  } 
#ifndef LINALG_NO_CHECKS
  else {

    using Utilities::count_nonzeros;
  
    auto source_sub_size = source_sub_block.last_row - 
                           source_sub_block.first_row;
    auto n_nonzeros      = count_nonzeros(source, source_sub_block);

    if (n_nonzeros != destination._n_nonzeros) {

      throw excBadArgument("copy(A, B): matrix B has different number of "
                           "non_zero elements than requested submatrix of "
                           "source A");

    } else if (source_sub_size != destination._size) {
    
      throw excBadArgument("copy(A, B): matrix sizes differ (submatrix "
                           "size = %d, B._size = %d", source_sub_size,
                           destination._size);

    }

  
  }
#endif

  auto source_first_index = source._first_index;
  auto source_edges       = source._edges.get();
  auto source_indices     = source._indices.get();
  auto source_values      = source._values.get();

  auto first_index        = destination._first_index;
  auto edges              = destination._edges.get();
  auto indices            = destination._indices.get();
  auto values             = destination._values.get();
  auto row_offset         = source_sub_block.first_row;
  auto col_offset         = source_sub_block.first_col;
  I_t  empty_rows         = 0;
  I_t  index              = destination._first_index;

  for (auto source_row = source_sub_block.first_row;
       source_row < source_sub_block.last_row; ++source_row) {

    bool current_row_empty = true;

    for (auto source_index = source_edges[source_row] - source_first_index;
         source_index < source_edges[source_row + 1] - source_first_index; 
         ++source_index) {

      auto source_col = source_indices[source_index] - source_first_index;

      if (source_col < source_sub_block.first_col) {

        continue;

      } else if (source_col < source_sub_block.last_col) {

        values[index]  = source_values[source_index];
        indices[index] = source_col - col_offset;

        auto row = source_row - row_offset; // C-style indexing
        assert(empty_rows < row + 1);
        for (I_t previous_row = row - empty_rows; previous_row < row; 
             ++previous_row) {
        
          edges[previous_row] = index; // includes first_index
        
        }

        if (current_row_empty) {
          
          edges[row] = index;
          current_row_empty = false;

        }

        ++index;

      } else {

        break;

      }

    }

    empty_rows = (current_row_empty) ? empty_rows + 1 : 0;

  }

#ifdef HAVE_MPI
  destination._row_offset = source._row_offset + row_offset;
#endif 

  // Update the target matrix

  if (destination._format == Format::CSR) {

    destination._minimal_index = source_sub_block.first_col;
    destination._maximal_index = source_sub_block.last_col;

  } else if (destination._format == Format::CSC) {

    destination._minimal_index = source_sub_block.first_row;
    destination._maximal_index = source_sub_block.last_row;

  } else {
  
    assert(false);
  
  }

  destination._properties = source._properties;

}

/** \brief            General asynchronous matrix copy
 *
 *  \param[in]        source
 *
 *  \param[in,out]    destination
 *
 *  \param[in]        stream
 *
 *  \returns          Ticket number in stream (or 0 if the stream doesn't
 *                    allow syncing with tickets)
 */

} /* namespace LinAlg */

#endif /* LINALG_COPY_H_ */
