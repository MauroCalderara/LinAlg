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
#include "../streams.h"
#include "../BLAS/copy.h"


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

  if ((sub_block.first_row < 0) ||
      (sub_block.last_row > matrix.rows())        ) {
    throw excBadArgument("count_nonzeros(): requested row range is not "
                         "contained in the source matrix.");
  } else if ((sub_block.first_col > matrix.cols() - 1) || 
             (sub_block.last_col > matrix.cols())         ) {
    throw excBadArgument("count_nonzeros(): requested col range is not "
                         "contained in the source matrix.");
  }
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

  if ((sub_block.first_row < 0) || (sub_block.last_row > matrix.rows())) {
    throw excBadArgument("count_nonzeros(): requested row range is not "
                         "contained in the source matrix.");
  } else if (matrix._minimal_index != matrix._maximal_index    &&
             ((sub_block.first_col < matrix._minimal_index) ||
              (sub_block.last_col  > matrix._maximal_index)   )  ) {
    throw excBadArgument("count_nonzeros(): requested col range is not "
                         "contained in the source matrix."); }
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

  if ((sub_block.first_row < 0) ||
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


///////////////////////
// complex <-> realimag

// Declarations of CUDA code wrappers
#ifdef HAVE_CUDA
void gpu_complex2realimag(int, int, float2*, int, float*, int, float*, int);
void gpu_complex2realimag(int, int, double2*, int, double*, int, double*, 
                          int);

void gpu_realimag2complex(int, int, float, float*, int, float*, int, float2, 
                          float2*, int);
void gpu_realimag2complex(int, int, double, double*, int, double*, int, 
                          double2, double2*, int);

void gpu_csr2dense(double*, int*, int*, int, int, double*, int);
void gpu_csr2dense_async(double*, int*, int*, int, int, double*, int, 
                         cudaStream_t);
#endif

/** \brief            Convert a complex matrix to a pair of real matrices
 *
 *  C -> R + i * I
 * 
 *  \param[in]        C
 *                    Complex matrix
 *
 *  \param[in]        R
 *                    Non-complex matrix
 *
 *  \param[in]        I
 *                    Non-complex matrix
 */
template <typename T, typename U>
inline void complex2realimag(const Dense<U>& C, Dense<T>& R, Dense<T>& I) {

  PROFILING_FUNCTION_HEADER

  if (R.is_empty()) R.reallocate_like(C);
  if (I.is_empty()) I.reallocate_like(C);

#ifndef LINALG_NO_CHECKS
  if (! (((type<U>() == Type::C) && (type<T>() == Type::S)) ||
         ((type<U>() == Type::Z) && (type<T>() == Type::D))   ) ) {
    throw excBadArgument("complex2realimage(C, R, I): template expanded "
                         "to invalid data type combination");
  }
  check_device(C, R, I, "complex2realimag(C, R, I)");
  check_same_dimensions(C, R, "complex2realimag(C, R, I), C, R");
  check_same_dimensions(C, I, "complex2realimag(C, R, I), C, I");
  check_input_transposed(C, "complex2realimag(C, R, I), C");
  check_output_transposed(R, "complex2realimag(C, R, I), R");
  check_output_transposed(I, "complex2realimag(C, R, I), I");
  check_device(C, R, "complex2realimag(C, R, I), C, R");
  check_device(C, I, "complex2realimag(C, R, I), C, I");
#endif

  auto location    = C._location;
  auto line_length = (C._format == Format::ColMajor) ? C._rows : C._cols;
  auto lines       = (C._format == Format::ColMajor) ? C._cols : C._rows;
  auto C_ptr       = C._begin();
  auto ldc         = C._leading_dimension;
  auto A_ptr       = R._begin();
  auto lda         = R._leading_dimension;
  auto B_ptr       = I._begin();
  auto ldb         = I._leading_dimension;

  if (location == Location::host) {

    for (auto line = 0; line < lines; ++line) {

      for (auto element = 0; element < line_length; ++element) {

        A_ptr[lda * line + element] = real(C_ptr[ldc * line + element]);
        B_ptr[ldb * line + element] = imag(C_ptr[ldc * line + element]);

      }

    }

  }
#ifdef HAVE_CUDA
  else if (location == Location::GPU) {

    gpu_complex2realimag(line_length, lines, C_ptr, ldc, A_ptr, lda, B_ptr, 
                         ldb);

  }
#endif

#ifndef LINALG_NO_CHECKS
  else {

    throw excUnimplemented("complex2realimag(C, R, I) not available on " 
                           "requested engine");

  }
#endif

}

/** \brief            Convert a pair of real matrices to a complex matrix
 *
 *  C = alpha * (R + i * I) + beta * C
 *
 *  \param[in]        alpha
 *                    OPTIONAL. Default: 1.0
 *
 *  \param[in]        R
 *
 *  \param[in]        I
 *
 *  \param[in]        beta
 *                    OPTIONAL: Default: 0.0
 * 
 *  \param[in]        C
 */
template <typename T, typename U>
inline void realimag2complex(T alpha, const Dense<T>& R, const Dense<T>& I,
                             U beta, Dense<U>& C) {

  PROFILING_FUNCTION_HEADER

  if (alpha == cast<T>(0.0)) return;
  if (C.is_empty()) C.reallocate_like(R);

#ifndef LINALG_NO_CHECKS
  if (! (((type<U>() == Type::C) && (type<T>() == Type::S)) ||
         ((type<U>() == Type::Z) && (type<T>() == Type::D))   ) ) {
    throw excBadArgument("realimag2complex(alpha, R, I, beta, C): template "
                         "expanded to invalid data type combination");
  }
  check_device(R, I, C, "realimag2complex(alpha, R, I, beta, C)");
  check_same_dimensions(R, I, "realimag2complex(alpha, R, I, beta, C), R, I");
  check_same_dimensions(R, C, "realimag2complex(alpha, R, I, beta, C), R, C");
  check_input_transposed(R, "realimag2complex(alpha, R, I, beta, C), R");
  check_input_transposed(I, "realimag2complex(alpha, R, I, beta, C), I");
  check_output_transposed(C, "realimag2complex(alpha, R, I, beta, C), C");
  check_device(R, I, "realimag2complex(alpha, R, I, beta, C), R, I");
  check_device(R, C, "realimag2complex(alpha, R, I, beta, C), R, C");
#endif

  auto location    = R._location;
  auto line_length = (R._format == Format::ColMajor) ? R._rows : R._cols;
  auto lines       = (R._format == Format::ColMajor) ? R._cols : R._rows;
  auto A_ptr       = R._begin();
  auto lda         = R._leading_dimension;
  auto B_ptr       = I._begin();
  auto ldb         = I._leading_dimension;
  auto C_ptr       = C._begin();
  auto ldc         = C._leading_dimension;

  if (location == Location::host) {

    U rivalue;
    U cvalue;

    for (auto line = 0; line < lines; ++line) {

      for (auto element = 0; element < line_length; ++element) {

        rivalue = cast<U>(A_ptr[lda * line + element],
                          B_ptr[lda * line + element]);
        cvalue  = C_ptr[ldc * line + element];

        C_ptr[ldc * line + element] = cast<U>(alpha) * rivalue + beta * cvalue;

      }

    }

  }
#ifdef HAVE_CUDA
  else if (location == Location::GPU) {

    gpu_realimag2complex(line_length, lines, alpha, A_ptr, lda, B_ptr, ldb, 
                         beta, C_ptr, ldc);

  }
#endif

#ifndef LINALG_NO_CHECKS
  else {

    throw excUnimplemented("realimag2complex(alpha, R, I, beta, C) not "
                           "available on requested engine");

  }
#endif

}
/** \overload
 */
template <typename T, typename U>
inline void realimag2complex(const Dense<T>& R, const Dense<T>& I,
                             Dense<U>& C) {
  realimag2complex(cast<T>(1.0), R, I, cast<U>(0.0), C);
}

#ifdef HAVE_CUDA
/** \brief            Routine to convert a CSR matrix to a dense matrix in 
 *                    Format::ColMajor on a GPU
 *
 *  \param[in]        sparse
 *
 *  \param[out]       dense
 *
 *  \param[in]        zero_first
 */
//template <typename T>
inline void csr2dense_gpu(const Sparse<D_t>& sparse, Dense<D_t>& dense,
                          bool zero_first = false) {

  PROFILING_FUNCTION_HEADER

  // TODO:
  // make some checks

  if (zero_first) LinAlg::Fills::zero(dense);

  auto values_dev        = sparse._values.get();
  auto indices_dev       = sparse._indices.get();
  auto edges_dev         = sparse._edges.get();
  auto rows              = sparse.rows();
  auto first_index       = sparse._first_index;
  auto dense_dev         = dense._begin();
  auto leading_dimension = dense._leading_dimension;

  gpu_csr2dense(values_dev, indices_dev, edges_dev, rows, first_index, 
                dense_dev, leading_dimension);


}

/** \brief            Routine to asynchronously convert a CSR matrix to a 
 *                    dense matrix in Format::ColMajor on a GPU
 *
 *  \param[in]        sparse
 *
 *  \param[out]       dense
 *
 *  \param[in]        zero_first
 */
//template <typename T>
inline I_t csr2dense_gpu_async(const Sparse<D_t>& sparse, Dense<D_t>& dense,
                                Stream& stream, bool zero_first = false) {

  PROFILING_FUNCTION_HEADER

  // TODO:
  // make some checks

  if (zero_first) LinAlg::Fills::zero(dense);

  auto values_dev        = sparse._values.get();
  auto indices_dev       = sparse._indices.get();
  auto edges_dev         = sparse._edges.get();
  auto rows              = sparse.rows();
  auto first_index       = sparse._first_index;
  auto dense_dev         = dense._begin();
  auto leading_dimension = dense._leading_dimension;
  auto cuda_stream       = stream.cuda_stream;

  gpu_csr2dense_async(values_dev, indices_dev, edges_dev, rows, first_index, 
                      dense_dev, leading_dimension, cuda_stream);


  return 0;

}
#endif /* HAVE_CUDA */

} /* namespace LinAlg::Utilities */

} /* namespace LinAlg */

#endif /* LINALG_UTILITIES_FORMAT_CONVERT_H_ */
