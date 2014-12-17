/** \file
 *
 *  \brief            Sparse matrix struct (Sparse<T>)
 *
 *  \date             Created:  Jul 11, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_SPARSE_H_
#define LINALG_SPARSE_H_

#include <memory>     // std::shared_ptr
#include <cassert>    // assert
#include <tuple>      // std::tuple, std::tie
#include <limits>     // std::numeric_limits

#include "preprocessor.h"

#ifdef HAVE_CUDA
# include <cuda_runtime.h> // various CUDA routines
# include <cusparse_v2.h>
# include "CUDA/cuda_memory_allocation.h"    // CUDA::cuda_make_shared
#endif

#include "types.h"
#include "profiling.h"
#include "matrix.h"
#include "exceptions.h"

// Forward declaration of templated utilities that are used in the constructors 
// and templated members of Sparse<T>
#include "forward.h"


namespace LinAlg {

/** \brief            Sparse matrix struct
 */
template <typename T>
struct Sparse : Matrix {

  // Construct empty
  Sparse();

  // Destructor.
  ~Sparse();

  // See dense.h for an argument why we don't provide assignment and copy
  // constructors
  Sparse(Sparse&& other);
#ifndef DOXYGEN_SKIP
  Sparse(Sparse& other) = delete;
  Sparse(const Sparse& other) = delete;
  Sparse& operator=(Sparse other) = delete;
#endif

  // Construct by allocating memory
  Sparse(I_t size, I_t n_nonzeros, int first_index = 1,
         Location location = Location::host, int device_id = 0,
         Format format = Format::CSR);

  // Construct from existing CSR or CSC array triplet
  Sparse(I_t size, I_t n_nonzeros, T* values, I_t* indices, I_t* edges,
         int first_index = 1, Location location = Location::host,
         int device_id = 0, Format format = Format::CSR);

  // Common Matrix operations

  // Explicitly clone
  inline void clone_from(const Sparse<T>& source);

  // Explicitly move
  inline void move_to(Sparse<T>& destination);

  // Allocate new memory (no memory is copied)
  inline void reallocate(I_t size, I_t n_nonzeros,
                         Location location = Location::host, int device_id = 0);

  // Allocate new memory according to the size of another matrix (no memory is 
  // copied)
  template <typename U>
  inline void reallocate_like(const U& other, SubBlock sub_block,
                              Location location, int device_id = 0);
  template <typename U>
  inline void reallocate_like(const U& other, Location location, int device_id);
  template <typename U>
  inline void reallocate_like(const U& other);

  // Explicitly copy
  inline void copy_from(const Sparse<T>& source, SubBlock sub_block);
  inline void copy_from(const Dense<T>& source, SubBlock sub_block);
  inline void copy_from(const Sparse<T>& source, I_t first_row, I_t last_row,
                        I_t first_col, I_t last_col);
  inline void copy_from(const Dense<T>& source, I_t first_row, I_t last_row,
                        I_t first_col, I_t last_col);
  inline void copy_from(const Sparse<T>& source);
  inline void copy_from(const Dense<T>& source);
  inline void operator<<(const Sparse<T>& source);
  inline void operator<<(const Dense<T>& source);

  // Should have an asynchronous variant as well
  // int copy_async_from(const Sparse<T>& source, SubBlock sub_block, Stream& 
  // stream); ...

  /// Mark the matrix as transposed
  inline void transpose() { _transposed = !_transposed; }

  // Return the number of rows in the matrix
  inline I_t rows() const;
  // Return the number of columns in the matrix
  inline I_t cols() const;

  /// Return the current location of the matrix
  inline Location location() const { return _location; }
  void location(Location new_location, int device_id = 0);

  // Free all memory, set matrix to empty
  void unlink();

  /// Return the storage format
  inline Format format() const { return _format; }

  // Return true if matrix is on host
  inline bool is_on_host() const;
  // Return true if matrix is on GPU
  inline bool is_on_GPU()  const;
  // Return true if matrix is on MIC
  inline bool is_on_MIC()  const;

  // Get properties
  inline bool is(Property property) const { return (_properties & property); }

  // Set properties
  inline void set(Property property);

  // Unset properties
  inline void unset(Property property);

  // Returns true if matrix is empty
  inline bool is_empty() const { return (_size == 0); }


  // Sparse specific operations

  /// Return index of first element in arrays
  inline int first_index() const { return _first_index; }
  // Set index of first element in arrays
  inline void first_index(int new_first_index);

  // Return the indexed width within a certain row or column range
  std::tuple<I_t, I_t> find_extent(const I_t first_line,
                                   const I_t last_line) const;

  // Update the extent
  inline void update_extent();


#ifndef DOXYGEN_SKIP
  ///////////////////////////////////////////////////////////////////////////
  // Not private: but the _ indicates that the idea is to not mess with these
  // directly unless neccessary.

  // Number of non zero elements
  I_t _n_nonzeros;

  // Number of rows for CSR, number of columns for CSC
  I_t _size;

  // Shared pointers used for the memory containing the matrix
  std::shared_ptr<T> _values;
  std::shared_ptr<I_t> _indices;
  std::shared_ptr<I_t> _edges;
  int _first_index;

  // Location of the matrix
  Location _location;
  int _device_id;

  // Transposition and formatting.
  bool _transposed;
  Format _format;

  // Whether the matrix is complex or not. Overridden in sparse.cc with
  // specializations for C_t and Z_t:
  inline bool _is_complex() const { return false; }

#ifdef HAVE_CUDA
  // The CUSPARSE matrix descriptor
  cusparseMatDescr_t _cusparse_descriptor;
#endif

#ifdef HAVE_MPI
  // For matrix partitions: the line number of the first line as seen within
  // the global (unpartitioned) matrix
  I_t _row_offset;
#endif

  // Properties of the matrix
  unsigned char _properties;

  // Optional width of the matrix (columns for CSR, rows for CSC matrices), 
  // C-style indexing
  I_t _minimal_index;
  I_t _maximal_index;

#endif /* DOXYGEN_SKIP */

};

/** \brief              Empty constructor
 *
 *  This constructor doesn't allocate any memory. Typically empty matrices are
 *  initialized suitably by operations that have output parameters.
 */
template <typename T>
Sparse<T>::Sparse()
                : _n_nonzeros(0),
                  _size(0),
                  _values(nullptr),
                  _indices(nullptr),
                  _edges(nullptr),
                  _first_index(0),
                  _location(Location::host),
                  _device_id(0),
                  _transposed(false),
                  _format(Format::CSR),
#ifdef HAVE_MPI
                  _row_offset(0),
#endif
                  _properties(0x0),
                  _minimal_index(0),
                  _maximal_index(0) {

  PROFILING_FUNCTION_HEADER

}

template <typename T>
Sparse<T>::~Sparse() {

  PROFILING_FUNCTION_HEADER

#ifdef HAVE_CUDA
  if (_location == Location::GPU) {
    checkCUSPARSE(cusparseDestroyMatDescr(_cusparse_descriptor));
  }
#endif

  unlink();

}


/** \brief              Move constructor
 */
template <typename T>
Sparse<T>::Sparse(Sparse&& other) : Sparse() {

  PROFILING_FUNCTION_HEADER

  // Default initialize (already happened at this point) and swap
  swap(*this, other);

}

/** \brief            Swap function
 *
 *  Swaps by swapping all data members. Allows for simple move constructor
 *  and assignment operator.
 *
 *  \param[in,out]    first
 *                    Pointer to 'first', will later point to what was
 *                    'second' before the function call.
 *
 *  \param[in,out]    second
 *                    Pointer to 'second', will later point to what was
 *                    'first' before the function call.
 *
 *  \note             We can't use std::swap to swap to instances directly
 *                    as it uses the assignment operator internally itself.
 */
template <typename U>
void swap(Sparse<U>& first, Sparse<U>& second) {

  PROFILING_FUNCTION_HEADER

  using std::swap;
  swap(first._n_nonzeros, second._n_nonzeros);
  swap(first._size, second._size);
  swap(first._values, second._values);
  swap(first._indices, second._indices);
  swap(first._edges, second._edges);
  swap(first._first_index, second._first_index);
  swap(first._location, second._location);
  swap(first._device_id, second._device_id);
  swap(first._transposed, second._transposed);
  swap(first._format, second._format);
#ifdef HAVE_CUDA
  swap(first._cusparse_descriptor, second._cusparse_descriptor);
#endif
#ifdef HAVE_MPI
  swap(first._row_offset, second._row_offset);
#endif
  swap(first._properties, second._properties);
  swap(first._minimal_index, second._minimal_index);
  swap(first._maximal_index, second._maximal_index);

}


/** \brief              Allocating-only constructor
 *
 *  This constructor allocates some memory area
 *
 *  \param[in]          size
 *                      Number of rows for CSR, number of columns for CSC
 *                      matrices.
 *
 *  \param[in]          n_nonzeros
 *                      Number of non zero elements.
 *
 *  \param[in]          first_index
 *                      OPTIONAL: Indexing for the elements. 0 for C-style
 *                      indexing, 1 for FORTRAN-style indexing. Default: 0.
 *
 *  \param[in]          location
 *                      OPTIONAL: Memory / device on which to allocate the
 *                      memory. Default: Location::host.
 *
 *  \param[in]          device_id
 *                      OPTIONAL: The number of the device on which to allocate
 *                      the memory. Ignored for Location::host. Default: 0.
 *
 *  \param[in]          format
 *                      OPTIONAL: The format for the matrix (Format::CSR or
 *                      Format::CSC).  Default: Format::CSR.
 */
template <typename T>
Sparse<T>::Sparse(I_t size, I_t n_nonzeros, int first_index, Location location,
                  int device_id, Format format)
              : _n_nonzeros(n_nonzeros),
                _size(size),
                _first_index(first_index),
                _location(location),
                _device_id(device_id),
                _transposed(false),
                _format(format),
                _properties(0x0),
                _minimal_index(0),
                _maximal_index(0) {

  PROFILING_FUNCTION_HEADER

#ifndef LINALG_NO_CHECKS
  if (format != Format::CSR && format != Format::CSC) {

    throw excBadArgument("Sparse(): can only construct Format::{CSR|CSC} "
                         "matrices.");

  }

  if (_first_index != 0 && _first_index != 1) {

    throw excBadArgument("Sparse(): invalid value for first_index specified "
                         "(must be zero or one)");

  }
#endif

#ifdef HAVE_CUDA
  if (_location == Location::GPU) {

    checkCUSPARSE(cusparseCreateMatDescr(&_cusparse_descriptor));
    if (_first_index == 1) {
      checkCUSPARSE(cusparseSetMatIndexBase(_cusparse_descriptor, \
                                            CUSPARSE_INDEX_BASE_ONE));
    }

  }
#endif

  reallocate(_size, _n_nonzeros, _location, _device_id);

#ifdef HAVE_MPI
  _row_offset = 0;
#endif

}

/** \brief              Constructor from 3 arrays
 *
 *  This constructor allows to create a matrix from preexisting data. No memory
 *  is allocated and no data is copied. Memory has to be allocated and
 *  deallocated by the user.
 *
 *  \param[in]          size
 *                      The number of rows for a CSR matrix, the number of
 *                      columns for a CSC matrix (also: the length of the
 *                      edges array minus one)
 *
 *  \param[in]          n_nonzeros
 *                      Number of non zero entries in the original matrix
 *                      (also: the minimal length of the values array).
 *
 *  \param[in]          values
 *                      The array containing the non zero values of the
 *                      matrix.
 *
 *  \param[in]          indices
 *                      The array containing the row (CSR) or column (CSC)
 *                      indices of the non zero elements.
 *
 *  \param[in]          edges
 *                      The array containing indices at which new rows (CSR)
 *                      or columns (CSC) start.
 *
 *  \param[in]          first_index
 *                      OPTIONAL: The index used to denote the first element
 *                      of an array (0 for C style indexing, 1 for FORTRAN
 *                      style indexing). Default: 0.
 *
 *  \param[in]          location
 *                      OPTIONAL: The location of the three arrays. Default:
 *                      Location::host.
 *
 *  \param[in]          device_id
 *                      OPTIONAL: The id of the device that stores the three
 *                      arrays.  Default: 0.
 *
 *  \param[in]          format
 *                      OPTIONAL: The storage format of the input array.
 *                      Default: Format::CSR.
 */
template <typename T>
Sparse<T>::Sparse(I_t size, I_t n_nonzeros, T* values, I_t* indices, I_t* edges,
                  int first_index, Location location, int device_id,
                  Format format)
                : _n_nonzeros(n_nonzeros),
                  _size(size),
                  _first_index(first_index),
                  _location(location),
                  _device_id(device_id),
                  _transposed(false),
                  _format(format),
                  _properties(0x0),
                  _minimal_index(0),
                  _maximal_index(0) {

  PROFILING_FUNCTION_HEADER

#ifndef LINALG_NO_CHECKS
  if (format != Format::CSR && format != Format::CSC) {

    throw excBadArgument("Sparse(): can only construct Format::{CSR|CSC} "
                         "matrices.");

  }

  if (_first_index != 0 && _first_index != 1) {

    throw excBadArgument("Sparse(): invalid value for first_index specified "
                         "(must be zero or one)");

  }
#endif

  // Create shared_ptrs that will not deallocate upon destruction (this is
  // common to all storage locations supported so far).
  _values  = std::shared_ptr<T>(values, [](T* values){});
  _indices = std::shared_ptr<I_t>(indices, [](I_t* indices){});
  _edges   = std::shared_ptr<I_t>(edges, [](I_t* edges){});


#ifdef HAVE_CUDA
  if (location == Location::GPU) {

    checkCUSPARSE(cusparseCreateMatDescr(&_cusparse_descriptor));
    if (_first_index == 1) {
      checkCUSPARSE(cusparseSetMatIndexBase(_cusparse_descriptor, \
                                            CUSPARSE_INDEX_BASE_ONE));
    }

  }
#endif

#ifdef HAVE_MPI
  _row_offset = 0;
#endif

}

/** \brief            Cloning from an existing matrix
 *
 * Creates another instance of Sparse<T> with the exact same parameters. No
 * memory is copied.
 *
 *  \param[in]        source
 *                    The matrix to clone from
 */
template <typename T>
inline void Sparse<T>::clone_from(const Sparse<T>& source) {

  PROFILING_FUNCTION_HEADER

  _n_nonzeros          = source._n_nonzeros;
  _size                = source._size;
  _values              = source._values;
  _indices             = source._indices;
  _edges               = source._edges;
  _first_index         = source._first_index;
  _location            = source._location;
  _device_id           = source._device_id;
  _transposed          = source._transposed;
  _format              = source._format;
#ifdef HAVE_CUDA
  _cusparse_descriptor = source._cusparse_descriptor;
#endif
#ifdef HAVE_MPI
  _row_offset          = source._row_offset;
#endif
  _properties          = source._properties;
  _minimal_index       = source._minimal_index;
  _maximal_index       = source._maximal_index;

}

/** \brief              Move matrix to another matrix
 *
 *  'Moves' an instance of Sparse<T> to another instance. No memory is copied,
 *  the instance which is moved to destination is left empty (unlinked) after
 *  the operation.
 *
 *  \param[in]          destination
 *                      The destination matrix to move the instance to
 */
template <typename T>
inline void Sparse<T>::move_to(Sparse<T>& destination) {

  PROFILING_FUNCTION_HEADER

  destination.clone_from(*this);
  unlink();

}

/** \brief            Allocates new empty memory for an already constructed
 *                    matrix.
 *
 *  \param[in]        size
 *                    Number of rows (for CSR) or columns (for CSC).
 *
 *  \param[in]        n_nonzeros
 *                    Number of non-zero elements.
 *
 *  \param[in]        location
 *                    OPTIONAL: Memory / device on which to allocate the
 *                    memory. Default: Location::host.
 *
 *  \param[in]        device_id
 *                    OPTIONAL: The number of the device on which to
 *                    allocate the memory. Ignored for Location::host.
 *                    Default: 0.
 *
 */
template <typename T>
inline void Sparse<T>::reallocate(I_t size, I_t n_nonzeros, Location location,
                                  int device_id) {

  PROFILING_FUNCTION_HEADER

#ifndef LINALG_NO_CHECKS
  if (size < 0 || n_nonzeros < 0) {
  
    throw excBadArgument("Sparse.reallocate(): size and n_nonzeros must be "
                         "non-negative.");
  
  }
#endif

  // Allocate new memory
  if (location == Location::host) {

    using Utilities::host_make_shared;
    _values  = host_make_shared<T>(n_nonzeros);
    _indices = host_make_shared<I_t>(n_nonzeros);
    _edges   = host_make_shared<I_t>(size + 1);

  }
#ifdef HAVE_CUDA
  else if (location == Location::GPU) {

    int prev_device;
    checkCUDA(cudaGetDevice(&prev_device));

    using CUDA::cuda_make_shared;
    checkCUDA(cudaSetDevice(device_id));
    _values  = cuda_make_shared<T>(n_nonzeros, device_id);
    _indices = cuda_make_shared<I_t>(n_nonzeros, device_id);
    _edges   = cuda_make_shared<I_t>(size + 1, device_id);
    checkCUDA(cudaSetDevice(prev_device));

    if (_location != Location::GPU) {

      checkCUSPARSE(cusparseCreateMatDescr(&_cusparse_descriptor));
      if (_first_index == 1) {
        checkCUSPARSE(cusparseSetMatIndexBase(_cusparse_descriptor, \
                                              CUSPARSE_INDEX_BASE_ONE));
      }

    }

  }
#endif
#ifdef HAVE_MIC
  else if (location == Location::MIC) {

    using Utilities::MIC::mic_make_shared;
    _values  = mic_make_shared<T>(n_nonzeros, _device_id);
    _indices = mic_make_shared<I_t>(n_nonzeros, _device_id);
    _edges   = mic_make_shared<I_t>(size + 1, _device_id);

  }
#endif

#ifdef HAVE_CUDA
  if (location != Location::GPU && _location == Location::GPU) {
  
    checkCUSPARSE(cusparseDestroyMatDescr(_cusparse_descriptor));

  }
#endif

  _location = location;
  _n_nonzeros = n_nonzeros;
  // Reinitialize values
  _transposed = false;
#ifdef HAVE_MPI
  _row_offset = 0;
#endif
  _properties = 0x0;
  _minimal_index = 0;
  _maximal_index = 0;
  // "Atomization point"
  _size = size;

}

/** \brief            Allocates new empty memory with the same dimensions,
 *                    transposition status and optionally the same location as 
 *                    a given matrix
 *
 *  \param[in]        other
 *                    Other matrix whose size and transposition status will be 
 *                    used for this allocation.
 *
 *  \param[in]        sub_block
 *                    OPTIONAL: Sub matrix specification for 'other'
 *
 *  \param[in]        location
 *                    OPTIONAL: Memory / device on which to allocate the
 *                    memory. Default: Location::host.
 *
 *  \param[in]        device_id
 *                    OPTIONAL: The number of the device on which to
 *                    allocate the memory. Ignored for Location::host.
 *                    Default: 0.
 */
template <typename T>
template <typename U>
inline void Sparse<T>::reallocate_like(const U& other, SubBlock sub_block, 
                                       Location location, int device_id) {

  Utilities::reallocate_like(*this, other, sub_block, location, device_id);

}
/** \overload
 */
template <typename T>
template <typename U>
inline void Sparse<T>::reallocate_like(const U& other, Location location,
                                       int device_id) {

  Utilities::reallocate_like(*this, other, location, device_id);

}
/** \overload
 */
template <typename T>
template <typename U>
inline void Sparse<T>::reallocate_like(const U& other) {

  Utilities::reallocate_like(*this, other);

}

/** \brief            Copies data from a (sub)matrix
 *
 *  \param[in]        source
 *                    The matrix from which to copy the data.
 *
 *  \param[in]        sub_block
 *                    Submatrix specification  (C-style indexing).
 *
 *  \note             This function copies memory
 */
template <typename T>
inline void Sparse<T>::copy_from(const Sparse<T>& source, SubBlock sub_block) {

  PROFILING_FUNCTION_HEADER

  copy(source, sub_block, *this);

}
/** \overload
 */
template <typename T>
inline void Sparse<T>::copy_from(const Dense<T>& source, SubBlock sub_block) {

  PROFILING_FUNCTION_HEADER

  copy(source, sub_block, *this);

}

/** \brief            Copies a (sub)matrix into the current matrix
 *
 *  \param[in]        source
 *                    The matrix from which to copy the data.
 *
 *  \param[in]        first_row
 *                    The first row of the source matrix which is part of
 *                    the submatrix (i.e. inclusive, C-style indexing).
 *
 *  \param[in]        last_row
 *                    The first row of the source matrix which is not part
 *                    of the submatrix (i.e. exclusive, C-style indexing).
 *
 *  \param[in]        first_col
 *                    The first column of the source matrix which is part of
 *                    the submatrix (i.e. inclusive, C-style indexing).
 *
 *  \param[in]        last_col
 *                    The first column of the source matrix which is not
 *                    part of the submatrix (i.e. exclusive, C-style indexing).
 */
template <typename T>
inline void Sparse<T>::copy_from(const Sparse<T>& source, I_t first_row, 
                                 I_t last_row, I_t first_col, I_t last_col) {
  copy_from(source, SubBlock(first_row, last_row, first_col, last_col));
}
/** \overload
 */
template <typename T>
inline void Sparse<T>::copy_from(const Dense<T>& source, I_t first_row, 
                                 I_t last_row, I_t first_col, I_t last_col) {
  copy_from(source, SubBlock(first_row, last_row, first_col, last_col));
}

/** \brief            Copies a matrix into the current matrix
 *
 *  \param[in]        source
 *                    The matrix from which to copy the data.
 */
template <typename T>
inline void Sparse<T>::copy_from(const Sparse<T>& source) {

  PROFILING_FUNCTION_HEADER

  copy(source, *this);

}
/** \overload
 */
template <typename T>
inline void Sparse<T>::copy_from(const Dense<T>& source) {

  PROFILING_FUNCTION_HEADER

  copy(source, *this);

}

/** \brief              Data copy operator, copies values of one (sub)matrix
 *                      to another matrix.
 *
 *  \param[in]          source
 *                      Right hand side of the operator, used as source of the
 *                      copy. The left hand side is the destination of the
 *                      copy.
 *
 *  \note               Usage:    A << B      // assign's B's values to A
 */
template <typename T>
inline void Sparse<T>::operator<<(const Sparse<T>& source) {

  PROFILING_FUNCTION_HEADER

  copy(source, *this);

}
/** \overload
 */
template <typename T>
inline void Sparse<T>::operator<<(const Dense<T>& source) {

  PROFILING_FUNCTION_HEADER

  copy(source, *this);

}

/** \brief            Return the number of rows in the matrix
 */
template <typename T>
inline I_t Sparse<T>::rows() const {

  if (_format == Format::CSR) {

    if (_minimal_index == 0 && _maximal_index == 0 && _transposed) {
      throw excUserError("rows(): run .update_extent() before querying the "
                         "rows of a transposed CSR matrix");
    }
    return (_transposed) ? _maximal_index - _minimal_index : _size;

  } else if (_format == Format::CSC) {

    if (_minimal_index == 0 && _maximal_index == 0 && !_transposed) {
      throw excUserError("rows(): run .update_extent() before querying the "
                         "rows of a CSC matrix");
    }
    return (_transposed) ? _size : _maximal_index - _minimal_index;

  }

#ifndef LINALG_NO_CHECKS
  else {
    throw excUnimplemented("Sparse.rows(): not implemented for the matrix "
                           "format");
  }
#else
  else return 0;
#endif

}

/** \brief            Return the number of columns in the matrix
 */
template <typename T>
inline I_t Sparse<T>::cols() const {

  if (_format == Format::CSR) {

    if (_minimal_index == 0 && _maximal_index == 0 && !_transposed) {
      throw excUserError("rows(): run .update_extent() before querying the "
                         "columns of a CSR matrix");
    }
    return (_transposed) ? _size : _maximal_index - _minimal_index;

  } else if (_format == Format::CSC) {

    if (_minimal_index == 0 && _maximal_index == 0 && _transposed) {
      throw excUserError("rows(): run .update_extent() before querying the "
                         "columns of a transposed CSC matrix");
    }
    return (_transposed) ? _maximal_index - _minimal_index : _size;

  }

#ifndef LINALG_NO_CHECKS
  else {
    throw excUnimplemented("Sparse.cols(): not implemented for the matrix "
                           "format");
  }
#else
  else return 0;
#endif

}


/** \brief                Changes the matrix' location
 *
 *  \param[in]            new_location
 *                        The new matrix location.
 *
 *  \param[in]            device_id
 *                        The device id of the new location.
 */
template <typename T>
void Sparse<T>::location(Location new_location, int device_id) {

  PROFILING_FUNCTION_HEADER

  if (new_location == _location) {

    if (new_location == Location::host) device_id = 0;
    if (device_id == _device_id) return;

  }

  if (is_empty()) {

    _location  = new_location;
    _device_id = device_id;

  } else {

    Sparse<T> tmp(_size, _n_nonzeros, _first_index, new_location, device_id,
                  _format);

    copy(*this, tmp);
    clone_from(tmp);

  }

}

/** \brief                Resets the matrix to empty state freeing all memory.
 */
template <typename T>
inline void Sparse<T>::unlink() {

  PROFILING_FUNCTION_HEADER

  _n_nonzeros = 0;
  _size = 0;
  _first_index = 0;
  _location = Location::host;
  _device_id = 0;
  _transposed = false;
  _format = Format::CSR;
#ifdef HAVE_MPI
  _row_offset = 0;
#endif
  _properties = 0x0;
  _minimal_index = 0;
  _maximal_index = 0;

  // This frees the memory
  _values = nullptr;
  _indices = nullptr;
  _edges = nullptr;

}

/** \brief            Return true if matrix is on host
  */
template <typename T>
inline bool Sparse<T>::is_on_host() const {

  return _location == Location::host;

}

/** \brief            Return true if matrix is on GPU
  */
template <typename T>
inline bool Sparse<T>::is_on_GPU() const {

#ifdef HAVE_CUDA
  return _location == Location::GPU;
#else
  return false;
#endif

}

/** \brief            Return true if matrix is on MIC
  */
template <typename T>
inline bool Sparse<T>::is_on_MIC() const {

#ifdef HAVE_MIC
  return _location == Location::MIC;
#else
  return false;
#endif

}

/** \brief            Setter for properties
 *
 *  \param[in]        property
 *                    Property to set on the matrix.
 */
template <typename T>
inline void Sparse<T>::set(Property property) {

  if (property == Property::hermitian) {

    if (!_is_complex()) {

      throw excBadArgument("Sparse.set(property): can't set "
                           "Property::hermitian on real matrices");

    }

  }

  _properties = _properties | property;

}

/** \brief            Unset properties
 *
 *  \param[in]        property
 *                    Property to remove from the matrix.
 */
template <typename T>
inline void Sparse<T>::unset(Property property) {

  _properties = _properties & ~(property);

}

/** \brief            Change the index style in the index and edge arrays.
 *
 *  \param[in]        new_first_index
 *                    The desired index of the first element (0 for C
 *                    style, 1 for FORTRAN style indexing). Other values
 *                    are invalid.
 *
 *  \todo             The loop over the elements could be vectorized and
 *                    parallelized.
 */
template <typename T>
inline void Sparse<T>::first_index(int new_first_index) {

  PROFILING_FUNCTION_HEADER

  if (new_first_index == _first_index) {
    return;
  }

#ifndef LINALG_NO_CHECKS
  if (new_first_index != 0 && new_first_index != 1) {

    throw excBadArgument("Sparse.first_index(): invalid value for first_index "
                         "specified (must be zero or one)");

  }
#endif

#ifndef HAVE_MIC
  if (_location == Location::host) {
#else
  if (_location == Location::host || _location == Location::MIC) {
#endif

    auto index_change = new_first_index - _first_index;
    auto _indices_ptr = _indices.get();
    auto _edges_ptr   = _edges.get();

    for (int i = 0; i < _n_nonzeros; ++i) {
      _indices_ptr[i] += index_change;
    }

    for (int i = 0; i < _size + 1; ++i) {
      _edges_ptr[i] += index_change;
    }

    _first_index = new_first_index;
  }

#ifdef HAVE_CUDA
  else if (_location == Location::GPU) {

# ifndef LINALG_NO_CHECKS
    throw excUnimplemented("Sparse.first_index(): Can not change first index on "
                           "matrices on the GPU");
# endif

    // To support this we would need a CUDA kernel that increments all elements
    // in the vectors.

  }
#endif

}

/** \brief            Find the extent of the matrix within a certain row or 
 *                    column range
 *
 *  Maximal and minimal col or rows in case of a CSR or CSC matrix, 
 *  respectively.
 *
 *  \param[in]        first_line
 *                    First row or column to analyze in a CSR or CSC matrix 
 *                    (C-style numbering)
 *
 *  \param[in]        last_line
 *                    Last row or column to analyze in a CSR or CSC matrix 
 *                    (C-style numbering, exclusive)
 *
 *  \returns          Tuple (first, last) where last is exclusive
 *
 *  Example usage:
 *  \code
 *    using std::tie;
 *    tie(first_col, last_col) = my_CSR.find_extent(first_row, last_row);
 *  \endcode
 */
template <typename T>
std::tuple<I_t, I_t> Sparse<T>::find_extent(const I_t first_line,
                                            const I_t last_line) const {

  PROFILING_FUNCTION_HEADER

#ifndef LINALG_NO_CHECKS
  if (_format != Format::CSR && _format != Format::CSC) {
    throw excUnimplemented("Sparse.find_extent(first_line, last_line): "
                           "function not implemented for given format");
  }
  if (first_line < 0) {
    throw excBadArgument("Sparse.find_extent(first_line, last_line), "
                         "first_line: invalid argument (< 0)");
  } else if (last_line > _size) {
    throw excBadArgument("Sparse.find_extent(first_line, last_line), "
                         "last_line: invalid argument (> size())");
  } else if (_location != Location::host) {
    throw excUnimplemented("Sparse.find_extentfirst_line, last_line(): "
                           "currently only matrices in main memory are "
                           "supported");
  }
#endif

  auto edges         = _edges.get();
  auto indices       = _indices.get();
  auto first_index   = _first_index;
  I_t smallest_index = std::numeric_limits<I_t>::max();
  I_t largest_index  = std::numeric_limits<I_t>::min();

  for (I_t line = first_line; line < last_line; ++line) {

    // We assume sorted CSR here
    auto start = indices[edges[line] - first_index];
    // + 1 because we want largest_index to be exclusive
    auto stop  = indices[edges[line + 1] - first_index - 1] + 1;

    smallest_index = (start < smallest_index) ? start : smallest_index;
    largest_index  = (stop  > largest_index)  ? stop  : largest_index;
  
  }

  return std::tuple<I_t, I_t>(smallest_index, largest_index);

}

/** \brief            Update the indexed width (needed for rows() and cols())
 */
template <typename T>
inline void Sparse<T>::update_extent() {

  PROFILING_FUNCTION_HEADER

  using std::tie;

  tie(_minimal_index, _maximal_index) = find_extent(0, _size);

}

} /* namespace LinAlg */

#endif /* LINALG_SPARSE_H_ */
