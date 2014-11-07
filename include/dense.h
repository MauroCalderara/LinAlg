/** \file
 *
 *  \brief            Dense matrix struct (Dense<T>)
 *
 *  \date             Created:  Jul 11, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_DENSE_H_
#define LINALG_DENSE_H_

#include <memory>     // std::shared_ptr
#include <iostream>   // std::cout
#include <iomanip>    // std::setw

#ifdef HAVE_CUDA
#include <cuda_runtime.h>       // various CUDA routines
#include <functional>           // std::bind
#include "CUDA/cuda_memory_allocation.h" // CUDA::cuda_make_shared
                                         // CUDA::cuda_deallocate
#endif

#include "profiling.h"
#include "types.h"
#include "matrix.h"
#include "exceptions.h"

// Forward declaration of routines that are used in the constructors and
// templated members of Dense<T>
#include "utilities/utilities_forward.h"
#include "BLAS/blas_forward.h"

namespace LinAlg {

/** \brief              Dense matrix struct
 */
template <typename T>
struct Dense : Matrix {

  // Construct empty
  Dense();

  // Destructor.
  ~Dense();

  // This would be the famous four and half. However, I find the arguments in
  // the Google C++ Style Guide against copy constructors and assignment
  // operators convincing, in particular for a library that is performance
  // critical. It makes everything a bit more explicit and clear as well. The
  // user code looks less elegant but is more expressive, which I consider a
  // good thing.
  Dense(Dense<T>&& other);
#ifndef DOXYGEN_SKIP
  Dense(Dense<T>& other) = delete;
  Dense(const Dense<T>& other) = delete;
  Dense& operator=(Dense<T> other) = delete;
#endif

  // Construct by allocating memory
  Dense(I_t rows, I_t cols, Location location = Location::host,
        int device_id = 0);

  // Construct from pre-allocated/existing (col-major) array, don't copy data.
  Dense(T* in_array, I_t leading_dimension_in, I_t rows, I_t cols, 
        Location location = Location::host, int device_id = 0);

  // Submatrix creation (from dense)
  Dense(const Dense<T>& source, SubBlock sub_block);
  Dense(const Dense<T>& source, IJ start, IJ stop);
  Dense(const Dense<T>& source, I_t first_row, I_t last_row, I_t first_col,
        I_t last_col);

  // Submatrix creation from ()-operator
  inline Dense<T> operator()(SubBlock sub_block);
  inline Dense<T> operator()(IJ start, IJ stop);
  inline Dense<T> operator()(I_t first_row, I_t last_row, I_t first_col,
                             I_t last_col);

  // Explicitly clone
  inline void clone_from(const Dense<T>& source);
  // Explicit cloning as submatrix
  inline void clone_from(const Dense<T>& source, SubBlock sub_block);
  inline void clone_from(const Dense<T>& source, IJ start, IJ stop);
  inline void clone_from(const Dense<T>& source, I_t first_row, I_t last_row,
                         I_t first_col, I_t last_col);

  // Explicitly move
  inline void move_to(Dense<T>& destination);

  // Allocate new memory (no memory is copied)
  inline void reallocate(I_t rows, I_t cols, Location location = Location::host,
                         int device_id = 0);
  // Allocate new memory according to the size of another matrix (no memory is 
  // copied)
  template <typename U>
  inline void reallocate_like(Dense<U>& other);
  template <typename U>
  inline void reallocate_like(Dense<U>& other, Location location,
                              int device_id);

  // Data copy from one (sub)matrix to another
  inline void operator<<(const Dense<T>& source);

  /// Return the number of rows in the matrix
  inline I_t rows() const { return (_transposed ? _cols : _rows); }
  /// Return the number of columns in the matrix
  inline I_t cols() const { return (_transposed ? _rows : _cols); }
  /// Mark the matrix as transposed
  inline void transpose() { _transposed = !_transposed; }

  /// Return the current location of the matrix
  inline Location location() const { return _location; }
  void location(Location new_location, int device_id = 0);

  void unlink();

  /// Return the storage format
  inline Format format() const { return _format; }

  // Print the matrix contents
  inline void print() const;

  // Return true if matrix is on the given location. These are needed to allow 
  // user code to check the location irrespective of the compilation flags of 
  // the LinAlg libraries
  inline bool is_on_host() const;
  // Return true if matrix is on GPU
  inline bool is_on_GPU()  const;
  // Return true if matrix is on MIC
  inline bool is_on_MIC()  const;

  // Return pointer to matrix begin
  inline T* operator&() const { return _begin(); }

  // Get properties
  inline bool is(Property property) const { return (_properties & property); }

  // Set properties
  inline void set(Property property);

  // Unset properties
  inline void unset(Property property);

  // Returns true if matrix is empty
  inline bool is_empty() const { return (_rows == 0 || _cols == 0); }


#ifndef DOXYGEN_SKIP
  ///////////////////////////////////////////////////////////////////////////
  // Not private: but the _ indicates that the idea is to not mess with these
  // directly unless neccessary.

  // Shared pointer used for the memory containing the matrix
  std::shared_ptr<T> _memory;

  // Offset of first element of the matrix, leading dimension
  I_t _offset;
  I_t _leading_dimension;
  Format _format;
  inline T* _begin() const { return (_memory.get() + _offset); }

  // Rows and columns
  I_t _rows;
  I_t _cols;

  // Location of the matrix
  Location _location;
  int _device_id;

  // Transposition
  bool _transposed;

  // Whether the matrix is complex or not. Overridden in dense.cc with
  // specializations for C_t and Z_t:
  inline bool _is_complex() const { return false; }

  // Properties of the matrix
  unsigned char _properties;
#endif

};

/** \brief              Default (empty) constructor
 *
 *  This constructor doesn't allocate any memory. Typically empty matrices
 *  are initialized suitably by all operations that have output parameters.
 */
template <typename T>
Dense<T>::Dense()
              : _memory(nullptr),
                _offset(0),
                _leading_dimension(0),
                _format(Format::ColMajor),
                _rows(0),
                _cols(0),
                _location(Location::host),
                _device_id(0),
                _transposed(false),
                _properties(0x0) {
#ifdef CONSTRUCTOR_VERBOSE
  std::cout << "Dense.empty_constructor\n";
#endif
}

/*  \brief              Destructor
 */
template <typename T>
Dense<T>::~Dense() {
#ifdef CONSTRUCTOR_VERBOSE
  std::cout << "Dense.destructor\n";
#endif
  unlink();
}

/** \brief              Move constructor
 */
template <typename T>
Dense<T>::Dense(Dense<T>&& other) : Dense() {

#ifdef CONSTRUCTOR_VERBOSE
  std::cout << "Dense.move_constructor\n";
#endif

  // Default initialize using the default initialization and swap
  swap(*this, other);

}

/** \brief              Swap function
 *
 *  Swaps by swapping all data members. Allows for simple move constructor
 *  and assignment operator.
 *
 *  \note               We can't use std::swap to swap to instances directly
 *                      as it uses the assignment operator internally itself.
 *
 *  \param[in,out]      first
 *                      Pointer to 'first', will later point to what was
 *                      'second' before the function call.
 *
 *  \param[in,out]      second
 *                      Pointer to 'second', will later point to what was
 *                      'first' before the function call.
 */
template <typename U>
void swap(Dense<U>& first, Dense<U>& second) {

  using std::swap;
  swap(first._memory, second._memory);
  swap(first._offset, second._offset);
  swap(first._leading_dimension, second._leading_dimension);
  swap(first._rows, second._rows);
  swap(first._cols, second._cols);
  swap(first._location, second._location);
  swap(first._device_id, second._device_id);
  swap(first._transposed, second._transposed);
  swap(first._properties, second._properties);

}


/** \brief            Allocating-only constructor
 *
 *  This constructor allocates some memory area
 *
 *  \param[in]        rows
 *                    Number of rows.
 *
 *  \param[in]        cols
 *                    Number of columns.
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
Dense<T>::Dense(I_t rows, I_t cols, Location location, int device_id)
              : Dense() {

#ifdef CONSTRUCTOR_VERBOSE
  std::cout << "Dense.allocating_constructor\n";
#endif

#ifndef LINALG_NO_CHECKS
  if (rows == 0 || cols == 0) {
    throw excBadArgument("Dense.allocating_constructor: rows or cols must not "
                         "be zero for this constructor. Use the empty "
                         "constructor instead");
  }
#endif

  reallocate(rows, cols, location, device_id);

}

/** \brief            Constructor from array
 *
 *  This constructor allows to create a matrix from pre-allocated/existing 
 *  data in ColMajor format.  No memory is allocated and no data is copied.  
 *  Memory has to be allocated and deallocated by the user.
 *
 *  \param[in]        in_array
 *                    Host pointer to the first element of the matrix. Device
 *                    pointers are not supported.
 *
 *  \param[in]        leading_dimension
 *                    Distance between the first elements of two consecutive
 *                    columns
 *
 *  \param[in]        rows
 *                    Number of rows in the matrix.
 *
 *  \param[in]        cols
 *                    Number of columns in the matrix.
 *
 *  \param[in]        location
 *                    OPTIONAL: Location of the data. Default: Location::host.
 *
 *  \param[in]        device_id
 *                    OPTIONAL: Device id for the data. Default: 0.
 */
template <typename T>
Dense<T>::Dense(T* in_array, I_t leading_dimension, I_t rows, I_t cols,
                Location location, int device_id)
              : _offset(0),
                _leading_dimension(leading_dimension),
                _format(Format::ColMajor),
                _rows(rows),
                _cols(cols),
                _location(location),
                _device_id(device_id),
                _transposed(false),
                _properties(0x0) {

#ifdef CONSTRUCTOR_VERBOSE
  std::cout << "Dense.constructor_from_array\n";
#endif

  // Create a shared_ptr that will not deallocate upon destruction.
  _memory = std::shared_ptr<T>(in_array, [](T* in_array){});

}

/** \brief            Submatrix constructor
 *
 *  Create a submatrix from an existing matrix.
 *  For construction from dense matrices no memory is copied and the ownership
 *  of the memory of the source matrix is shared with the source and all other
 *  submatrices.
 *
 *  \param[in]        source
 *                    The matrix from which to construct the new dense
 *                    matrix.
 *
 *  \param[in]        sub_block
 *                    Submatrix specification.
 */
template <typename T>
Dense<T>::Dense(const Dense<T>& source, SubBlock sub_block)
              : _memory(source._memory),
                _leading_dimension(source._leading_dimension),
                _format(source._format),
                _rows(sub_block.last_row - sub_block.first_row),
                _cols(sub_block.last_col - sub_block.last_col),
                _location(source._location),
                _device_id(source._device_id),
                _transposed(source._transposed) {

#ifdef CONSTRUCTOR_VERBOSE
  std::cout << "Dense.submatrix_constructor\n";
#endif

  if (_format == Format::ColMajor) {
    _offset = source._offset + sub_block.first_col * 
              _leading_dimension + sub_block.first_row;
  } else {
    _offset = source._offset + sub_block.first_row * 
              _leading_dimension + sub_block.first_col;
  }

  if ((sub_block.first_row == sub_block.first_col) &&
      (sub_block.last_row  == sub_block.last_col )   ) {
  
    if (source.is(Property::symmetric)) {
      set(Property::symmetric);
    }
    if (source.is(Property::hermitian)) {
      set(Property::hermitian);
    }
  
  }

}

/** \brief            Submatrix constructor
 *
 *  Create a submatrix from an existing matrix.
 *  For construction from dense matrices no memory is copied and the ownership
 *  of the memory of the source matrix is shared with the source and all other
 *  submatrices.
 *
 *  \param[in]        source
 *                    The matrix from which to construct the new dense
 *                    matrix.
 *
 *  \param[in]        start
 *                    Point to mark the upper left corner of the submatrix 
 *                    (included, c-numbering).
 *
 *  \param[in]        stop
 *                    Point to mark the lower right corner of the submatrix 
 *                    (excluded, c-numbering).
 */
template <typename T>
Dense<T>::Dense(const Dense<T>& source, IJ start, IJ stop)
              : Dense(source, SubBlock(start, stop)) {
}

/** \brief            Submatrix constructor
 *
 *  Create a submatrix from an existing matrix.
 *  For construction from dense matrices no memory is copied and the ownership
 *  of the memory of the source matrix is shared with the source and all other
 *  submatrices.
 *
 *  \param[in]        source
 *                    The matrix from which to construct the new dense
 *                    matrix.
 *
 *  \param[in]        first_row
 *                    The first row of the source matrix which is part of
 *                    the submatrix (i.e. inclusive).
 *
 *  \param[in]        last_row
 *                    The first row of the source matrix which is not part
 *                    of the submatrix (i.e. exclusive).
 *
 *  \param[in]        first_col
 *                    The first column of the source matrix which is part of
 *                    the submatrix (i.e. inclusive).
 *
 *  \param[in]        last_col
 *                    The first column of the source matrix which is not
 *                    part of the submatrix (i.e. exclusive).
 */
template <typename T>
Dense<T>::Dense(const Dense<T>& source, I_t first_row, I_t last_row,
                I_t first_col, I_t last_col)
              : Dense(source, IJ(first_row, first_col), IJ(last_row, last_col)){
}

/** \brief            Submatrix creation
 *
 *  \param[in]        sub_block
 *                    Submatrix specification.
 *
 *  \returns          A submatrix with the given coordinates
 */
template <typename T>
inline Dense<T> Dense<T>::operator()(SubBlock sub_block) {

  return Dense<T>(*this, sub_block);

}

/** \brief            Submatrix creation
 *
 *  \param[in]        start
 *                    Point to mark the upper left corner of the submatrix 
 *                    (included, c-numbering).
 *
 *  \param[in]        stop
 *                    Point to mark the lower right corner of the submatrix 
 *                    (excluded, c-numbering).
 *
 *  \returns          A submatrix with the given coordinates
 */
template <typename T>
inline Dense<T> Dense<T>::operator()(IJ start, IJ stop) {

  return Dense<T>(*this, SubBlock(start, stop));

}

/** \brief            Submatrix creation
 *
 *  \param[in]        first_row
 *                    The first row of the source matrix which is part of
 *                    the submatrix (i.e. inclusive)
 *
 *  \param[in]        last_row
 *                    The first row of the source matrix which is not part
 *                    of the submatrix (i.e. exclusive)
 *
 *  \param[in]        first_col
 *                    The first column of the source matrix which is part of
 *                    the submatrix (i.e. inclusive)
 *
 *  \param[in]        last_col
 *                    The first column of the source matrix which is not
 *                    part of the submatrix (i.e. exclusive).
 *
 *  \returns          A submatrix with the given coordinates
 */
template <typename T>
inline Dense<T> Dense<T>::operator()(I_t first_row, I_t last_row, I_t first_col,
                                     I_t last_col) {

  return Dense<T>(*this, SubBlock(first_row, last_row, first_col, last_col));

}

/** \brief              Cloning from an existing matrix
 *
 *  Applies the parameters of another instance \<source\> to the left hand 
 *  instance. No memory is copied.
 *
 *  \param[in]          source
 *                      The matrix to clone from
 */
template <typename T>
inline void Dense<T>::clone_from(const Dense<T>& source) {

  _memory            = source._memory;
  _offset            = source._offset;
  _format            = source._format;
  _leading_dimension = source._leading_dimension;
  _rows              = source._rows;
  _cols              = source._cols;
  _location          = source._location;
  _device_id         = source._device_id;
  _transposed        = source._transposed;
  _properties        = source._properties;

}

/** \brief            Cloning from an existing matrix
 *
 *  Applies the parameters of another instance \<source\> to the left hand 
 *  instance. No memory is copied.
 *
 *  \param[in]        source
 *                    The matrix to clone from.
 *
 *  \param[in]        sub_block
 *                    Submatrix specification.
 *
 */
template <typename T>
inline void Dense<T>::clone_from(const Dense<T>& source, SubBlock sub_block) {

  clone_from(source);

  if (_format == Format::ColMajor) {
    _offset = source._offset + sub_block.first_col * 
              _leading_dimension + sub_block.first_row;
  } else {
    _offset = source._offset + sub_block.first_col *
              _leading_dimension + sub_block.first_col;
  }

  _rows = sub_block.last_row - sub_block.first_row;
  _cols = sub_block.last_col - sub_block.first_col;

}

/** \brief            Cloning from an existing matrix
 *
 *  Applies the parameters of another instance \<source\> to the left hand 
 *  instance. No memory is copied.
 *
 *  \param[in]        source
 *                    The matrix to clone from.
 *
 *  \param[in]        start
 *                    Point to mark the upper left corner of the submatrix 
 *                    (included, c-numbering).
 *
 *  \param[in]        stop
 *                    Point to mark the lower right corner of the submatrix 
 *                    (excluded, c-numbering).
 */
template <typename T>
inline void Dense<T>::clone_from(const Dense<T>& source, IJ start, IJ stop) {

  clone_from(source, SubBlock(start, stop));

}

/** \brief            Cloning from an existing matrix
 *
 *  Applies the parameters of another instance \<source\> to the left hand 
 *  instance. No memory is copied.
 *
 *  \param[in]        source
 *                    The matrix to clone from
 *
 *  \param[in]        first_row
 *                    The first row of the source matrix which is part of
 *                    the submatrix (i.e. inclusive)
 *
 *  \param[in]        last_row
 *                    The first row of the source matrix which is not part
 *                    of the submatrix (i.e. exclusive)
 *
 *  \param[in]        first_col
 *                    The first column of the source matrix which is part of
 *                    the submatrix (i.e. inclusive)
 *
 *  \param[in]        last_col
 *                    The first column of the source matrix which is not
 *                    part of the submatrix (i.e. exclusive).
 */
template <typename T>
inline void Dense<T>::clone_from(const Dense<T>& source, I_t first_row, 
                                 I_t last_row, I_t first_col, I_t last_col) {

  clone_from(source, SubBlock(first_row, last_row, first_col, last_col));

}

/** \brief              Move matrix to another matrix
 *
 *  'Moves' an instance of Dense<T> to another instance. No memory is copied,
 *  the instance which is moved to destination is left empty (unlinked) after
 *  the operation.
 *
 *  \param[in]          destination
 *                      The destination matrix to move the instance to
 */
template <typename T>
inline void Dense<T>::move_to(Dense<T>& destination) {

  destination.clone_from(*this);
  unlink();

}

/** \brief            Allocates new empty memory for an already constructed
 *                    matrix.
 *
 *  \param[in]        rows
 *                    Number of rows.
 *
 *  \param[in]        cols
 *                    Number of columns.
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
inline void Dense<T>::reallocate(I_t rows, I_t cols, Location location,
                                 int device_id) {

#ifndef LINALG_NO_CHECKS
  if (rows == 0 || cols == 0) {
    throw excBadArgument("Dense.reallocate(): rows or cols must not be zero, "
                         "use unlin() instead");
  }
#endif

  // Allocation for main memory
  if (location == Location::host) {

    _memory = Utilities::host_make_shared<T>(rows * cols);

  }

#ifdef HAVE_CUDA
  // Custom allocator and destructor for GPU memory. Even if cuda_allocate
  // throws we won't leak memory.
  else if (location == Location::GPU) {

    using CUDA::cuda_make_shared;
    _memory = cuda_make_shared<T>(rows * cols, device_id);

  }
#endif

#ifdef HAVE_MIC
  else if (location == Location::MIC) {

    using Utilities::MIC::mic_make_shared;

    // NOTE: the MIC seems not to support 'MIC only' memory such that there must
    // always be a corresponding block of memory on the host itself.
    _memory = mic_make_shared<T>(rows * cols, device_id);

  }
#endif

  _offset = 0;
  _leading_dimension = (_format == Format::ColMajor) ? rows : cols;
  _device_id = 0;
  _location = location;
  _cols = cols;
  _rows = rows;

}


/** \brief            Allocates new empty memory with the same dimensions,
 *                    transposition status and optionally the same location as 
 *                    a given matrix
 *
 *  \param[in]        matrix
 *                    Other matrix whose size and transposition status will be 
 *                    used for this allocation.
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
inline void Dense<T>::reallocate_like(Dense<U>& other) {

  reallocate(other._rows, other._cols, other._location, other._device_id);
  _transposed = other._transposed;

}
/** \overload
 */
template <typename T>
template <typename U>
inline void Dense<T>::reallocate_like(Dense<U>& other, Location location,
                                      int device_id) {

  reallocate(other._rows, other._cols, location, device_id);
  _transposed = other._transposed;

}

/** \brief              Data copy operator, copies values from one (sub)matrix
 *                      to another (sub)matrix.
 *
 *  \param[in]          source
 *                      Right hand side of the operator, used as source of the
 *                      copy. The left hand side is the destination of the
 *                      copy.
 *
 *  \note               Usage:    A << B      // assign's B's values to A
 */
template <typename T>
inline void Dense<T>::operator<<(const Dense<T>& source) {

#ifdef WARN_COPY
  std::cout << "Warning: matrix data copy\n";
#endif

#ifndef LINALG_NO_CHECKS
  // Can't assign to transposed matrices
  if (this->_transposed) {
    throw excBadArgument("DenseA^t << DenseB: can't assign to transposed "
                         "matrices");
  }
#endif

  // If both matrices are empty, this is a noop
  if (source.is_empty() && this->is_empty()) {
    return;
  } else if (this->is_empty()) {
    // If only 'this' is empty allocate memory accordingly
    this->reallocate(source.rows(), source.cols(), source._location,
                      source._device_id);
  }

#ifndef LINALG_NO_CHECKS
  else if (source.rows() != this->rows() || source.cols() != this->cols()) {
    throw excBadArgument("DenseA%s << DenseB%s: dimension mismatch",
                         this->_transposed ? "^t" : " ",
                         source._transposed ? "^t" : " ");
  }

  try {

#endif
    Utilities::copy_2Darray(source._transposed, source._format, source._begin(),
                            source._leading_dimension, source._location,
                            source._device_id, source._rows, source._cols,
                            this->_format, this->_begin(),
                            this->_leading_dimension, this->_location,
                            this->_device_id);
#ifndef LINALG_NO_CHECKS
  } catch (excUnimplemented e) {

    throw excUnimplemented("Dense.operator<<(): exception from copy_2Darray:  "
                           " %s", e.what());

  }
#endif

}

/** \brief            Changes the matrix' location
 *
 *  \param[in]        new_location
 *                    The new matrix location.
 *
 *  \param[in]        device_id
 *                    OPTIONAL: the device id of the new location. DEFAULT: 0
 */
template <typename T>
void Dense<T>::location(Location new_location, int device_id) {

  if (new_location == Location::host) {
    device_id = 0;
  }

  if (new_location == _location) {
    return;
  }

#ifndef LINALG_NO_CHECKS
  else if ((new_location != Location::host) && (_location != Location::host)){

    throw excUnimplemented("Dense.location(): Moves only to and from main "
                           "memory supported. Try moving to main memory first "
                           "and move to target from there.");
    // The reason for this is that it is tedious to guarantee a consistent state
    // in all cases. It is easier if the user handles potential failures.
  }
#endif

  // The matrix is empty, we just update the meta data
  if (is_empty()) {

    _location = new_location;
    _device_id = device_id;

    return;

  }

  // Create a temporary (just increases the pointer count in ._memory so we
  // don't deallocate yet)
  Dense<T> tmp;
  tmp.clone_from(*this);

  // Reallocate the memory on the target
  this->reallocate(_rows, _cols, new_location, device_id);

  // Copy the data over
  *this << tmp;

  // At exit, tmp will be destroyed and the memory is released unless there

}

/** \brief              Resets the matrix/submatrix to empty state.
 *
 *  \note               If the matrix has submatrices, those will not be
 *                      unlinked and the shared memory will not be deleted.
 */
template <typename T>
inline void Dense<T>::unlink() {

  _offset = 0;
  _leading_dimension = 0;
  _rows = 0;
  _cols = 0;
  _location = Location::host;
  _device_id = 0;
  _transposed = false;
  _properties = 0x0;

  // This potentially frees the memory
  _memory = nullptr;

}

/** \brief            Prints the contents of the matrix to std::cout
 */
template <typename T>
inline void Dense<T>::print() const {

  if (is_empty()) {
    return;
  }

  // Make sure we have the data in main memory
  auto source = *this;
  if (_location != Location::host) {
    source.reallocate(rows(), cols());
    source << *this;
  }

  // Set output field width. We use setw which doesn't change the state of
  // std::cout permanently
  auto width = std::cout.precision() + 4;
  if (source._is_complex()) {
    width = std::cout.precision() * 2 + 4;
  }

  auto data = source._begin();
  auto ld   = source._leading_dimension;

  if ((source._format == Format::ColMajor && source._transposed) ||
      (source._format == Format::RowMajor && !source._transposed)  ) {

    for (I_t row = 0; row < source.rows(); ++row) {
      for (I_t col = 0; col < source.cols(); ++col) {

        auto value = data[row * ld + col];
        std::cout << std::setw(width) << std::left << value;

      }
      std::cout << "\n";
    }

  } else {

    for (I_t row = 0; row < source.rows(); ++row) {
      for (I_t col = 0; col < source.cols(); ++col) {

        auto value = data[col * ld + row];
        std::cout << std::setw(width) << std::left << value;

      }
      std::cout << "\n";
    }

  }

}

/** \brief            Return true if matrix is on host
 */
template <typename T>
inline bool Dense<T>::is_on_host() const {

  return _location == Location::host;

}

/** \brief            Return true if matrix is on GPU
 */
template <typename T>
inline bool Dense<T>::is_on_GPU() const {

#ifdef HAVE_CUDA
  return _location == Location::GPU;
#else
  return false;
#endif

}

/** \brief            Return true if matrix is on MIC
 */
template <typename T>
inline bool Dense<T>::is_on_MIC() const {

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
inline void Dense<T>::set(Property property) {

  if (property == Property::hermitian) {

    if (!_is_complex()) {

      throw excBadArgument("Dense.set(property): can't set Property::hermitian "
                           "on real matrices");

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
inline void Dense<T>::unset(Property property) {

  _properties = _properties & ~(property);

}


} /* namespace LinAlg */


#endif /* LINALG_DENSE_H_ */
