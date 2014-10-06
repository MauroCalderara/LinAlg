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

#include <memory>
#define WARN_COPY
#ifdef WARN_COPY
#include <iostream>   // std::cout;
#endif

#ifdef CONSTRUCTOR_VERBOSE
#include <iostream>
#endif

#ifdef HAVE_CUDA
#include <cuda_runtime.h> // various CUDA routines
#include <cusparse_v2.h>
#include "CUDA/cuda_memory_allocation.h"    // CUDA::cuda_make_shared
#endif

#include "types.h"
#include "matrix.h"
#include "exceptions.h"

// Forward declaration of templated utilities that are used in the constructors
// and templated members of Sparse<T>
#include "utilities/utilities_forward.h"


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
  Sparse(I_t size, I_t n_nonzeros, int first_index = 0,
         Location location = Location::host, int device_id = 0,
         Format format = Format::CSR);

  // Construct from existing CSR or CSC array triplet
  Sparse(I_t size, I_t n_nonzeros, T* values, I_t* indices, I_t* edges,
         int first_index = 0, Location location = Location::host,
         int device_id = 0, Format format = Format::CSR);

  // Explicitly clone
  inline void clone_from(const Sparse<T>& source);

  // Explicitly move
  inline void move_to(Sparse<T>& destination);

  // Allocate new memory (no memory is copied)
  inline void reallocate(I_t size, I_t n_nonzeros,
                         Location location = Location::host, int device_id = 0);

  /// Transpose the matrix (but retain the format)
  inline void transpose() { _transposed = !_transposed; };

  /// Get index of first element in arrays
  inline int first_index() const { return _first_index; };
  // Set index of first element in arrays
  inline void first_index(int new_first_index);

  /// Get location
  inline Location location() const { return _location; };
  // Set location
  void location(Location new_location, int device_id = 0);

  // Free all memory, set matrix to empty
  void unlink();

  /// Get format
  inline Format format() const { return _format; };

  // Return true if matrix is on host
  inline bool is_on_host() const;
  // Return true if matrix is on GPU
  inline bool is_on_GPU()  const;
  // Return true if matrix is on MIC
  inline bool is_on_MIC()  const;

  // Get properties
  inline bool is(Property property) const { return (_properties & property); };

  // Set properties
  inline void set(Property property);

  // Unset properties
  inline void unset(Property property);

  // Returns true if matrix is empty
  inline bool is_empty() const { return (_size == 0); };


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
  inline bool _is_complex() const { return false; };

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
                  _format(Format::CSR) {
#ifdef CONSTRUCTOR_VERBOSE
  std::cout << "Sparse.empty_constructor\n";
#endif
#ifdef HAVE_MPI
  _row_offset = 0;
#endif
};

template <typename T>
Sparse<T>::~Sparse() {
#ifdef CONSTRUCTOR_VERBOSE
  std::cout << "Sparse.destructor\n";
#endif
};


/** \brief              Move constructor
 */
template <typename T>
Sparse<T>::Sparse(Sparse&& other) : Sparse() {

#ifdef CONSTRUCTOR_VERBOSE
  std::cout << "Sparse.move_constructor\n";
#endif

  // Default initialize and swap
  swap(*this, other);

};

/** \brief            Swap function
 *
 *  Swaps by swapping all data members. Allows for simple move constructor
 *  and assignment operator.
 *
 *  \note             We can't use std::swap to swap to instances directly
 *                    as it uses the assignment operator internally itself.
 *
 *  \param[in,out]    first
 *                    Pointer to 'first', will later point to what was
 *                    'second' before the function call.
 *
 *  \param[in,out]    second
 *                    Pointer to 'second', will later point to what was
 *                    'first' before the function call.
 */
template <typename U>
void swap(Sparse<U>& first, Sparse<U>& second) {

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
                _format(format) {

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

};

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
                  _format(format) {

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
};

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

  destination.clone_from(*this);
  unlink();

};

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
    checkCUDA(cudaSetDevice(_device_id));
    _values  = cuda_make_shared<T>(n_nonzeros, _device_id);
    _indices = cuda_make_shared<I_t>(n_nonzeros, _device_id);
    _edges   = cuda_make_shared<I_t>(size + 1, _device_id);
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

  _location = location;
  _n_nonzeros = n_nonzeros;
  // "Atomization point"
  _size = size;


};

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

#ifndef LINALG_NO_CHECKS
    throw excUnimplemented("Sparse.first_index(): Can not change first index on "
                           "matrices on the GPU");
#endif

    // To support this we would need a CUDA kernel that increments all elements
    // in the vectors.

  }
#endif


}


/** \brief                Changes the matrix' location
 *
 *  \param[in]            new_location
 *                        The new matrix location.
 *
 *  \param[in]            device_id
 *                        The device id of the new location.
 *
 *  \todo                 If this were supposed to be really safe, we'd need a
 *                        two phase move (first allocate on target, then move
 *                        and delete)
 */
template <typename T>
void Sparse<T>::location(Location new_location, int device_id) {

  if (new_location == Location::host) {
    device_id = 0;
  }

  if (new_location == _location) {
    return;
  }
#ifndef LINALG_NO_CHECKS
  else if ((new_location != Location::host) && (_location != Location::host)) {

    throw excUnimplemented("Moves only to and from main memory supported. "
                           "Try moving to main memory first and move to "
                           "target from there.");
    // The reason for this is that it is tedious to guarantee a consistent state
    // in all cases. It is easier if the user handles potential failures.
  }
#endif
#ifdef HAVE_CUDA
  else if ((new_location == Location::GPU) && (_location == Location::host)) {

#ifdef WARN_COPY
    std::cout << "Warning: copying matrix to GPU\n";
#endif

    using CUDA::cuda_make_shared;
    using Utilities::copy_1Darray;

    int prev_device;
    checkCUDA(cudaGetDevice(&prev_device));

    // Allocate memory on GPU
    checkCUDA(cudaSetDevice(device_id));
    auto device_values  = cuda_make_shared<T>(_n_nonzeros, _device_id);
    auto device_indices = cuda_make_shared<I_t>(_n_nonzeros, _device_id);
    auto device_edges   = cuda_make_shared<I_t>(_size + 1, _device_id);

    // Copy matrix to GPU
    copy_1Darray(_values.get(), _n_nonzeros, device_values.get(), _location,
                 _device_id, new_location, device_id);
    copy_1Darray(_indices.get(), _n_nonzeros, device_indices.get(), _location,
                 _device_id, new_location, device_id);
    copy_1Darray(_edges.get(), _size + 1, device_edges.get(), _location,
                 _device_id, new_location, device_id);

    checkCUDA(cudaSetDevice(prev_device));

    // Change members to reflect the change
    _location = new_location;
    _device_id = device_id;

    // This frees the memory on the host
    _values = device_values;
    _indices = device_indices;
    _edges = device_edges;


  }
  else if ((new_location == Location::host) && (_location == Location::GPU)) {

#ifdef WARN_COPY
    std::cout << "Warning: copying matrix from GPU\n";
#endif

    using Utilities::host_make_shared;
    using Utilities::copy_1Darray;

    int prev_device;
    checkCUDA(cudaGetDevice(&prev_device));

    // Allocate memory on CPU
    auto host_values  = host_make_shared<T>(_n_nonzeros);
    auto host_indices = host_make_shared<I_t>(_n_nonzeros);
    auto host_edges   = host_make_shared<I_t>(_size + 1);

    // Copy matrix to GPU
    checkCUDA(cudaSetDevice(_device_id));
    copy_1Darray(_values.get(), _n_nonzeros, host_values.get(), _location,
                 _device_id, new_location, device_id);
    copy_1Darray(_indices.get(), _n_nonzeros, host_indices.get(), _location,
                 _device_id, new_location, device_id);
    copy_1Darray(_edges.get(), _size + 1, host_edges.get(), _location,
                 _device_id, new_location, device_id);

    checkCUDA(cudaSetDevice(prev_device));

    // Change members to reflect the change
    _location = new_location;
    _device_id = device_id;

    // This frees the memory on the device
    _values = host_values;
    _indices = host_indices;
    _edges = host_edges;

  }
#endif
#ifdef HAVE_MIC

  // Note: AFAICT we can't deallocate the memory on the host while having the
  // matrix on the MIC if we ever want to transfer it back.
  //
  // One possible way out would be to use the shared address space feature.

  else if ((new_location == Location::MIC) && (_location == Location::host)) {

    #pragma offload_transfer target (mic:device_id) \
                             in (_values.get():length(_n_nonzeros), \
                                 alloc_if(true), free_if(false)) \
                             in (_indices.get():length(_n_nonzeros), \
                                 alloc_if(true), free_if(false)) \
                             in (_edges.get():length(_n_nonzeros), \
                                 alloc_if(true), free_if(false))

    _location = Location::MIC;
    _device_id = device_id;

  }
  else if ((new_location == Location::host) && (_location == Location::MIC)) {

    #pragma offload_transfer target (mic:device_id) \
                             out (_values.get():length(_n_nonzeros), \
                                  alloc_if(true), free_if(false)) \
                             out (_indices.get():length(_n_nonzeros), \
                                  alloc_if(true), free_if(false)) \
                             out (_edges.get():length(_n_nonzeros), \
                                  alloc_if(true), free_if(false))

    _location = Location::host;
    _device_id = 0;

  }
#endif

}

/** \brief                Resets the matrix to empty state freeing all memory.
 */
template <typename T>
inline void Sparse<T>::unlink() {

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

  // This frees the memory
  _values = nullptr;
  _indices = nullptr;
  _edges = nullptr;

};

/** \brief            Return true if matrix is on host
  */
template <typename T>
inline bool Sparse<T>::is_on_host() const {

  return _location == Location::host;

};

/** \brief            Return true if matrix is on GPU
  */
template <typename T>
inline bool Sparse<T>::is_on_GPU() const {

#ifdef HAVE_CUDA
  return _location == Location::GPU;
#else
  return false;
#endif

};

/** \brief            Return true if matrix is on MIC
  */
template <typename T>
inline bool Sparse<T>::is_on_MIC() const {

#ifdef HAVE_MIC
  return _location == Location::MIC;
#else
  return false;
#endif

};

/** \brief            Setter for properties
 *
 *  \param[in]        property
 *                    Property to set on the matrix.
 */
template <typename T>
inline void Sparse<T>::set(Property property) {

  if (property == Property::Hermitian) {

    if (!_is_complex()) {

      throw excBadArgument("Sparse.set(property): can't set "
                           "Property::Hermitian on real matrices");

    }

  }

  _properties = _properties | property;

};

/** \brief            Unset properties
 *
 *  \param[in]        property
 *                    Property to remove from the matrix.
 */
template <typename T>
inline void Sparse<T>::unset(Property property) {

  _properties = _properties & ~(property);

};


} /* namespace LinAlg */


#endif /* LINALG_SPARSE_H_ */
