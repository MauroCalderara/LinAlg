/** \file
 *
 *  \brief            Sending/receiving matrices using MPI
 *
 *  \date             Created:  Jul 15, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_MPI_SEND_RECV_MATRIX_H_
#define LINALG_MPI_SEND_RECV_MATRIX_H_

#ifdef HAVE_MPI

#include <mpi.h>      // all MPI stuff

#include <vector>     // std::vector
#include <string>     // std::string
#include <sstream>    // std::stringstream
#include <cstdio>     // std::printf
#include <utility>    // std::move

#include "../types.h"
#include "../profiling.h"
#include "../exceptions.h"
#include "../streams.h"
#include "../utilities/timer.h"  // LinAlg::Utilities::timer
#include "../metadata.h"   // LinAlg::MetaData
#include "send_receive.h"
#include "send_receive_meta.h" // LinAlg::MPI::send_meta
#include "status.h"

#include "../dense.h"

namespace LinAlg {

namespace MPI {

/** \brief            Send a matrix to a remote host synchronously
 *
 *  \param[in]        matrix
 *                    Matrix to send
 *
 *  \param[in]        communicator
 *                    MPI communicator for the transfer
 *
 *  \param[in]        receiving_rank
 *                    MPI rank of the receiving party
 *
 *  \param[in]        tag
 *                    Tag of the transfer (must match tag on receiver)
 *
 *  \param[in]        recipient_preallocated
 *                    Whether the recipient of the matrix preallocated the 
 *                    memory already.
 */
template <typename T>
void send_matrix(Dense<T>& matrix, MPI_Comm communicator, int receiving_rank,
                 int tag, bool recipient_preallocated) {

  PROFILING_FUNCTION_HEADER

  if (matrix.is_empty()) {
    return;
  }

  // Avoid collisions with multi-tag operations. If sender and receiver are 
  // inconsistent in whether they need to exchange the metadata first, this 
  // should rise an error as the data types don't match.
  auto internal_meta_tag = MPI_TAG_OFFSET * tag;
  auto internal_data_tag = MPI_TAG_OFFSET * tag + 1;

  // If needed, send dimensions first so the sender can allocate memory
  if (!recipient_preallocated) { 
    send_meta(matrix, communicator, receiving_rank, internal_meta_tag);
  }

  // Check if the matrix is continuous or not. If it is not, we'll need to
  // create a buffer. We also create a buffer if this is requested 
  // specifically.
  Dense<T> buffer;
  auto buffered = false;

  // Conditions under which we need to buffer anyway
  if (matrix._leading_dimension != matrix._rows &&
      matrix._format == Format::ColMajor           ) buffered = true;
  if (matrix._leading_dimension != matrix._cols &&
      matrix._format == Format::RowMajor           ) buffered = true;

# ifndef HAVE_CUDA_AWARE_MPI
  if (matrix.is_on_GPU()) buffered = true;
# endif

  if (buffered) {

#   ifndef HAVE_CUDA_AWARE_MPI
    auto buffer_location  = (matrix.is_on_GPU()) ? Location::host : 
                                                   matrix._location;
    auto buffer_device_id = (matrix.is_on_GPU()) ? 0 : matrix._device_id;
#   else
    auto buffer_location  = matrix._location;
    auto buffer_device_id = matrix._device_id;
#   endif

    buffer.reallocate(matrix._rows, matrix._cols, buffer_location,
                      buffer_device_id);

    buffer << matrix;

  } else {

    buffer.clone_from(matrix);

  }

  auto array = buffer._begin();
  auto size = buffer._rows * buffer._cols;

  auto error = mpi_send(array, size, receiving_rank, internal_data_tag,
                        communicator);

#ifndef LINALG_NO_CHECKS
  if (error != MPI_SUCCESS) {

    // Construct a status and a corresponding exception
    auto my_status = construct_status(error, communicator, tag);

    excMPIError my_exception("send_matrix(): MPI error, ");

    my_exception.set_status(my_status);

    throw my_exception;

  }
#endif

}

/** \brief            Send a matrix to a remote host synchronously.  
 *
 * The remote host is assumed to have allocated the memory for receiving the 
 * transfer.
 *
 *  \param[in]        matrix
 *                    Matrix to send
 *
 *  \param[in]        communicator
 *                    MPI communicator for the transfer
 *
 *  \param[in]        receiving_rank
 *                    MPI rank of the receiving party
 *
 *  \param[in]        tag
 *                    Tag of the transfer (must match tag on receiver)
 */
template <typename T>
inline void send_matrix(Dense<T>& matrix, MPI_Comm communicator, 
                        int receiving_rank, int tag) {

  send_matrix(matrix, communicator, receiving_rank, tag, true);

}


/** \brief            Send a matrix to a remote host asynchronously
 *
 *  \param[in]        matrix
 *                    Matrix to send
 *
 *  \param[in]        communicator
 *                    MPI communicator for the transfer
 *
 *  \param[in]        receiving_rank
 *                    MPI rank of the receiving party
 *
 *  \param[in]        tag
 *                    Tag of the transfer (must match tag on receiver)
 *
 *  \param[in]        stream
 *                    Stream to use
 *
 *  \param[in]        recipient_preallocated
 *                    Whether the recipient of the matrix preallocated the 
 *                    memory already.
 *
 *  \param[in]        buffered
 *                    Whether the transfer should be buffered. If the send is 
 *                    buffered, the data can be modified immediately after the 
 *                    call returns.
 *
 *  \returns          The ticket number to synchronize with the stream.
 */
template <typename T>
inline I_t send_matrix_async(Dense<T>& matrix, MPI_Comm communicator,
                             int receiving_rank, int tag, Stream& stream,
                             bool recipient_preallocated, bool buffered) {

  PROFILING_FUNCTION_HEADER

  Dense<T> buffer;

  if (buffered) buffer << matrix;
  else          buffer.clone_from(matrix);

  // Create a bundle of the buffer and a lambda function that gets all static 
  // parameters by value
  DenseMatrixFunctionBundle<T> bundle(buffer, 
    [=] (Dense<T>& tmp_matrix) { 
      send_matrix(tmp_matrix, communicator, receiving_rank, tag, 
                  recipient_preallocated, buffered); 
    }
  );

  // This here creates a lasting copy of bundle, including a clone of buffer.
  // Thus buffer is not deleted when leadving the scope.
  return stream.add(std::bind(bundle));

}


/** \brief            Receive a remote matrix into an optionally preallocated 
 *                    local matrix
 *
 *  \param[in,out]    matrix
 *                    Local matrix to store the remote matrix. An emtpy matrix 
 *                    is signaling that the dimensions need to be exchanged 
 *                    prior to the exchange of any data. This must match the 
 *                    'recipient_preallocated' setting on the sender, that is  
 *                    if matrix is empty, recipient_preallocated must be set to 
 *                    false on the sender.
 *
 *  \param[in]        communicator
 *                    MPI communicator for the transfer
 *
 *  \param[in]        sending_rank
 *                    MPI rank of the sending party
 *
 *  \param[in]        tag
 *                    Tag of the transfer (must match tag on sender)
 */
template <typename T>
void receive_matrix(Dense<T>& matrix, MPI_Comm communicator, int sending_rank,
                    int tag) {

  PROFILING_FUNCTION_HEADER

  // Avoid collisions with multi-tag operations. If sender and receiver are 
  // inconsistent in whether they need to exchange the metadata first, this 
  // should rise an error as the data types don't match.
  auto internal_meta_tag = MPI_TAG_OFFSET * tag;
  auto internal_data_tag = MPI_TAG_OFFSET * tag + 1;

  if (matrix.is_empty()) {
    receive_meta(matrix, communicator, sending_rank, internal_meta_tag);
  }

  // Check if we need to buffer
  Dense<T> buffer;
  auto buffered = false;

  if (matrix._leading_dimension != matrix._rows &&
      matrix._format == Format::ColMajor           ) buffered = true;
  if (matrix._leading_dimension != matrix._cols &&
      matrix._format == Format::RowMajor           ) buffered = true;

# ifndef HAVE_CUDA_AWARE_MPI
  if (matrix.is_on_GPU()) buffered = true;
# endif

  if (buffered) {

#   ifndef HAVE_CUDA_AWARE_MPI
    auto buffer_location  = (matrix.is_on_GPU()) ? Location::host : 
                                                   matrix._location;
    auto buffer_device_id = (matrix.is_on_GPU()) ? 0 : matrix._device_id;
#   else
    auto buffer_location  = matrix._location;
    auto buffer_device_id = matrix._device_id;
#   endif

    buffer.reallocate(matrix._rows, matrix._cols, buffer_location,
                      buffer_device_id);

  } else {

    buffer.clone_from(matrix);

  }

  auto array = buffer._begin();
  auto size = buffer._rows * buffer._cols;

  MPI_Status status;
  auto error = mpi_recv(array, size, sending_rank, internal_data_tag,
                        communicator, &status);

#ifndef LINALG_NO_CHECKS
  if (error != MPI_SUCCESS) {

    excMPIError my_exception("receive_matrix(): MPI error, ");
    my_exception.set_status(status);
    throw my_exception;

  }
#endif

  if (buffered) {

    matrix << buffer;

  }

}

/** \brief            Receive a remote matrix into a local, preallocated matrix
 *                    asynchronously using an MPIStream
 *
 *  \param[in,out]    matrix
 *                    Preallocated local matrix to store the remote matrix
 *
 *  \param[in]        communicator
 *                    MPI communicator for the transfer
 *
 *  \param[in]        sending_rank
 *                    MPI rank of the sending party
 *
 *  \param[in]        tag
 *                    Tag of the transfer (must match tag on sender)
 *
 *  \param[in]        stream
 *                    MPI stream to use for asynchronous transfers. If no
 *                    stream or an MPIStream(0) is specified, the operation is
 *                    synchronous.
 *
 *  \note             The function is unbuffered. The caller is responsible to 
 *                    ensure that the matrix object is still alive when the 
 *                    stream executes the task. Failure to do so will lead to 
 *                    undefined behavior. Use with care.
 */
template <typename T>
void receive_matrix_async(Dense<T>& matrix, MPI_Comm communicator,
                          int sending_rank, int tag, MPIStream& stream) {

  PROFILING_FUNCTION_HEADER

  // Check if the default stream has been passed, that is synchronous behavior
  // has been requested
  if (stream.synchronous_operation) {

    std::printf("receive_matrix_async(): default stream specified, reverting "
                "to synchronous operation.\n");

    receive_matrix(matrix, communicator, sending_rank, tag);

    return;

  }

  if (matrix.is_empty()) {

    throw excBadArgument("receive_matrix_async(): asynchronous transfer with "
                         "MPIStream requires preallocated matrices");

  }

  // Check if the matrix is continuous/sendable or not. If it is not, we can't 
  // receive it asynchronously since we can't create a buffer with an MPIStream
  bool directly_sendable = true;

  if (matrix._leading_dimension != matrix._rows &&
      matrix._format == Format::ColMajor           ) directly_sendable = false;
  if (matrix._leading_dimension != matrix._cols &&
      matrix._format == Format::RowMajor           ) directly_sendable = false;

# ifndef HAVE_CUDA_AWARE_MPI
  if (matrix.is_on_GPU()) directly_sendable = false;
# endif

  if (!directly_sendable) {

    throw excBadArgument("receive_matrix(): asynchronous transfer with "
                         "MPIStream requires that the matrix can be received "
                         "using a single MPI call. Create a suitable buffer "
                         "and flush it after synchronizing as a workaround. "
                         "Alternatively use a general stream.");

  }

  // Avoid collisions with multi-tag operations
  auto internal_tag = MPI_TAG_OFFSET * tag;

  auto array = matrix._begin();
  auto size = matrix._rows * matrix._cols;

  // Add task to the stream
  auto pos = stream.add_operations(1);
  auto request = &stream.requests[pos];
  stream.synchronized = false;

  auto error = mpi_irecv(array, size, sending_rank, internal_tag,
                         communicator, request);

  // Errors are handled at synchronization time
  stream.statuses[pos] = construct_status(error, communicator, tag);

}

/** \brief            Receive a remote matrix into a local, optionally 
 *                    preallocated matrix asynchronously using a general stream
 *
 *  \param[in,out]    matrix
 *                    Local matrix to store the remote matrix. If matrix is 
 *                    preallocated, the transaction is buffered and no meta  
 *                    data is exchanged. If matrix is not preallocated, the 
 *                    transaction is unbuffered and metadata will be exchanged.
 *                    See below for a rationale.
 *
 *  \param[in]        communicator
 *                    MPI communicator for the transfer
 *
 *  \param[in]        sending_rank
 *                    MPI rank of the sending party
 *
 *  \param[in]        tag
 *                    Tag of the transfer (must match tag on sender)
 *
 *  \param[in]        stream
 *                    MPI stream to use for asynchronous transfers. If no
 *                    stream or an MPIStream(0) is specified, the operation is
 *                    synchronous.
 *
 *  \returns          The ticket number to synchronize with the stream.
 *
 *  \note             If matrix is not preallocated, the function exchanges the 
 *                    metadata, allocates the memory and updates the matrix 
 *                    metadata accordingly. Thus it cannot be buffered. If the 
 *                    matrix is allocated, the function stores the transfer in 
 *                    the memory region that matrix pointed to at invocation 
 *                    time.
 */
template <typename T>
I_t receive_matrix_async(Dense<T>& matrix, MPI_Comm communicator,
                          int sending_rank, int tag, Stream& stream) {

  if (matrix.is_empty()) {
  
    // Can't buffer as we need to update the matrix meta data. That is we pass 
    // matrix by reference and trust the user to maintain lifetime of the  
    // object till invocation of the function in the stream.
    auto payload = [=, &matrix] () {
      receive_matrix(matrix, communicator, sending_rank, tag);
    };
  
    return stream.add(payload);
  
  } else {
  
    // Create a bundle of the buffer and a lambda function that gets all static 
    // parameters by value
    Dense<T> buffer;
    buffer.clone_from(matrix);
    DenseMatrixFunctionBundle<T> bundle(buffer, 
      [=] (Dense<T>& tmp_matrix) { 
        receive_matrix(tmp_matrix, communicator, sending_rank, tag);
      }
    );

    return stream.add(std::bind(bundle));
  
  }

}

} /* namespace MPI */

} /* namespace LinAlg */

#endif /* HAVE_MPI */

#endif /* LINALG_MPI_SEND_RECV_MATRIX_H_ */
