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
#include "../exceptions.h"
#include "../streams.h"
#include "../utilities/timer.h"  // LinAlg::Utilities::timer
#include "../metadata.h"   // LinAlg::MetaData
#include "send_recv.h"

#include "../dense.h"

namespace LinAlg {

namespace MPI {

/*  \brief            Receive meta data about a matrix asynchronously and
 *                    save/apply it to the supplied matrix.
 *
 *  \param[in]        matrix
 *                    The matrix to which the meta data is to be applied.
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
// This would require general streams to be implemented.

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
 */
template <typename T>
void send_matrix(Dense<T>& matrix, MPI_Comm communicator, int receiving_rank,
                 int tag) {

#ifndef LINALG_NO_CHECKS
  if (matrix._rows == 0) {

    throw excBadArgument("send_matrix(): matrix to send must not be empty.");

  }
#endif

  // Check if the matrix is continuous or not. If it is not, we'll later need to
  // create a buffer.
  Dense<T> buffer;

  if (((matrix._leading_dimension == matrix._rows) &&
       (matrix._format == Format::ColMajor)          ) ||
      ((matrix._leading_dimension == matrix._cols) &&
       (matrix._format == Format::RowMajor)          )   ) {

    // The matrix is continous within memory, so the buffer is the matrix
    // itself.
    buffer.clone_from(matrix);

  } else {

    // Non continuous matrices in memory need a buffer
    buffer.reallocate(matrix._rows, matrix._cols, matrix._location,
                      matrix._device_id);
    buffer << matrix;

  }

  // Avoid collisions with multi-tag operations
  auto internal_tag = MPI_TAG_OFFSET * tag;

  auto array = buffer._begin();
  auto size = buffer._rows * buffer._cols;

  auto error = mpi_send(array, size, receiving_rank, internal_tag,
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

};

/** \brief            Send a matrix asynchronously
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
 */
template <typename T>
void send_matrix_async(Dense<T>& matrix, MPI_Comm communicator,
                       int receiving_rank, int tag, MPIStream& stream) {

#ifndef LINALG_NO_CHECKS
  if (matrix._rows == 0) {

    throw excBadArgument("send_matrix(): matrix to send must not be empty.");

  }
#endif

  // Check if the default stream has been passed, that is synchronous behavior
  // has been requested
  if (stream.synchronous_operation) {

    std::printf("send_matrix_async(): default stream specified, reverting to "
                "synchronous operation.\n");

    send_matrix(matrix, communicator, receiving_rank, tag);

    return;

  }

  // Check if the matrix is continuous or not.
  Dense<T> buffer;

  if (((matrix._leading_dimension == matrix._rows) &&
       (matrix._format == Format::ColMajor)          ) ||
      ((matrix._leading_dimension == matrix._cols) &&
       (matrix._format == Format::RowMajor)          )   ) {

    // The matrix is continous within memory, so the buffer is the matrix
    // itself.
    buffer.clone_from(matrix);

  }

#ifndef LINALG_NO_CHECKS
  else {

    throw excUnimplemented("send_matrix_async(): asynchronous transfer for non-"
                           "continuous matrices currently unimplemented."
                           "Create a suitable buffer and delete it after "
                           "synchronizing as a workaround");
    // For the sender that would require that we have create a buffer whose
    // lifetime we can extend beyond the scope of the function call and
    // then end at .sync() (i.e. a normal matrix wouldn't work and it also
    // can't be pushed into a vector since there could be different types of
    // matrices) .
    //
    // One can create a buffer from a shared_ptr<void> with a custom deleter
    // (delete[]), and push that onto a std::vector<shared_ptr<void>> that is
    // a member of the stream. At synchronization the vector is cleared and
    // all buffers free'd. Thus the sender part is comparably simple, but the
    // receiver still needs threads to work off the queue...

  }
#endif

  // Avoid collisions with multi-tag operations
  auto internal_tag = MPI_TAG_OFFSET * tag;

  auto array = buffer._begin();
  auto size = buffer._rows * buffer._cols;

  // Add task to stream
  auto stream_position = stream.add_operations(1);
  auto request = &stream.requests[stream_position];
  stream.synchronized = false;

  auto error = mpi_isend(array, size, receiving_rank, internal_tag,
                         communicator, request);

  // Error handling happens when synchronizing on the stream.
  stream.statuses[stream_position] = construct_status(error, communicator,
                                                       tag);

};
/** \overload
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
 *  \returns          stream
 *                    Stream to synchronize the operation
 */
template <typename T>
inline MPIStream send_matrix_async(Dense<T>& matrix, MPI_Comm communicator,
                                   int receiving_rank, int tag) {

  MPIStream stream;

  send_matrix_async(matrix, communicator, receiving_rank, tag, stream);

  return std::move(stream);

};



/** \brief            Receive a remote matrix into a preallocated local matrix
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
 */
template <typename T>
void receive_matrix(Dense<T>& matrix, MPI_Comm communicator, int sending_rank,
                    int tag) {

#ifndef LINALG_NO_CHECKS
  if (matrix._rows == 0) {

    throw excBadArgument("receive_matrix(): receiving matrix must not be "
                         "empty. Preallocate with the right size");

  }
#endif

  // Check if the matrix is continuous or not. If it is not, we'll later need to
  // create a buffer.
  Dense<T> buffer;
  bool buffered;

  if (((matrix._leading_dimension == matrix._rows) &&
       (matrix._format == Format::ColMajor)          ) ||
      ((matrix._leading_dimension == matrix._cols) &&
       (matrix._format == Format::RowMajor)          )   ) {

    // The matrix is continous within memory, so the buffer is the matrix
    // itself.
    buffer.clone_from(matrix);
    buffered = false;

  } else {

    // Non continuous matrices in memory need a buffer (and later
    // synchronization)
    buffer.reallocate(matrix._rows, matrix._cols, matrix._location,
                      matrix._device_id);
    buffered = true;

  }

  // Avoid collisions with multi-tag operations
  auto internal_tag = MPI_TAG_OFFSET * tag;

  auto array = buffer._begin();
  auto size = buffer._rows * buffer._cols;

  MPI_Status status;
  auto error = mpi_recv(array, size, sending_rank, internal_tag,
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

};

/** \brief            Receive a remote matrix into a preallocated local matrix
 *                    asynchronously
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
 *                    synchronous
 */
template <typename T>
void receive_matrix_async(Dense<T>& matrix, MPI_Comm communicator,
                          int sending_rank, int tag, MPIStream& stream) {

#ifndef LINALG_NO_CHECKS
  if (matrix._rows == 0) {

    throw excBadArgument("receive_matrix_async(): receiving matrix must not be "
                         "empty. Preallocate with the right size");

  }
#endif

  // Check if the default stream has been passed, that is synchronous behavior
  // has been requested
  if (stream.synchronous_operation) {

    std::printf("receive_matrix_async(): default stream specified, reverting "
                "to synchronous operation.\n");

    receive_matrix(matrix, communicator, sending_rank, tag);

    return;

  }

  // Check if the matrix is continuous or not. If it is not, we'll later need to
  // create a buffer.
  Dense<T> buffer;

  if (((matrix._leading_dimension == matrix._rows) &&
       (matrix._format == Format::ColMajor)          ) ||
      ((matrix._leading_dimension == matrix._cols) &&
       (matrix._format == Format::RowMajor)          )   ) {

    // The matrix is continous within memory, so the buffer is the matrix
    // itself.
    buffer.clone_from(matrix);

  }

#ifndef LINALG_NO_CHECKS
  else {

    throw excUnimplemented("receive_matrix(): asynchronous transfer for non-"
                           "continuous matrices currently unimplemented. "
                           "Create a suitable buffer and flush it after "
                           "synchronizing as a workaround");
    // That would require that we have a worker thread that initialized the
    // MPI transfer, waits for completion, flushes the buffer into the matrix
    // and delete the buffer.

  }
#endif

  // Avoid collisions with multi-tag operations
  auto internal_tag = MPI_TAG_OFFSET * tag;

  auto array = buffer._begin();
  auto size = buffer._rows * buffer._cols;

  // Add task to the stream
  auto pos = stream.add_operations(1);
  auto request = &stream.requests[pos];
  stream.synchronized = false;

  auto error = mpi_irecv(array, size, sending_rank, internal_tag,
                         communicator, request);

  // Errors are handled at synchronization time
  stream.statuses[pos] = construct_status(error, communicator, tag);

};
/** \overload
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
 *  \returns          stream to synchronize the operation
 */
template <typename T>
inline MPIStream receive_matrix_async(Dense<T>& matrix, MPI_Comm communicator,
                                      int sending_rank, int tag) {

  MPIStream stream;
  receive_matrix_async(matrix, communicator, sending_rank, tag, stream);
  return std::move(stream);

};

} /* namespace MPI */

} /* namespace LinAlg */

#endif /* HAVE_MPI */

#endif /* LINALG_MPI_SEND_RECV_MATRIX_H_ */
