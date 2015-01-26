/** \file
 *
 *  \brief            Sending/receiving metadata using MPI
 *
 *  \date             Created:  Jul 15, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_MPI_SEND_RECV_META_H_
#define LINALG_MPI_SEND_RECV_META_H_

#include "../preprocessor.h"

#ifdef HAVE_MPI

# include <mpi.h>      // all MPI stuff

# include <vector>     // std::vector
# include <string>     // std::string
# include <sstream>    // std::stringstream
# include <cstdio>     // std::printf
# include <utility>    // std::move
# include <functional> // std::bind

# include "../types.h"
# include "../profiling.h"
# include "../exceptions.h"
# include "../streams.h"
# include "../metadata.h"   // LinAlg::MetaData
# include "send_receive.h"
# include "status.h"

# include "../dense.h"

namespace LinAlg {

namespace MPI {

using LinAlg::Utilities::check_rank;

/** \brief            Send meta data
 *
 *  \param[in]        meta
 *                    The meta data to be sent.
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
inline void send_meta(MetaData meta, MPI_Comm communicator, int receiving_rank,
                      int tag) {

  PROFILING_FUNCTION_HEADER

# ifndef LINALG_NO_CHECKS
  check_rank(receiving_rank, communicator, "send_meta()");
# endif

  auto internal_tag = MPI_TAG_OFFSET * tag;

  auto error = mpi_send(meta.data(), meta.size(), receiving_rank, internal_tag,
                        communicator);

# ifndef LINALG_NO_CHECKS
  if (error != MPI_SUCCESS) {

    // Construct a status and a corresponding exception
    auto my_status = construct_status(error, communicator, tag);

    excMPIError my_exception("send_meta(): MPI error, ");

    my_exception.set_status(my_status);

    throw my_exception;

  }
# endif

}

/** \brief            Send meta data about a matrix asynchronously using a 
 *                    Stream
 *
 *  \param[in]        meta
 *                    The meta data to be sent.
 *
 *  \param[in]        communicator
 *                    MPI communicator for the transfer
 *
 *  \param[in]        receiving_rank
 *                    MPI rank of the receiving party
 *
 *  \param[in]        tag
 *                    Tag of the transfer (must match tag on sender)
 *
 *  \param[in]        stream
 *                    Stream to use
 *
 *  \returns          Ticket number for synchronizing on the stream.
 *
 *  \note             The function is unbuffered if the stream has 
 *                    'perfer_native' set to true. The caller is responsible 
 *                    to ensure that the matrix object is still alive when the 
 *                    stream executes the task. Failure to do so will lead to 
 *                    undefined behavior. Use with care.
 *                    The function is buffered if the stream has 
 *                    'prefer_native' set to false.
 */
inline I_t send_meta_async(MetaData& meta, MPI_Comm communicator,
                           int receiving_rank, int tag, Stream& stream) {

  // Note: we have to accept by reference as mpi_isend copies the pointer

# ifndef LINALG_NO_CHECKS
  check_rank(receiving_rank, communicator, "send_meta_async()");
# endif

  PROFILING_FUNCTION_HEADER

  if (stream.synchronous) {

    send_meta(meta, communicator, receiving_rank, tag);

    return 0;

  }

  if (stream.prefer_native) {

    auto internal_tag = MPI_TAG_OFFSET * tag;

    // Add task to stream
    auto stream_position = stream._add_mpi_tasks(1);

    auto request = &(stream.mpi_requests[stream_position]);

    auto error = mpi_isend(meta.data(), meta.size(), receiving_rank,
                           internal_tag, communicator, request);

    // Error handling happens when synchronizing on the stream.
    stream.mpi_statuses[stream_position] = construct_status(error, communicator,
                                                            tag);

    stream.mpi_synchronized = false;

    return 0;

  } else {

# ifndef LINALG_NO_CHECKS
    if (!stream.thread_alive) {
      throw excBadArgument("send_meta_async(): provided stream has no active "
                           "worker thread");
    }
# endif

    auto task = [=]() { send_meta(meta, communicator, receiving_rank, tag); };
  
    return stream.add(task);
  
  }

}

/** \brief            Send meta data extracted from a matrix
 *
 *  \param[in]        matrix
 *                    The matrix whose meta data is to be sent.
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
inline void send_meta(Dense<T>& matrix, MPI_Comm communicator,
                      int receiving_rank, int tag) {

  PROFILING_FUNCTION_HEADER

  MetaData meta(matrix);

  send_meta(meta, communicator, receiving_rank, tag);

}


/** \brief            Send meta data about a matrix asynchronously
 *
 *  \param[in]        matrix
 *                    The matrix whose meta data is to be sent.
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
 *  \returns          Ticket number for synchronizing on the stream.
 *
 *  \note             The call is buffered, that is the matrix can be deleted 
 *                    upon return from this function.
 */
template <typename T>
inline I_t send_meta_async(Dense<T>& matrix, MPI_Comm communicator,
                           int receiving_rank, int tag, Stream& stream) {

  PROFILING_FUNCTION_HEADER

  MetaData meta(matrix);

  return send_meta_async(meta, communicator, receiving_rank, tag, stream);

}

/** \brief            Receive meta data
 *
 *  \param[in]        meta
 *                    The meta data container to store the meta data in
 *
 *  \param[in]        communicator
 *                    MPI communicator for the transfer
 *
 *  \param[in]        sending_rank
 *                    MPI rank of the sending party
 *
 *  \param[in]        tag
 *                    Tag of the transfer (must match tag on receiver)
 */
inline void receive_meta(MetaData& meta, MPI_Comm communicator, 
                         int sending_rank, int tag) {

  PROFILING_FUNCTION_HEADER

# ifndef LINALG_NO_CHECKS
  check_rank(sending_rank, communicator, "receive_meta()");
# endif

  auto internal_tag = MPI_TAG_OFFSET * tag;

  MPI_Status status;

  auto error = mpi_recv(meta.data(), meta.size(), sending_rank, internal_tag,
                        communicator, &status);

# ifndef LINALG_NO_CHECKS
  if (error != MPI_SUCCESS) {

    // Construct a status and a corresponding exception
    auto my_status = construct_status(error, communicator, tag);

    excMPIError my_exception("receive_meta(): MPI error, ");

    my_exception.set_status(my_status);

    throw my_exception;

  }
# endif

}

/** \brief            Receive meta data
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
 *  \returns          The meta data
 */
inline MetaData receive_meta(MPI_Comm communicator, int sending_rank, int tag) {

  MetaData meta;

  receive_meta(meta, communicator, sending_rank, tag);

  return std::move(meta);

}

/** \brief            Receive meta data asynchronously
 *
 *  \param[in]        meta
 *                    The instance of meta to contain the meta data received.
 *
 *  \param[in]        communicator
 *                    MPI communicator for the transfer
 *
 *  \param[in]        sending_rank
 *                    MPI rank of the sending arty
 *
 *  \param[in]        tag
 *                    Tag of the transfer (must match tag on sender)
 *
 *  \param[in]        stream
 *                    Stream to use
 *
 *  \returns          Ticket number for synchronizing on the stream.
 *
 *  \note             The function is unbuffered. The caller is responsible to 
 *                    ensure that the meta data object is still alive when the 
 *                    stream executes the task. Failure to do so will lead to 
 *                    undefined behavior. Use with care.
 */
inline I_t receive_meta_async(MetaData& meta, MPI_Comm communicator,
                              int sending_rank, int tag, Stream& stream) {

  PROFILING_FUNCTION_HEADER

# ifndef LINALG_NO_CHECKS
  check_rank(sending_rank, communicator, "receive_meta_async()");
# endif

  if (stream.synchronous) {

    meta = receive_meta(communicator, sending_rank, tag);

    return 0;

  }

  if (stream.prefer_native) {

    auto internal_tag = MPI_TAG_OFFSET * tag;

    // Add task to stream
    auto stream_position = stream._add_mpi_tasks(1);

    auto request = &(stream.mpi_requests[stream_position]);

    auto error = mpi_irecv(meta.data(), meta.size(), sending_rank, 
                           internal_tag,
                           communicator, request);

    // Error handling happens when synchronizing on the stream.
    stream.mpi_statuses[stream_position] = construct_status(error, communicator,
                                                            tag);

    stream.mpi_synchronized = false;

    return 0;

  } else {

# ifndef LINALG_NO_CHECKS
    if (!stream.thread_alive) {
      throw excBadArgument("send_meta_async(): provided stream has no active "
                           "worker thread");
    }
# endif

    auto task = [=, &meta]() mutable { receive_meta(meta, communicator, 
                                                    sending_rank, tag); };

    return stream.add(task);
  
  }

}

/** \brief            Receive meta data about a matrix and save/apply it to
 *                    the supplied matrix
 *
 *  \param[in]        matrix
 *                    The matrix to which the meta data is to be applied.
 *
 *  \param[in]        communicator
 *                    MPI communicator for the transfer
 *
 *  \param[in]        sending_rank
 *                    MPI rank of the receiving party
 *
 *  \param[in]        tag
 *                    Tag of the transfer (must match tag on receiver)
 */
template <typename T>
inline void receive_meta(Dense<T>& matrix, MPI_Comm communicator,
                         int sending_rank, int tag) {

  PROFILING_FUNCTION_HEADER

# ifndef LINALG_NO_CHECKS
  check_rank(sending_rank, communicator, "receive_meta()");
# endif

  MetaData meta = receive_meta(communicator, sending_rank, tag);

  meta.apply(matrix);

}

/** \brief            Receive meta data about a matrix asynchronously and
 *                    save/apply it to the supplied matrix.
 *
 *  \param[in]        matrix
 *                    The matrix to which the meta data is to be applied.
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
 *  \returns          Ticket number for synchronizing on the stream.
 *
 *  \note             The function is unbuffered. The caller is responsible to 
 *                    ensure that the meta data object is still alive when the 
 *                    stream executes the task. Failure to do so will lead to 
 *                    undefined behavior. Use with care.
 */
template <typename T>
inline I_t receive_meta_async(Dense<T>& matrix, MPI_Comm communicator,
                              int sending_rank, int tag, Stream& stream) {

  PROFILING_FUNCTION_HEADER

# ifndef LINALG_NO_CHECKS
  check_rank(sending_rank, communicator, "receive_meta_async()");
# endif

  if (stream.synchronous) {

    receive_meta(matrix, communicator, sending_rank, tag);

    return 0;

  } else {

    auto task = [=, &matrix]() mutable { receive_meta(matrix, communicator, 
                                                      sending_rank, tag); };

    return stream.add(task);

  }

}

} /* namespace MPI */

} /* namespace LinAlg */

#endif /* HAVE_MPI */

#endif /* LINALG_MPI_SEND_RECV_META_H_ */
