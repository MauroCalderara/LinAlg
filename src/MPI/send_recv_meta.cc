/** \file
 *
 *  \brief            Sending/receiving matrix metadata via MPI
 *
 *  \date             Created:  Jul 15, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifdef HAVE_MPI

#include <mpi.h>      // all MPI stuff

#include "exceptions.h"
#include "streams.h"
#include "MPI/send_recv.h"      // The function overloads for MPI_*send/*recv
#include "MPI/status.h"
#include "MPI/send_recv_meta.h"

namespace LinAlg {

namespace MPI {

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
void send_meta(MetaData& meta, MPI_Comm communicator, int receiving_rank,
               int tag) {

  auto internal_tag = MPI_TAG_OFFSET * tag;

  auto error = mpi_send(meta.data(), meta.size(), receiving_rank, internal_tag,
                        communicator);

#ifndef LINALG_NO_CHECKS
  if (error != MPI_SUCCESS) {

    // Construct a status and a corresponding exception
    auto my_status = construct_status(error, communicator, tag);

    excMPIError my_exception("send_meta(): MPI error, ");

    my_exception.set_status(my_status);

    throw my_exception;

  }
#endif

};

/** \brief            Send meta data about a matrix asynchronously
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
 *
 *  \param[in]        stream
 *                    Stream to use
 */
void send_meta_async(MetaData meta, MPI_Comm communicator,
                     int receiving_rank, int tag, MPIStream& stream) {

  // Note: we don't pass by reference to ensure we have a valid copy of the
  // meta data.

  if (stream.synchronous_operation) {

    send_meta(meta, communicator, receiving_rank, tag);

    return;

  }

  auto internal_tag = MPI_TAG_OFFSET * tag;

  // Add task to stream
  auto stream_position = stream.add_operations(1);

  auto request = &stream.requests[stream_position];

  stream.synchronized = false;

  auto error = mpi_isend(meta.data(), meta.size(), receiving_rank,
                         internal_tag, communicator, request);

  // Error handling happens when synchronizing on the stream.
  stream.statuses[stream_position] = construct_status(error, communicator,
                                                       tag);
};

/** \brief            Receive meta data
 *
 *  \param[in]        communicator
 *                    MPI communicator for the transfer
 *
 *  \param[in]        sending_rank
 *                    MPI rank of the sending party
 *
 *  \param[in]        tag
 *                    Tag of the transfer (must match tag on receiver)
 *
 *  \returns          meta
 *                    The meta data
 */
MetaData receive_meta(MPI_Comm communicator, int sending_rank, int tag) {

  MetaData meta;

  auto internal_tag = MPI_TAG_OFFSET * tag;

  MPI_Status status;

  auto error = mpi_recv(meta.data(), meta.size(), sending_rank, internal_tag,
                        communicator, &status);

#ifndef LINALG_NO_CHECKS
  if (error != MPI_SUCCESS) {

    // Construct a status and a corresponding exception
    auto my_status = construct_status(error, communicator, tag);

    excMPIError my_exception("receive_meta(): MPI error, ");

    my_exception.set_status(my_status);

    throw my_exception;

  }
#endif

  return std::move(meta);

};

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
 *                    Tag of the transfer (must match tag on receiver)
 *
 *  \param[in]        stream
 *                    Stream to use
 */
void receive_meta_async(MetaData& meta, MPI_Comm communicator,
                        int sending_rank, int tag, MPIStream& stream) {

  if (stream.synchronous_operation) {

    meta = receive_meta(communicator, sending_rank, tag);

    return;

  }

  auto internal_tag = MPI_TAG_OFFSET * tag;

  // Add task to stream
  auto stream_position = stream.add_operations(1);

  auto request = &stream.requests[stream_position];

  stream.synchronized = false;

  auto error = mpi_irecv(meta.data(), meta.size(), sending_rank, internal_tag,
                         communicator, request);

  // Error handling happens when synchronizing on the stream.
  stream.statuses[stream_position] = construct_status(error, communicator,
                                                       tag);

};

} /* namespace MPI */

} /* namespace LinAlg */

#endif /* HAVE_MPI */
