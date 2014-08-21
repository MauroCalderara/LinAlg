/** \file             send_recv_meta.h
 *
 *  \brief            Routines to send and receive matrix metadata via MPI
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

#ifdef HAVE_MPI

#include <mpi.h>      // all MPI stuff

#include <vector>     // std::vector
#include <string>     // std::string
#include <sstream>    // std::stringstream
#include <cstdio>     // std::printf

#include "types.h"
#include "exceptions.h"
#include "streams.h"
#include "metadata.h"   // LinAlg::MetaData
#include "MPI/send_recv.h"

#include "dense.h"

namespace LinAlg {

namespace MPI {

/*  \brief            Send meta data
 *
 *  See MPI/send_recv_meta.cc
 */
void send_meta(MetaData& meta, MPI_Comm communicator, int receiving_rank,
               int tag);

/*  \brief            Send meta data asynchronously
 *
 *  See MPI/send_recv_meta.cc
 */
void send_meta_async(MetaData meta, MPI_Comm communicator, int receiving_rank,
                     int tag, MPIStream& stream);
/** \overload
 */
inline MPIStream send_meta_async(MetaData meta, MPI_Comm communicator,
                                 int receiving_rank, int tag) {

  MPIStream stream;

  send_meta_async(meta, communicator, receiving_rank, tag, stream);

  return std::move(stream);

};

/** \brief            Send meta data about a matrix
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
 *                    OPTIONAL: stream to use. If none is specified, the
 *                    function returns one that can be used to synchronize on
 *                    the operation.
 *
 *  \returns          stream
 *                    OPTIONAL: If no stream is specified, the function
 *                    returns one to synchronize the operation.
 */
template <typename T>
inline void send_meta_async(Dense<T>& matrix, MPI_Comm communicator,
                            int receiving_rank, int tag, MPIStream& stream) {
  MetaData meta(matrix);
  send_meta_async(meta, communicator, receiving_rank, tag, stream);
}
/** \overload
 */
template <typename T>
inline MPIStream send_meta_async(Dense<T>& matrix, MPI_Comm communicator,
                            int receiving_rank, int tag) {

  MPIStream stream;

  MetaData meta(matrix);

  send_meta_async(meta, communicator, receiving_rank, tag, stream);

  return std::move(stream);

}

/*  \brief            Receive meta data
 *
 *  See MPI/send_recv_meta.cc
 */
MetaData receive_meta(MPI_Comm communicator, int receiving_rank, int tag);

/*  \brief            Receive meta data asynchronously
 *
 *  See MPI/send_recv_meta.cc
 */
void receive_meta_async(MetaData& meta, MPI_Comm communicator,
                        int sending_rank, int tag, MPIStream& stream);
/** \overload
 */
inline MPIStream receive_meta_async(MetaData& meta, MPI_Comm communicator,
                                    int sending_rank, int tag) {

  MPIStream stream;

  receive_meta_async(meta, communicator, sending_rank, tag, stream);

  return std::move(stream);

};


/** \brief            Receive meta data about a matrix and save/apply it to
 *                    the supplied matrix
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
template <typename T>
inline void receive_meta(Dense<T>& matrix, MPI_Comm communicator,
                         int sending_rank, int tag) {

  MetaData meta = receive_meta(communicator, sending_rank, tag);

  meta.apply(matrix);

}

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

} /* namespace MPI */

} /* namespace LinAlg */

#endif /* HAVE_MPI */

#endif /* LINALG_MPI_SEND_RECV_META_H_ */
