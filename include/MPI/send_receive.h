/** \file
 *
 *  \brief            Overloads for MPI_Xsend() and MPI_Xrecv()
 *
 *  \date             Created:  Jul 15, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_MPI_SEND_RECV_H_
#define LINALG_MPI_SEND_RECV_H_

#include "../preprocessor.h"

#ifdef HAVE_MPI

# include <mpi.h>      // all MPI stuff

# ifndef DOXYGEN_SKIP
#   define MPI_TAG_OFFSET 15
# endif

namespace LinAlg {

namespace MPI {

// Overloaded MPI functions for the required data types we generate templates
// for

/** \brief            Wrapper for MPI_Send
 *
 *  See [MPI_Send](http://www.mpich.org/static/docs/v3.1/www3/MPI_Recv.html)
 */
inline int mpi_send(S_t* array, I_t size, int receiving_rank, int tag,
                    MPI_Comm communicator) {

  PROFILING_FUNCTION_HEADER

  return MPI_Send(array, size, MPI_FLOAT, receiving_rank, tag, communicator);

}
/** \overload
 */
inline int mpi_send(D_t* array, I_t size, int receiving_rank, int tag,
                    MPI_Comm communicator) {

  PROFILING_FUNCTION_HEADER

  return MPI_Send(array, size, MPI_DOUBLE, receiving_rank, tag, communicator);

}
/** \overload
 */
inline int mpi_send(C_t* array, I_t size, int receiving_rank, int tag,
                    MPI_Comm communicator) {

  PROFILING_FUNCTION_HEADER

  return MPI_Send(array, size, MPI_C_FLOAT_COMPLEX, receiving_rank, tag,
                  communicator);

}
/** \overload
 */
inline int mpi_send(Z_t* array, I_t size, int receiving_rank, int tag,
                    MPI_Comm communicator) {

  PROFILING_FUNCTION_HEADER

  return MPI_Send(array, size, MPI_C_DOUBLE_COMPLEX, receiving_rank, tag,
                  communicator);

}
/** \overload
 */
inline int mpi_send(I_t* array, I_t size, int receiving_rank, int tag,
                    MPI_Comm communicator) {

  return MPI_Send(array, size, LINALG_MPI_INT, receiving_rank, tag,
                  communicator);

}

/** \brief            Wrapper for MPI_Recv
 *
 *  See [MPI_Send](http://www.mpich.org/static/docs/v3.1/www3/MPI_Recv.html)
 */
inline int mpi_recv(S_t* array, I_t size, int sending_rank, int tag,
                    MPI_Comm communicator, MPI_Status* status) {

  PROFILING_FUNCTION_HEADER

  return MPI_Recv(array, size, MPI_FLOAT, sending_rank, tag, communicator,
                  status);

}
/** \overload
 */
inline int mpi_recv(D_t* array, I_t size, int sending_rank, int tag,
                    MPI_Comm communicator, MPI_Status* status) {

  PROFILING_FUNCTION_HEADER

  return MPI_Recv(array, size, MPI_DOUBLE, sending_rank, tag, communicator,
                  status);

}
/** \overload
 */
inline int mpi_recv(C_t* array, I_t size, int sending_rank, int tag,
                    MPI_Comm communicator, MPI_Status* status) {

  PROFILING_FUNCTION_HEADER

  return MPI_Recv(array, size, MPI_C_FLOAT_COMPLEX, sending_rank, tag,
                  communicator, status);

}
/** \overload
 */
inline int mpi_recv(Z_t* array, I_t size, int sending_rank, int tag,
                    MPI_Comm communicator, MPI_Status* status) {

  PROFILING_FUNCTION_HEADER

  return MPI_Recv(array, size, MPI_C_DOUBLE_COMPLEX, sending_rank, tag,
                  communicator, status);

}
/** \overload
 */
inline int mpi_recv(I_t* array, I_t size, int sending_rank, int tag,
                    MPI_Comm communicator, MPI_Status* status) {

  PROFILING_FUNCTION_HEADER

  return MPI_Recv(array, size, LINALG_MPI_INT, sending_rank, tag, communicator,
                  status);

}

/** \brief            Wrapper for MPI_Isend
 *
 *  See [MPI_Isend](http://www.mpich.org/static/docs/v3.1/www3/MPI_Isend.html)
 */
inline int mpi_isend(S_t* array, I_t size, int receiving_rank, int tag,
                     MPI_Comm communicator, MPI_Request* request) {

  PROFILING_FUNCTION_HEADER

  return MPI_Isend(array, size, MPI_FLOAT, receiving_rank, tag, communicator,
                   request);

}
/** \overload
 */
inline int mpi_isend(D_t* array, I_t size, int receiving_rank, int tag,
                     MPI_Comm communicator, MPI_Request* request) {

  PROFILING_FUNCTION_HEADER

  return MPI_Isend(array, size, MPI_DOUBLE, receiving_rank, tag, communicator,
                   request);

}
/** \overload
 */
inline int mpi_isend(C_t* array, I_t size, int receiving_rank, int tag,
                     MPI_Comm communicator, MPI_Request* request) {

  PROFILING_FUNCTION_HEADER

  return MPI_Isend(array, size, MPI_C_FLOAT_COMPLEX, receiving_rank, tag,
                   communicator, request);

}
/** \overload
 */
inline int mpi_isend(Z_t* array, I_t size, int receiving_rank, int tag,
                     MPI_Comm communicator, MPI_Request* request) {

  PROFILING_FUNCTION_HEADER

  return MPI_Isend(array, size, MPI_C_DOUBLE_COMPLEX, receiving_rank, tag,
                   communicator, request);

}
/** \overload
 */
inline int mpi_isend(I_t* array, I_t size, int receiving_rank, int tag,
                     MPI_Comm communicator, MPI_Request* request) {

  PROFILING_FUNCTION_HEADER

  return MPI_Isend(array, size, LINALG_MPI_INT, receiving_rank, tag,
                   communicator, request);

}

/** \brief            Wrapper for MPI_Irecv
 *
 *  See [MPI_Irecv](http://www.mpich.org/static/docs/v3.1/www3/MPI_Irecv.html)
 */
inline int mpi_irecv(S_t* array, I_t size, int sending_rank, int tag,
                     MPI_Comm communicator, MPI_Request* request) {

  PROFILING_FUNCTION_HEADER

  return MPI_Irecv(array, size, MPI_FLOAT, sending_rank, tag, communicator,
                   request);

}
/** \overload
 */
inline int mpi_irecv(D_t* array, I_t size, int sending_rank, int tag,
                     MPI_Comm communicator, MPI_Request* request) {

  PROFILING_FUNCTION_HEADER

  return MPI_Irecv(array, size, MPI_DOUBLE, sending_rank, tag, communicator,
                   request);

}
/** \overload
 */
inline int mpi_irecv(C_t* array, I_t size, int sending_rank, int tag,
                     MPI_Comm communicator, MPI_Request* request) {

  PROFILING_FUNCTION_HEADER

  return MPI_Irecv(array, size, MPI_C_FLOAT_COMPLEX, sending_rank, tag,
                   communicator, request);

}
/** \overload
 */
inline int mpi_irecv(Z_t* array, I_t size, int sending_rank, int tag,
                     MPI_Comm communicator, MPI_Request* request) {

  PROFILING_FUNCTION_HEADER

  return MPI_Irecv(array, size, MPI_C_DOUBLE_COMPLEX, sending_rank, tag,
                   communicator, request);

}
/** \overload
 */
inline int mpi_irecv(I_t* array, I_t size, int sending_rank, int tag,
                     MPI_Comm communicator, MPI_Request* request) {

  PROFILING_FUNCTION_HEADER

  return MPI_Irecv(array, size, LINALG_MPI_INT, sending_rank, tag, communicator,
                   request);

}


} /* namespace MPI */

} /* namespace LinAlg */

#endif /* HAVE_MPI */

#endif /* LINALG_MPI_SEND_RECV_H_ */
