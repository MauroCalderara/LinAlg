/** \file             status.cc
 *
 *  \brief            Routine to construct an MPI status
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

#include "MPI/status.h"

namespace LinAlg {

namespace MPI {

/** \brief            Routine to construct an MPI_Status
 *
 *  This is to be used for failing MPI routines that don't create an MPI_Status
 *  which can be used for the exception themselves.
 *
 *  \param[in]        error_code
 *                    Return value of the MPI call
 *
 *  \param[in]        communicator
 *                    Communicator used
 *
 *  \param[in]        tag
 *                    Tag used
 *
 *  \returns          An MPI_Status struct
 */
MPI_Status construct_status(int error_code, MPI_Comm communicator, int tag) {

  MPI_Status status;

  int my_rank;
  MPI_Comm_rank(communicator, &my_rank);

  status.MPI_SOURCE = my_rank;

  status.MPI_TAG = tag;

  status.MPI_ERROR = error_code;

  return std::move(status);

};

} /* namespace MPI */

} /* namespace LinAlg */

#endif /* HAVE_MPI */
