/** \file
 *
 *  \brief            MPI_Status parsing
 *
 *  \date             Created:  Jul 15, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_MPI_STATUS_H_
#define LINALG_MPI_STATUS_H_

#include "../preprocessor.h"

#ifdef HAVE_MPI

# include <mpi.h>      // all MPI stuff

namespace LinAlg {

namespace MPI {

/*  \brief            Routine to construct an MPI_Status
 *
 *  See src/MPI/status.cc
 */
MPI_Status construct_status(int error_code, MPI_Comm communicator, int tag);

} /* namespace MPI */

} /* namespace LinAlg */

#endif /* HAVE_MPI */

#endif /* LINALG_MPI_STATUS_H_ */
