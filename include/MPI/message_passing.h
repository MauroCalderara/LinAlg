/** \file
 *
 *  \brief            Inclusion of all MPI headers
 *
 *  \note             The name is chosen to avoid confusion with the system
 *                    mpi.h header
 *
 *  \date             Created:  Jul 15, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_MESSAGE_PASSING_H_
#define LINALG_MESSAGE_PASSING_H_

#ifdef HAVE_MPI

// Keep this in alphabetical order
#include "send_recv.h"
#include "send_recv_matrix.h"
#include "send_recv_meta.h"
#include "status.h"

#endif /* HAVE_MPI */

#endif /* LINALG_MESSAGE_PASSING_H_ */
