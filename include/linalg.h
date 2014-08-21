/** \file             linalg.h
 *
 *  \brief            A meta-header that includes all other .h files of the
 *                    linalg library.
 *
 *  \date             Created:  Jul  3, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_LINALG_H_
#define LINALG_LINALG_H_

#include "types.h"
#include "exceptions.h"
#include "streams.h"
#include "matrix.h"
#include "dense.h"
#include "sparse.h"
#include "abstract.h"
#include "metadata.h"

#ifdef HAVE_CUDA
#include "CUDA/cuda_helper.h"
#endif

#ifdef HAVE_MPI
#include "MPI/message_passing.h"
#endif

#include "utilities/utilities.h"

#include "fills.h"
#include "BLAS/blas.h"
#include "LAPACK/lapack.h"

#endif /* LINALG_LINALG_H_ */
