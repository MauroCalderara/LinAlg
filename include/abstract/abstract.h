/** \file
 *
 *  \brief            Convenience bindings like multiply(), inv() and solve()
 *
 *  Organization of the namespace:
 *
 *    LinAlg
 *        functions like 'solve' and 'multiply' (abstract/\*.h)
 *
 *    LinAlg::BLAS
 *
 *        convenience bindings supporting different locations for Dense<T>
 *
 *    LinAlg::BLAS::\<backend\>
 *
 *        bindings for the backend
 *
 *    LinAlg::LAPACK
 *
 *        convenience bindings supporting different locations for Dense<T>
 *
 *    LinAlg::LAPACK::\<backend\>
 *
 *        bindings for the backend
 *
 *
 *  \date             Created:  Jul 23, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_ABSTRACT_H_
#define LINALG_ABSTRACT_H_

// Keep this in alphabetical order

#include "add.h"
#include "invert.h"
#include "multiply.h"
#include "solve.h"

#endif /* LINALG_ABSTRACT_H_ */
