/** \file
 *
 *  \brief            Specialized members of Dense<T>
 *
 *  \date             Created:  Jul 12, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */

// There is not much to see here as dense.h is completely templated

#include "dense.h"

namespace LinAlg {

#ifndef DOXYGEN_SKIP
/*  \brief            Returns whether the matrix is complex or not
 *
 *  \return           True if the matrix is of complex data type (C_t, Z_t),
 *                    false otherwise.
 */
template <> inline bool Dense<C_t>::_is_complex() const { return true; }
template <> inline bool Dense<Z_t>::_is_complex() const { return true; }
#endif

} /* namespace LinAlg */


