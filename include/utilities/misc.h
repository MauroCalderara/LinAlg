/** \file
 *
 *  \brief            Routines that don't fit elsewhere.
 *
 *  \date             Created:  Jul 17, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcaldrara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_UTILITIES_MISC_H_
#define LINALG_UTILITIES_MISC_H_

#include <fstream>    // std::fstream
#include <limits>     // std::numeric_limits

namespace LinAlg {

namespace Utilities {

/** \brief            A routine that jumps to a specific line in a fstream
 *
 *  \param[in,out]    stream
 *                    The fstream on which to advance to the requested line.
 *
 *  \param[in]        line
 *                    The line to advance to. The next getline call on the
 *                    stream will return the contents of line number \<line\>.
 */
inline void goto_line(std::ifstream& stream, unsigned int line) {

  stream.seekg(std::ios::beg);

  for (unsigned int i = 0; i < line - 1; ++i) {

    stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  }

};

} /* namespace Utilities */

} /* namespace LinAlg */

#endif /* LINALG_UTILITIES_MISC_H_ */
