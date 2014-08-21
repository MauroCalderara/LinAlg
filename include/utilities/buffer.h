/** \file
 *
 *  \brief            A memory buffer
 *
 *  \date             Created:  Aug 10, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_UTILITIES_BUFFER_H_
#define LINALG_UTILITIES_BUFFER_H_

#include <vector>     // std::vector
#include <functional> // std::function
#include <future>     // std::future

#include "types.h"
#include "exceptions.h"
#include "streams.h"

#ifndef DOXYGEN_SKIP
// If BUFFER_DISPLAY is defined, the buffer will print it's status during
// operations
#define BUFFER_DISPLAY 0
#ifdef BUFFER_DISPLAY
#include <iostream>     // std::cout
#endif
#endif /* DOXYGEN_SKIP */

namespace LinAlg {

namespace Utilities {

/** \brief            A simple class for buffer management.
 */
struct BufferHelper {

  BufferHelper() {};
  BufferHelper(I_t size, I_t lookahead, BufferType type,
               std::function<void(I_t)> loader,
               std::function<void(I_t)> deleter);

  void clear();
  void wait(I_t n);
  void preload(BufferDirection direction);

#ifndef DOXYGEN_SKIP
  // The idea being that you don't need to access these.
  I_t _size;
  I_t _lookahead;
  std::function<void(I_t)> _loader;
  std::function<void(I_t)> _deleter;
  BufferType _type;
  BufferDirection _direction;

  bool _initialized;
  int  _last_requested;
  std::vector<std::future<void>> _buffer_queue;
  std::vector<int> _buffer_status;    // -1 deleting, 0 clear,
                                      // 1 in flight, 2 loaded
#endif

};

} /* namespace LinAlg::Utilities */

} /* namespace LinAlg */


#endif /* LINALG_UTILITIES_BUFFER_H_ */
