/** \file
 *
 *  \brief            A memory buffer
 *
 *  \date             Created:  Jul 12, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */

#include <future>       // std::future, std::async
#include <functional>   // std::function

#include "types.h"
#include "utilities/buffer.h"

namespace LinAlg {

namespace Utilities {

/** \brief            Create a new buffer
 *
 *  \param[in]        size
 *                    Number of elements the buffer holds. Accessing the
 *                    elements is done in C-style indexing.
 *
 *  \param[in]        lookahead
 *                    Number of elements the buffer should try to load in
 *                    advance.
 *
 *  \param[in]        type
 *                    Type of the buffer.
 *
 *  \param[in]        loader
 *                    Functor the buffer uses to load an element from the
 *                    buffer. The functor must not return a value and take
 *                    the index of the element to fetch as single argument.
 *
 *  \param[in]        deleter
 *                    Functor the buffer uses to unload an element from the
 *                    buffer. The functor must not return a value and take
 *                    the index of the element to unload as single argument.
 */
BufferHelper::BufferHelper(I_t size, I_t lookahead, BufferType type,
                           std::function<void(I_t)> loader,
                           std::function<void(I_t)> deleter)
                         : _size(size),
                           _lookahead(lookahead),
                           _loader(loader),
                           _deleter(deleter),
                           _type(type) {
  _initialized = false;
  _last_requested = 0;
  _buffer_queue.resize(_size);
  _buffer_status.resize(_size);
  for (auto& e : _buffer_status) { e = 0; }
}

/** \brief            Clear/empty the buffer, resetting it to the default
 *                    state
 */
void BufferHelper::clear() {

  if (!_initialized) return;

  if (_type == BufferType::TwoPass) {

    // For all requests (in the right order), wait till they're done and then
    // call the deleter.
    if (_direction == BufferDirection::increasing) {

      for (I_t i = 0; i < _size; ++i) {

        if (_buffer_status[i] != 0) {
          _buffer_queue[i].get();
          _deleter(i);
          _buffer_status[i] = 0;
        }

      }

    } else {

      for (I_t i = _size; i < 0; --i) {

        if (_buffer_status[i - 1] != 0) {
          _buffer_queue[i - 1].get();
          _deleter(i - 1);
          _buffer_status[i - 1] = 0;
        }

      }

    }

  }

  _initialized = false;
  _last_requested = 0;

}

/** \brief            Wait for an element in the buffer to be loaded
 *
 *  \param[in]        n
 *                    The element to wait for.
 */
void BufferHelper::wait(I_t n) {

  if (_type == BufferType::OnePass || _type == BufferType::TwoPass) {

    bool two_pass = (_type == BufferType::OnePass) ? false : true;

    if (!_initialized) {

      // Set direction in which to buffer the next elements
      if (n == 0) {

        _initialized = true;
        _direction = BufferDirection::increasing;
        _last_requested = -1;

      } else if (n == _size - 1) {

        _initialized = true;
        _direction = BufferDirection::decreasing;
        _last_requested = _size;

      } else {

#ifndef LINALG_NO_CHECKS
        throw excBufferHelper("BufferHelper.wait(): can only start either at 0 "
                              "or at %d", _size - 1);
#endif
      }

    }

    // Going from left to right (increasing indices)
    if (_direction == BufferDirection::increasing) {

#ifndef LINALG_NO_CHECKS
      // Check for invalid request
      if (two_pass) {
        if (n < _last_requested + 1 && n != 0) {
          throw excBufferHelper("BufferHelper.wait(): with this buffer type "
                                "(TwoPass) requests for previous blocks are "
                                "only valid after reaching the end of the "
                                "buffer");
        }
      } else {
        if (n < _last_requested + 1) {
          throw excBufferHelper("BufferHelper.wait(): with this buffer type "
                                "(OnePass) requests for previous blocks are "
                                "invalid");
        }
      }
#endif

      // If the requested element is being deleted, wait for the deletion to
      // complete
      if (_buffer_status[n] == -1) {
        printf("BUFFER: WARNING: element %d still in deletion when waiting\n", n);
        _buffer_queue[n].get();
        _buffer_status[n] = 0;
      }

      // If the requested one is not in flight, make it fly
      if (_buffer_status[n] == 0) {
        printf("BUFFER: WARNING: element %d not in flight when waiting\n", n);
        _buffer_queue[n] = std::async(std::launch::async, _loader, n);
        _buffer_status[n] = 1;
      }

      // If it is in flight, wait for it
      if (_buffer_status[n] == 1) {
        _buffer_queue[n].get();
        _buffer_status[n] = 2;
      }

      // Prefetch the next ones
      for (I_t i = n; (i < n + _lookahead + 1) && (i < _size); ++i) {
        if (_buffer_status[i] == 0) {
          _buffer_queue[i] = std::async(std::launch::async, _loader, i);
          _buffer_status[i] = 1;
        }
      }

      // Remove the element that has been requested previously (unless we're
      // at the beginning of the buffer (_last_requested == -1) OR a TwoPass
      // buffer reaching the end)
      if (_last_requested != -1 &&
          !((n > _size - _lookahead) && two_pass)) {

        _buffer_status[_last_requested] = -1;
        _buffer_queue[_last_requested] = std::async(std::launch::async,
                                                    _deleter,
                                                    _last_requested);
      }

      // Once we reached the end we change direction
      if (n == _size - 1) {
        _direction = BufferDirection::decreasing;
        _last_requested = _size;
      } else {
        _last_requested = n;
      }

    }

    // Decreasing
    else {

#ifndef LINALG_NO_CHECKS
      // Check for invalid request
      if (two_pass) {
        if (n > _last_requested - 1 && n != _size - 1) {
          throw excBufferHelper("BufferHelper.wait(): with this buffer type "
                                "(TwoPass) requests for previous blocks are "
                                "only valid after reaching the end of the "
                                "buffer");
        }
      } else {
        if (n > _last_requested - 1) {
          throw excBufferHelper("BufferHelper.wait(): with this buffer type "
                                "(OnePass) requests for previous blocks are "
                                "invalid");
        }
      }
#endif

      // If the requested element is being deleted, wait for the deletion to
      // complete
      if (_buffer_status[n] == -1) {
        printf("BUFFER: WARNING: element %d still in deletion when waiting\n", n);
        _buffer_queue[n].get();
        _buffer_status[n] = 0;
      }

      // If the requested one is not in flight, make it fly
      if (_buffer_status[n] == 0) {
        printf("BUFFER: WARNING: element %d not in flight when waiting\n", n);
        _buffer_queue[n] = std::async(std::launch::async, _loader, n);
        _buffer_status[n] = 1;
      }

      // If it is in flight, wait for it
      if (_buffer_status[n] == 1) {
        _buffer_queue[n].get();
        _buffer_status[n] = 2;
      }

      // Prefetch the next ones
      for (I_t i = n; (i > n - _lookahead - 1) && (i >= 0); --i) {
        if (_buffer_status[i] == 0) {
          _buffer_status[i] = 1;
          _buffer_queue[i] = std::async(std::launch::async, _loader, i);
        }
      }

      // Remove the element that has been requested previously (unless we're
      // at the end of the buffer (_last_requested == _size)) OR a TwoPass
      // buffer reaching the beginning)
      if (_last_requested != _size &&
          !((n < _lookahead) && two_pass)) {
        _buffer_status[_last_requested] = -1;
        _buffer_queue[_last_requested] = std::async(std::launch::async,
                                                    _deleter,
                                                    _last_requested);
      }

      // Once we reached the end we change direction
      if (n == 0) {
        _direction = BufferDirection::increasing;
        _last_requested = -1;
      } else {
        _last_requested = n;
      }

    }

  }

#ifdef BUFFER_DISPLAY
  // Visualization
  std::cout << "Buffer status: [|";
  for (const auto& e : _buffer_status) { std::cout << e << "|"; }
  std::cout << "]\n";
#endif

}


/** \brief            Preload elements into the buffer
 *
 *  This function is intended to 'warm up' the buffer before waiting for the
 *  first element.
 *
 *  \param[in]        direction
 *                    The direction in which to preload.
 *                    BufferDirection::increasing makes the buffer fetch the
 *                    elements 0, 1, ... \<prefetch\>,
 *                    BufferDirection::decreasing the elements size-1, size-2,
 *                    ... size - \<prefetch\>
 */
void BufferHelper::preload(BufferDirection direction) {

#ifndef LINALG_NO_CHECKS
  if (_initialized == true) {

    throw excBufferHelper("BufferHelper.preload(): can only preload on an "
                          "uninitialized buffer");

  }
#endif

  _direction = direction;

  if (_type == BufferType::OnePass || _type == BufferType::TwoPass) {

    if (_direction == BufferDirection::increasing) {

      // Initialize buffer
      _initialized = true;
      _last_requested = -1;

      // Preload the next buffers
      for (I_t i = 0; (i < _lookahead + 1) && (i < _size); ++i) {
        if (_buffer_status[i] == 0) {
          _buffer_queue[i] = std::async(_loader, i);
          _buffer_status[i] = 1;
        }
#ifndef LINALG_NO_CHECKS
        else {
          throw excBufferHelper("BufferHelper.preload(): status of element %d "
                                "is nonzero");
        }
#endif
      }

    } else {

      _initialized = true;
      _last_requested = _size;

      for (I_t i = _size - 1; (i > _size - 1 - _lookahead - 1) && (i >= 0);
           --i) {
        if (_buffer_status[i] == 0) {
          _buffer_queue[i] = std::async(std::launch::async, _loader, i);
          _buffer_status[i] = 1;
        }
#ifndef LINALG_NO_CHECKS
        else {
          throw excBufferHelper("BufferHelper.preload(): status of element %d "
                                "is nonzero");
        }
#endif
      }

    }

  }

}

} /* namespace LinAlg::Utilities */

} /* namespace LinAlg */
