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

#include <functional>   // std::function

#include "types.h"
#include "profiling.h"
#include "utilities/buffer_helper.h"

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
 *
 *  \param[in]        stream
 *                    OPTIONAL: The stream to be used by the buffer. If a 
 *                    stream is specified it is assumed to be managed 
 *                    externally (started, stopped, cleared etc.). If no 
 *                    stream is specified, the buffer manages its own stream.
 *                    To create a buffer that works in the same thread as the 
 *                    one that constructs it, create a synchronous stream and 
 *                    pass it to the constructor as argument. Default: the 
 *                    buffer creates and manages its own, asynchronous stream.
 */
BufferHelper::BufferHelper(I_t size, I_t lookahead, BufferType type,
                           std::function<void(I_t)> loader,
                           std::function<void(I_t)> deleter)
                         : _manage_stream(true),
                           _size(size),
                           _lookahead(lookahead),
                           _loader(loader),
                           _deleter(deleter),
                           _type(type),
                           _initialized(false),
                           _last_requested(0) {

  PROFILING_FUNCTION_HEADER

  _stream = new Stream; 
  _buffer_tickets.resize(_size);
  _buffer_status.resize(_size);
  for (auto& element : _buffer_status) { element = 0; }

}
/** \overload
 */
BufferHelper::BufferHelper(I_t size, I_t lookahead, BufferType type,
                           std::function<void(I_t)> loader,
                           std::function<void(I_t)> deleter, Stream& stream)
                         : _stream(&stream),
                           _manage_stream(false),
                           _size(size),
                           _lookahead(lookahead),
                           _loader(loader),
                           _deleter(deleter),
                           _type(type),
                           _initialized(false),
                           _last_requested(0) {

  PROFILING_FUNCTION_HEADER

  _buffer_tickets.resize(_size);
  _buffer_status.resize(_size);
  for (auto& element : _buffer_status) { element = 0; }

}

/** \brief            Destructor
 */
BufferHelper::~BufferHelper() {

  PROFILING_FUNCTION_HEADER

  flush();

}

/** \brief            Flush/empty the buffer, resetting it to the default
 *                    state. All transfers that have been requested are worked 
 *                    off, including the corresponding deleter. Transfers that 
 *                    have not yet been requested (either by the user or by 
 *                    prefetching) are left as they are.
 *                    
 *  \note             In order to ensure all tasks are performed in the order 
 *                    of their queueing in the stream, the deletion of the 
 *                    last requested element is also performed via the stream 
 *                    of the buffer. Thus if the buffer uses a shared stream 
 *                    and operations unrelated to the buffer have been queued 
 *                    in said shared stream, all tasks that have been queued 
 *                    before the call to flush() are synchronized as well.
 */
void BufferHelper::flush() {

  PROFILING_FUNCTION_HEADER

  if (!_initialized) return;

  if (_type == BufferType::TwoPass || _type == BufferType::OnePass) {

    // For all requests either in flight (status == 1) or already synchronized
    // (status == 2) queue the corresponding deleter and synchronize on the 
    // last of them. This ensures that deleters are called in the right order 
    // and after the corresponding loaders.
    I_t last_deleter_ticket = 0;
    if (_direction == BufferDirection::increasing) {

      for (I_t i = 0; i < _size; ++i) {

        if (_buffer_status[i] > 0) {

          last_deleter_ticket = _stream->add(std::bind(_deleter, i));

          verbose_print("added _deleter(%d) -> ticket=%d\n", i, 
                        last_deleter_ticket); 

        }

      }

    } else if (_direction == BufferDirection::decreasing) {

      // i - 1 covers all elements of _buffer_status
      for (I_t i = _size; i > 0; --i) {

        if (_buffer_status[i - 1] > 0) {

          last_deleter_ticket = _stream->add(std::bind(_deleter, i - 1));

          verbose_print("added _deleter(%d) -> ticket=%d\n", i - 1, 
                        last_deleter_ticket); 

        } 

      }

    }

    verbose_print("synchronizing ticket %d\n", last_deleter_ticket);
    _stream->sync(last_deleter_ticket);

    for (auto& element : _buffer_status) { element = 0; }

  }

  if (_manage_stream) delete _stream;
  _initialized = false;
  _last_requested = 0;

}

/** \brief            Wait for an element in the buffer to be loaded
 *
 *  \param[in]        n
 *                    The element to wait for.
 */
void BufferHelper::wait(I_t n) {

  PROFILING_FUNCTION_HEADER

  if (_type == BufferType::OnePass || _type == BufferType::TwoPass) {

    bool two_pass = (_type == BufferType::OnePass) ? false : true;

    if (!_initialized) {

      // Set direction in which to buffer the next elements
      if (n == 0) {

        _direction = BufferDirection::increasing;
        _last_requested = -1;
        if (_manage_stream) {
          _stream->start_thread();
        }
        _initialized = true;

      } else if (n == _size - 1) {

        _direction = BufferDirection::decreasing;
        _last_requested = _size;
        if (_manage_stream) {
          _stream->start_thread();
        }
        _initialized = true;

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
      } else if (_direction == BufferDirection::decreasing) {
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
        printf("BUFFER: WARNING: element %d still in deletion when waiting\n", 
               n);
        verbose_print("synchronizing ticket %d\n", _buffer_tickets[n]);
        _stream->sync(_buffer_tickets[n]);
        _buffer_status[n] = 0;
      }

      // If the requested one is not in flight, make it fly
      if (_buffer_status[n] == 0) {
        printf("BUFFER: WARNING: element %d not in flight when waiting\n", n);
        _buffer_tickets[n] = _stream->add(std::bind(_loader, n));
        verbose_print("added _loader(%d) -> ticket=%d\n", n,
                      _buffer_tickets[n]);
        _buffer_status[n] = 1;
      }

      // If it is in flight, wait for it
      if (_buffer_status[n] == 1) {
        verbose_print("synchronizing ticket %d\n", _buffer_tickets[n]);
        _stream->sync(_buffer_tickets[n]);
        _buffer_status[n] = 2;
      }

      // Prefetch the next ones
      for (I_t i = n; (i < n + _lookahead + 1) && (i < _size); ++i) {
        if (_buffer_status[i] == 0) {
          _buffer_tickets[i] = _stream->add(std::bind(_loader, i));
          verbose_print("added _loader(%d) -> ticket=%d\n", i, 
                        _buffer_tickets[i]);
          _buffer_status[i] = 1;
        }
      }

      // Remove the element that has been requested previously (unless we're
      // at the beginning of the buffer (_last_requested == -1) OR a TwoPass
      // buffer reaching the end)
      if (_last_requested != -1 &&
          !((n > _size - _lookahead) && two_pass)) {

        _buffer_status[_last_requested] = -1;
        _buffer_tickets[_last_requested] = _stream->add(std::bind(_deleter, 
                                                            _last_requested));
        verbose_print("added _deleter(%d) -> ticket=%d\n", _last_requested, 
                      _buffer_tickets[_last_requested]);
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
        printf("BUFFER: WARNING: element %d still in deletion when waiting\n", 
               n);
        verbose_print("synchronizing ticket %d\n", _buffer_tickets[n]);
        _stream->sync(_buffer_tickets[n]);
        _buffer_status[n] = 0;
      }

      // If the requested one is not in flight, make it fly
      if (_buffer_status[n] == 0) {
        printf("BUFFER: WARNING: element %d not in flight when waiting\n", n);
        _buffer_tickets[n] = _stream->add(std::bind(_loader, n));
        verbose_print("added _loader(%d) -> ticket=%d\n", n, 
                      _buffer_tickets[n]);
        _buffer_status[n] = 1;
      }

      // If it is in flight, wait for it
      if (_buffer_status[n] == 1) {
        verbose_print("synchronizing ticket %d\n", _buffer_tickets[n]);
        _stream->sync(_buffer_tickets[n]);
        _buffer_status[n] = 2;
      }

      // Prefetch the next ones
      for (I_t i = n; (i > n - _lookahead - 1) && (i >= 0); --i) {
        if (_buffer_status[i] == 0) {
          _buffer_tickets[i] = _stream->add(std::bind(_loader, i));
          verbose_print("added _loader(%d) -> ticket=%d\n", i, 
                        _buffer_tickets[i]);
          _buffer_status[i] = 1;
        }
      }

      // Remove the element that has been requested previously (unless we're
      // at the end of the buffer (_last_requested == _size)) OR a TwoPass
      // buffer reaching the beginning)
      if (_last_requested != _size &&
          !((n < _lookahead) && two_pass)) {
        _buffer_status[_last_requested] = -1;
        _buffer_tickets[_last_requested] = _stream->add(std::bind(_deleter,
                                                             _last_requested));
        verbose_print("added _deleter(%d) -> ticket=%d\n", _last_requested, 
                      _buffer_tickets[_last_requested]);
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

#ifdef BUFFER_HELPER_DISPLAY
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

  PROFILING_FUNCTION_HEADER

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
      _last_requested = -1;
      if (_manage_stream) _stream->start_thread();
      _initialized = true;

      // Preload the next buffers
      for (I_t i = 0; (i < _lookahead + 1) && (i < _size); ++i) {
        if (_buffer_status[i] == 0) {
          _buffer_tickets[i] = _stream->add(std::bind(_loader, i));
          verbose_print("added _loader(%d) -> ticket=%d\n", i, 
                        _buffer_tickets[i]);
          _buffer_status[i] = 1;
        }
#ifndef LINALG_NO_CHECKS
        else {
          throw excBufferHelper("BufferHelper.preload(): status of element %d "
                                "is nonzero");
        }
#endif
      }

    } else if (_direction == BufferDirection::decreasing) {

      _last_requested = _size;
      if (_manage_stream) _stream->start_thread();
      _initialized = true;

      for (I_t i = _size - 1; (i > _size - 1 - _lookahead - 1) && (i >= 0);
           --i) {
        if (_buffer_status[i] == 0) {
          _buffer_tickets[i] = _stream->add(std::bind(_loader, i));
          verbose_print("added _loader(%d) -> ticket=%d\n", i,
                        _buffer_tickets[i]);
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
