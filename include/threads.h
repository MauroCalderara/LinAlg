/** \file
 *
 *  \brief            Abstractions for handling threading primitives. Most of 
 *                    this file is typedefs when using C++11 threads or simple 
 *                    classes and functions that mimick C++11 threading 
 *                    primitives using POSIX threading primitives
 *
 *  \date             Created:  Nov 30, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_THREADS_H_
#define LINALG_THREADS_H_

#include "preprocessor.h"

#ifdef USE_POSIX_THREADS
# include <pthread.h>
#else // C++11 threading primitives
# include <thread>
# include <mutex>
#endif

namespace LinAlg {

namespace Threads {

/** \brief            Simple wrapper for a mutex
 */
#ifdef USE_POSIX_THREADS
struct Mutex {

  pthread_mutex_t mutex;

  Mutex() {
    auto error = pthread_mutex_init(&mutex, NULL);
# ifndef LINALG_NO_CHECKS
    if (error != 0) {
      throw excSystemError("Mutex(): Unable to initialize POSIX mutex, error "
                           "= %d", error);
    }
# endif
  }

  ~Mutex() {
    auto error = pthread_mutex_destroy(&mutex);
# ifndef LINALG_NO_CHECKS
    //if (error != 0) {
    //  throw excSystemError("Mutex(): Unable to destroy POSIX mutex, error "
    //                       "= %d", error);
    //}
# endif
  }

};
#else 
// C++11 std::mutex already provides a class
typedef std::mutex Mutex;
#endif

/** \brief            Simple RAII wrapper for locking mutexes
 */
#ifdef USE_POSIX_THREADS
struct MutexLock {

  pthread_mutex_t&  mutex;
  bool              active;

  MutexLock(Mutex& mutex_) : mutex(mutex_.mutex), active(false) {
    pthread_mutex_lock(&mutex);
    active = true;
  }

  ~MutexLock() {
    if (active) pthread_mutex_unlock(&mutex);
  }

  inline void unlock() { 
    if (active) { 
      pthread_mutex_unlock(&mutex);
      active = false; 
    }
  }

};
#else 
// C++11 std::unique_lock already provides RAII
typedef std::unique_lock<std::mutex> MutexLock;
#endif

/** \brief            Simple RAII wrapper for a condition variable
 */
#ifdef USE_POSIX_THREADS
struct ConditionVariable {

  // This is supposed to make a pthread condition variable look roughly like a 
  // C++ condition variable

  pthread_cond_t condition_variable;

  ConditionVariable() {
    auto error = pthread_cond_init(&condition_variable, NULL);
# ifndef LINALG_NO_CHECKS
    if (error != 0) {
      throw excSystemError("ConditionVariable(): Unable to initialize POSIX "
                           "condition variable, error = %d", error);
    }
# endif
  }

  ~ConditionVariable() {
    auto error = pthread_cond_destroy(&condition_variable);
# ifndef LINALG_NO_CHECKS
    if (error != 0) {
      throw excSystemError("ConditionVariable(): Unable to destroy POSIX "
                           "condition variable, error = %d", error);
    }
# endif
  }

  inline void wait(MutexLock lock, std::function<bool()> condition) {
  
    while (!condition()) pthread_cond_wait(&condition_variable, &(lock.mutex));
  
  }

  inline void notify_all() { pthread_cond_signal(&condition_variable); }

};
#else
typedef std::condition_variable ConditionVariable;
#endif

} /* namespace LinAlg::Utilities */

} /* namespace LinAlg */

#endif /* LINALG_UTILITIES_THREADING_H_ */
