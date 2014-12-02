/** \file
 *
 *  \brief            Streams (handling of concurrent execution)
 *
 *  \date             Created:  Aug 12, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcaldrara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_STREAMS_H_
#define LINALG_STREAMS_H_

#include "preprocessor.h"

#ifdef HAVE_CUDA
# include <cuda_runtime.h> // various CUDA routines
# include <cublas_v2.h>
# include <cusparse_v2.h>
# include "CUDA/cuda_checks.h"
#endif

#ifdef HAVE_MPI
# include <mpi.h>
#endif

#include <functional> // std::function
#include <queue>      // std::deque
#include <atomic>     // std::atomic

#include "types.h"
#include "profiling.h"
#include "threads.h"
#include "exceptions.h"
#include "dense.h"
#include "sparse.h"

namespace LinAlg {

/** \brief            Signals that even though a stream is provided, the
 *                    operation should be synchronous.
 */
enum class Streams { Synchronous };

/** \brief            StreamBase
 *
 *  A stream is a unit of asynchronous execution. The .sync() routine
 *  synchronizes with the execution of the stream. Some streams may be
 *  appendable (depending on the underlying implementation).
 *
 *  Streams are semantically similar to of C++11 std::futures with the
 *  std::future.get() routine replaced by Stream.sync()
 */
struct StreamBase {

  StreamBase() : synchronous_operation(false), synchronized(true) {}

  /** \brief          Synchronous stream constructor
   */
  StreamBase(Streams stream_spec) {

    synchronous_operation = (stream_spec == Streams::Synchronous) ? true :
                                                                    false;
    synchronized = true;

  }

  virtual ~StreamBase() {}

#ifndef DOXYGEN_SKIP
  virtual inline void sync() = 0;

  bool synchronous_operation;
  bool synchronized;
#endif

};

/** Generic stream:
 *
 *  Tasks are std::function<void()> objects. Each stream has one worker thread
 *  assigned that fetches tasks from the tail of a deque. Appending tasks is 
 *  pushing to the front to the queue. Tasks are processed in the order of 
 *  being pushed to the queue.
 */
struct Stream : StreamBase {

#ifndef DOXYGEN_SKIP

  Threads::Mutex                    lock;
  Threads::ConditionVariable        cv;

# ifdef USE_POSIX_THREADS
  pthread_t                         worker_thread;
  // Pthreads expect a C-style pointer to a void* function and support passing 
  // void* arguments to it. Thus unlike in the C++11 thread case below passing 
  // a std::function object doesn't work. So we create a static wrapper that 
  // expects a pointer to the class instance as argument, internally casts it 
  // to a Stream* pointer and then calls the corresponding worker function.  
  // This is neither elegant nor type safe but seems to be the idiomatic way 
  // of dealing with this situation in C++
  static void* worker_wrapper(void* instance) {
    ((Stream*)instance)->worker();
    return nullptr;
  }
# else /* use C++11 threads */
  std::thread                       worker_thread;
# endif /* USE_POSIX_THREADS */

  std::deque<std::function<void()>> queue;
  std::atomic<I_t> next_in_queue;
  bool thread_alive;

  inline void worker();
  I_t next_ticket;             // next_ticket = next_in_queue -> no work;
  bool terminate;

#endif /* not DOXYGEN_SKIP */

  Stream();
  Stream(Streams stream_spec);
  ~Stream();

  inline void start();
  inline I_t  add(std::function<void()> task);
  inline void sync(I_t ticket);
  inline void sync();
  inline void stop();
  inline void clear();

};

/** \brief            Constructor for a stream
 *
 *  Note that this constructor doesn't spawn a thread to avoid having 
 *  lingering threads for classes with streams as members. Use start() to 
 *  create the worker thread and 'activate' the queue.
 */
inline Stream::Stream() : StreamBase() {

  PROFILING_FUNCTION_HEADER

  // These here can't be in the initializer list since we have delegating 
  // constructors
  thread_alive = false;
  next_ticket  = 0;
  terminate    = false;

  next_in_queue.store(0);

}

/** \brief            Constructor for a stream without a worker thread
 *                    (causes synchronous execution on all routines that use
 *                    this stream)
 */
// Note: we delegate to Stream() and not to StreamBase(stream_spec) in order 
// to avoid reproducing the constructor above
inline Stream::Stream(Streams stream_spec) : Stream() {

  synchronous_operation = (stream_spec == Streams::Synchronous) ? true : false;

}

/** \brief            Enables the stream by creating the worker thread
 *
 *  For synchronous streams, no worker thread is spawned.
 */
inline void Stream::start() {

  PROFILING_FUNCTION_HEADER

  if (synchronous_operation) {
  
    return;
  
  } else {

    // Start the worker thread
#ifdef USE_POSIX_THREADS

    // See in the class definition why we need the wrapper for the worker 
    // function
    printf("CREATING POSIX THREAD\n");
    auto error = pthread_create(&worker_thread, NULL, 
                                &Stream::worker_wrapper, this);
# ifndef LINALG_NO_CHECKS
    if (error != 0) {
      throw excSystemError("Unable to spawn thread, error = %d", error);
    }
# endif

#else /* use C++11 threads */

    worker_thread = std::thread(&Stream::worker, this);

#endif

    thread_alive = true;

  }

}

/** \brief            Terminate the stream
 */
inline Stream::~Stream() {

  PROFILING_FUNCTION_HEADER

  stop();
  clear();

}

// The routine for the worker thread
#ifndef DOXYGEN_SKIP
inline void Stream::worker() {

  PROFILING_FUNCTION_HEADER

  // Buffer so we can release the lock before executing
  std::function<void()> current_task;

  while (!terminate) {

    { // RAII scope for thread_lock

      // Safely examine the condition. The lock prevents other threads to 
      // modify it while we check the condition.
      Threads::MutexLock thread_lock(lock);

      if (queue.empty()) {

        // If the queue is empty we wait (releases the lock while waiting and 
        // reacquires it when the condition becomes true)
        cv.wait(thread_lock, [this](){ return (!queue.empty() || terminate); });

        if (terminate) return;

      }

      // Fetch work
      current_task = queue.back();
      queue.pop_back();

    } // Implicit release of thread_lock at the end of the scope block

    // Execute the task
    current_task();

    // Atomically increment counter (next_in_queue is an atomic variable)
    ++next_in_queue;

    // Signal to waiters
    cv.notify_all();

  }

  return;

}
#endif /* DOXYGEN_SKIP */

/** \brief            Add a new task to the queue
 *
 *  \param[in]        task
 *                    Functor to be put on the queue and processed
 *                    asynchronously.
 *
 *  \returns          Ticket number of the task.
 */
inline I_t Stream::add(std::function<void()> task) {

  PROFILING_FUNCTION_HEADER

  I_t my_ticket;

  if (synchronous_operation || !thread_alive) {

    // No worker thread, no locking required
    my_ticket = next_ticket;

    queue.push_front(task);

    ++next_ticket;
  
  } else {

    // Do the locking dance

    { // RAII scope for adder_lock

      // Safely examine the condition. The lock prevents other threads to  
      // modify it while we check the condition.
      Threads::MutexLock adder_lock(lock);

      my_ticket = next_ticket;

      queue.push_front(task);

      ++next_ticket;

    } // Implicit release of adder_lock at the end of the scope block

    cv.notify_all();

  }

  return my_ticket;

}

/** \brief            Synchronize with a specific task, i.e. wait for a
 *                    specific task to be completed
 *
 *  \param[in]        ticket
 *                    'Ticket' number of the task to be processed.
 */
inline void Stream::sync(I_t ticket) {

  PROFILING_FUNCTION_HEADER

  if (next_in_queue > ticket) {

    return;

  } else {

    if (synchronous_operation || !thread_alive) {
    
      // Work off the queue in the current thread (no locking required)

      std::function<void()> current_task;

      while (next_in_queue < ticket + 1) {

        current_task = queue.back();
        current_task();

        ++next_in_queue;
        queue.pop_back();
      
      }
    
    } else {

      // Acquire the lock, check the condition, wait till next_in_queue is 
      // larger than the requested ticket (release the lock in the mean time)
      Threads::MutexLock sync_lock(lock);
      cv.wait(sync_lock, [this, ticket](){ return next_in_queue > ticket; });

    }

  }

}

/** \brief            Synchronize with the stream (i.e. waits till all tasks
 *                    are processed)
 */
inline void Stream::sync() {

  sync(next_ticket - 1);

}

/** \brief            Signal the worker thread to exit, thereby stopping the 
 *                    queue
 */
inline void Stream::stop() {

  PROFILING_FUNCTION_HEADER

  if (!synchronous_operation && thread_alive) {

    {

      Threads::MutexLock terminate_lock(lock);

      terminate = true;

    }

    cv.notify_all();

#ifdef USE_POSIX_THREADS
    pthread_join(worker_thread, NULL);
#else /* C++11 threads */
    worker_thread.join();
#endif

  }

  thread_alive = false;

}

/** \brief            Remove all tasks from the stream
 */
inline void Stream::clear() {

  PROFILING_FUNCTION_HEADER

  // Locking blocks if there's another thread
  if (!synchronous_operation && thread_alive) {

    Threads::MutexLock clear_lock(lock);

    // Remove all tasks from the queue
    queue.clear();

    // This signals that no work is available
    next_ticket = next_in_queue;

  } else {
  
    // Remove all tasks from the queue
    queue.clear();

    // This signals that no work is available
    next_ticket = next_in_queue;
  
  }

}


#ifndef DOXYGEN_SKIP
/*  \brief            Helper class to bundle some matrix with a function.  
 *                    Effectively it is a function object that holds a matrix. 
 *
 *  \note             This class is neccessary only because Dense<T> and 
 *                    Sparse<T> explicitly don't have a copy constructor (see
 *                    in dense.h for an explanation). Otherwise matrices could 
 *                    be passed to lambda functions and std::bind calls by copy.
 *                    This class on the other hand is both copyable and 
 *                    callable, thereby solving the problem.
 *
 *                    This is probably not required anymore with C++14's 
 *                    'identifier initializer' lambda captures...
 *
 *                    All in all this definitely smells like bad design. Maybe 
 *                    it'd be better to provide copy constructors for Dense<T> 
 *                    and Sparse<T> in the first place ...
 */
template <typename T>
struct DenseMatrixFunctionBundle {

  Dense<T>               bundled_matrix;
  std::function<void()>  bundled_function;

  /*  \param[in]      matrix
   *                  Matrix to bundle with the function
   *
   *  \param[in]      function
   *                  Function object that takes a reference to a matrix as 
   *                  single argument. This function object would typically be 
   *                  created using a lambda function.
   *
   *  \example        See MPI/send_receive_matrix.h
   */
  DenseMatrixFunctionBundle(Dense<T>& matrix, T function) 
                         : bundled_function(function) {
    bundled_matrix.clone_from(matrix);
  }

  // Make the object copyable
  DenseMatrixFunctionBundle(const DenseMatrixFunctionBundle& other) {
    bundled_matrix.clone_from(other.bundled_matrix);
    bundled_function = other.bundled_function;
  }

  ~DenseMatrixFunctionBundle() { bundled_matrix.unlink(); }

  // Call operator
  inline void operator()() { bundled_function(bundled_matrix); }

};
#endif /* not DOXYGEN_SKIP */


#ifdef HAVE_CUDA
/** \brief            Class to encapsulate a CUDA stream
 *
 *  The stream is bound to a device and includes a cublasHandle and
 *  cusparseHandle associated with the stream. The stream is ordered but can't
 *  be synchronized on arbitrary positions. It only supports CUDA, cuBLAS and
 *  cuSPARSE functions.
 */
struct CUDAStream : StreamBase {

# ifndef DOXYGEN_SKIP
  cudaStream_t   cuda_stream;
  cublasHandle_t cublas_handle;
  int device_id;
# endif

  CUDAStream();
  CUDAStream(Streams stream_spec);
  CUDAStream(int device_id);
  ~CUDAStream();
  inline void sync();
  inline void set(int device_id, bool synchronous);

};

/** \brief              Constructor for a stream on the current device
 */
inline CUDAStream::CUDAStream() {

  PROFILING_FUNCTION_HEADER

  checkCUDA(cudaGetDevice(&device_id));
  checkCUDA(cudaStreamCreate(&cuda_stream));
  checkCUBLAS(cublasCreate(&cublas_handle));

}

/** \brief              Constructor for the default (synchronous) stream on
 *                      the current device
 */
inline CUDAStream::CUDAStream(Streams stream_spec)
                     : StreamBase(stream_spec) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasCreate(&cublas_handle));

}

/** \brief              Constructor for a stream on a specific device
 *
 *  \param[in]          device_id_
 *                      Id of the device for which to create the stream.
 */
inline CUDAStream::CUDAStream(int device_id_)
                     : device_id(device_id_) {

  PROFILING_FUNCTION_HEADER

  int prev_device;
  checkCUDA(cudaGetDevice(&prev_device));

  checkCUDA(cudaSetDevice(device_id));
  checkCUDA(cudaStreamCreate(&cuda_stream));
  checkCUBLAS(cublasCreate(&cublas_handle));

  checkCUDA(cudaSetDevice(prev_device));

}

// Destructor
inline CUDAStream::~CUDAStream() {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasDestroy(cublas_handle));

  if (!synchronous_operation) {

    checkCUDA(cudaStreamDestroy(cuda_stream));

  }

}

/** \brief            Synchronize with the stream (i.e. waits till all tasks
 *                    are processed)
 */
inline void CUDAStream::sync() {

  PROFILING_FUNCTION_HEADER

  if (synchronized || synchronous_operation) {

    return;

  }

  checkCUDA(cudaStreamSynchronize(cuda_stream));
  synchronized = true;

}

/** \brief            Function change/reinitialize a stream
 *
 *  \param[in]        device_id_
 *                    The id of the device the stream should be bound to
 */
inline void CUDAStream::set(int device_id_, bool synchronous) {

  // A stream is always pre-initialized, so we need to destruct that first
  checkCUBLAS(cublasDestroy(cublas_handle));
  if (!synchronous_operation) checkCUDA(cudaStreamDestroy(cuda_stream));

  device_id = device_id_;
  synchronous_operation = synchronous;

  if (synchronous_operation) {
  
    checkCUBLAS(cublasCreate(&cublas_handle));
  
  } else {

    int prev_device;
    checkCUDA(cudaGetDevice(&prev_device));

    checkCUDA(cudaSetDevice(device_id));
    checkCUDA(cudaStreamCreate(&cuda_stream));
    checkCUBLAS(cublasCreate(&cublas_handle));

    checkCUDA(cudaSetDevice(prev_device));

  }

}


#endif /* HAVE_CUDA */


#ifdef HAVE_MPI
/** \brief            MPIStream
 *
 *  The MPI stream handle allows for synchronization of multiple asynchronous
 *  MPI calls. However, there is no ordering guarantee for the calls (as there
 *  is none with one synchronization unit in MPI itself). It is in that sense
 *  not appendable as it does not guarantee that earlier operations are
 *  completed before later operations.
 */
struct MPIStream : StreamBase {

  MPIStream();
  MPIStream(Streams stream_spec);

# ifndef DOXYGEN_SKIP
  std::vector<MPI_Request> requests;
  std::vector<MPI_Status>  statuses;
  inline I_t add_operations(I_t i);
# endif

  inline void sync();

};

/** \brief            Constructor for an asynchronous stream
 */
inline MPIStream::MPIStream() {

  PROFILING_FUNCTION_HEADER

}

/** \brief            Constructor for a synchronous stream
 */
inline MPIStream::MPIStream(Streams stream_spec) : StreamBase(stream_spec) {

  PROFILING_FUNCTION_HEADER

}


// Internal helper function to append to the stream
# ifndef DOXYGEN_SKIP
inline I_t MPIStream::add_operations(I_t i) {

  PROFILING_FUNCTION_HEADER

  auto current = requests.size();
  requests.resize(current + i);
  statuses.resize(current + i);

  return current;

}
# endif

/** \brief            Synchronize with the stream (i.e. waits till all tasks
 *                    are processed)
 */
inline void MPIStream::sync() {

  PROFILING_FUNCTION_HEADER

  if (synchronized) {

    return;

  } else {

    for (unsigned int i = 0; i < requests.size(); ++i) {

      MPI_Wait(&requests[i], &statuses[i]);

# ifndef LINALG_NO_CHECKS
      if (statuses[i].MPI_ERROR != MPI_SUCCESS) {

        MPI::excMPIError my_exception("MPIStream: operation %d in stream "
                                      "failed: ", i);
        my_exception.set_status(statuses[i]);

        throw my_exception;

      }
# endif

    }

    requests.clear();
    statuses.clear();
    synchronized = true;

    return;

  }

}

#endif /* HAVE_MPI */

} /* namespace LinAlg */

#endif /* LINALG_STREAMS_H_ */
