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

// Makes the constructor for synchronous streams a bit more explicit
enum class Streams { Synchronous };

/** \brief            Base class to handle mechanisms for asynchronous 
 *                    execution
 *
 *  A stream is a unit of asynchronous execution. The .sync() routine
 *  synchronizes with the execution of the stream. The class is semantically 
 *  similar to of C++11 std::futures with the std::future.get() functionality 
 *  provided by Stream.sync()
 *
 *  This stream implementation contains a thread based facility for generic 
 *  asynchronous execution as well as an interface to CUDA streams and MPIs 
 *  asynchronous facilities.
 *
 *  Thread based sub stream
 *  -----------------------
 *    - Tasks supported: arbitrary functions
 *    - Ordering:        global (tasks get executed in the order they are
 *                       added to the stream)
 *    - Synchronization: arbitrary queue positions (incl. global 
 *                       synchronization)
 *
 *  CUDA based sub stream
 *  ---------------------
 *    - Tasks supported: asynchronous CUDA/CuBLAS/CuSPARSE functions
 *    - Ordering:        global (tasks get executed in the order they are
 *                       added to the stream)
 *    - Synchroniztion:  global
 *
 *  MPI based sub stream
 *  --------------------
 *    - Tasks supported: asynchronous MPI functions
 *    - Ordering:        none (tasks get executed in arbitrary order)
 *    - Synchroniztion:  global
 */
struct Stream {

  Stream();
  Stream(Streams ignored);
#ifdef HAVE_CUDA // or MIC
  Stream(int device_id_); // use the same for MIC
  Stream(int device_id_, Streams ignored);
#endif
  ~Stream();

  // General facilities
  inline void sync(I_t ticket);
  inline void sync();
  inline void clear();
  inline void set(int device_id_, bool asynchronous_);


#ifndef DOXYGEN_SKIP
  inline void load_defaults();

  // Signal that the stream operates synchronous
  bool synchronous;

  // Signal whether to prefer the native facilities (CUDA, MPI) over the 
  // generic, thread based stream (this is read by the routines to which the 
  // stream is passed as argument)
  bool prefer_native;

  // Device the stream is bund to (shared for all engine types)
  int device_id;


  /////////////////////////////////////////
  // Facilities for the thread based stream
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
  std::atomic<I_t>                  next_in_queue;

  inline void                       start_thread();
  inline void                       stop_thread();
  bool                              terminate_thread;
  bool                              thread_alive;
  inline void                       worker();

  inline I_t                        add(std::function<void()> task);
  I_t                               next_ticket;
  // next_ticket == next_in_queue => no work;
  inline void                       sync_generic(I_t ticket);
  bool                              generic_synchronized;

# ifdef HAVE_CUDA
  //////////////////////////////
  // Facilities for CUDA streams
  cudaStream_t                      cuda_stream;
  cublasHandle_t                    cublas_handle;
  cusparseHandle_t                  cusparse_handle;
  inline void                       sync_cuda();
  bool                              cuda_synchronized;
  inline void                       cuda_create();
  inline void                       cuda_create(int device_id);
  inline void                       cuda_destroy();
# endif

# ifdef HAVE_MPI
  /////////////////////////////
  // Facilities for MPI streams
  std::vector<MPI_Request>          mpi_requests;
  std::vector<MPI_Status>           mpi_statuses;
  inline I_t                        _add_mpi_tasks(I_t n_tasks);
  inline void                       sync_mpi();
  bool                              mpi_synchronized;
# endif

#endif /* not DOXYGEN_SKIP */

};

/** \brief            Constructor
 *
 *  Generic stream: The constructor doesn't spawn a thread to avoid having 
 *  lingering threads for classes with streams as members. Use start_thread() 
 *  to create the worker thread and 'activate' the generic stream.
 *
 *  CUDA stream: The constructor creates a stream and handles for the current 
 *  device. To create a stream for a specific device, use the corresponding 
 *  constructor.
 *
 *  MPI stream: standard handler for asynchronous MPI calls.
 */
inline Stream::Stream() {

  PROFILING_FUNCTION_HEADER

  load_defaults();

#ifdef HAVE_CUDA
  cuda_create();
#endif

}

/** \brief            Constructor a synchronous stream
 *
 *  \param[in]        ignored
 *
 *  \example
 *    Stream mystream(Streams::Synchronous);
 */
inline Stream::Stream(Streams ignored) {

  PROFILING_FUNCTION_HEADER

  load_defaults();
  synchronous = false;

#ifdef HAVE_CUDA
  cuda_create();
#endif

}


#ifdef HAVE_CUDA // or MIC
/** \brief            Constructor for a stream on a specific device
 *
 *  With the exception of the device specification the same as the default 
 *  constructor.
 *
 *  \param[in]        device_id
 *                    Id of the device to be used by the stream. Shared across 
 *                    all sub streams that support devices as there is 
 *                    typically only one sort of accelerator in one system.
 */
inline Stream::Stream(int device_id_) {

  PROFILING_FUNCTION_HEADER

  load_defaults();

# ifdef HAVE_CUDA
  cuda_create(device_id_);
# endif

}

/** \brief            Constructor for a synchronous stream on a specific 
 *                    device
 *
 *  \param[in]        device_id
 *                    Id of the device to be used by the stream. Shared across 
 *                    all sub streams that support devices as there is 
 *                    typically only one sort of accelerator in one system.
 *
 *  \param[in]        ignored
 */
inline Stream::Stream(int device_id_, Streams ignored) {

  PROFILING_FUNCTION_HEADER

  load_defaults();
  synchronous = false;

# ifdef HAVE_CUDA
  cuda_create(device_id_);
# endif

}
#endif /* HAVE_CUDA  or MIC */

/** \brief            Destructor
 */
inline Stream::~Stream() {

  PROFILING_FUNCTION_HEADER

  stop_thread();
  clear();
  sync();

#ifdef HAVE_CUDA
  cuda_destroy();
#endif

}

/** \brief            Synchronize with a specific task i.e. wait for a 
 *                    specific task to be completed
 *
 *  For streams that don't support synchronizing with a specific ticket, 
 *  global synchronization is performed.
 *
 *  \param[in]        ticket
 *                    OPTIONAL: 'Ticket' number of the task to be processed. A 
 *                    ticket number of 0 indicates that global synchronization 
 *                    is requested. DEFAULT: 0
 */
inline void Stream::sync(I_t ticket) {

  PROFILING_FUNCTION_HEADER

  sync_generic(ticket);
#ifdef HAVE_CUDA
  sync_cuda();
#endif
#ifdef HAVE_MPI
  sync_mpi();
#endif

}
/** \overload
 */
inline void Stream::sync() { sync(0); }


/** \brief            Remove all tasks from the stream
 *
 *  For streams not supporting clearing, synchronization is performed.
 */
inline void Stream::clear() {

  PROFILING_FUNCTION_HEADER

  // Generic stream operations

  // Locking blocks if there's another thread
  if (!synchronous && thread_alive) {

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

#ifdef HAVE_CUDA
  sync_cuda();
#endif
#ifdef HAVE_MPI
  sync_mpi();
#endif

}

/** \brief            Set properties of the stream
 *
 *  \param[in]        device_id_
 *                    Device to bind the stream to
 *
 *  \param[in]        asynchronous_
 *                    True -> make the stream asynchronous
 *                    False -> make the stream synchronous
 */
inline void Stream::set(int device_id_, bool asynchronous_) {

#ifdef HAVE_CUDA
  sync_cuda();
  cuda_destroy();
#endif

  synchronous = !asynchronous_;
  device_id = device_id;

#ifdef HAVE_CUDA
  cuda_create(device_id);
#endif

}

#ifndef DOXYGEN_SKIP
/** \brief            Set all members to default values
 */
inline void Stream::load_defaults() {

  synchronous          = false;
  prefer_native        = true;
  device_id            = 0;
  thread_alive         = false;
  next_ticket          = 1;
  generic_synchronized = true;
# ifdef HAVE_CUDA
  cuda_synchronized    = true;
# endif
# ifdef HAVE_MPI
  mpi_synchronized     = true;
# endif
  terminate_thread     = false;
  next_in_queue.store(1);

}
#endif /* not DOXYGEN_SKIP */

// Generic stream facilities
/** \brief            Enables the generic stream by creating the worker thread
 *
 *  For synchronous streams, no worker thread is spawned.
 */
inline void Stream::start_thread() {

  PROFILING_FUNCTION_HEADER

  if (synchronous) {

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

/** \brief            Signal the worker thread to exit, thereby stopping the 
 *                    queue
 */
inline void Stream::stop_thread() {

  PROFILING_FUNCTION_HEADER

  if (!synchronous && thread_alive) {

    {

      Threads::MutexLock terminate_lock(lock);

      terminate_thread = true;

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

#ifndef DOXYGEN_SKIP
/** \brief            Payload for the thread
 */
inline void Stream::worker() {

  PROFILING_FUNCTION_HEADER

  // Buffer so we can release the lock before executing
  std::function<void()> current_task;

  while (!terminate_thread) {

    { // RAII scope for thread_lock

      // Safely examine the condition. The lock prevents other threads to 
      // modify it while we check the condition.
      Threads::MutexLock thread_lock(lock);

      if (queue.empty()) {

        // If the queue is empty we wait (releases the lock while waiting and 
        // reacquires it when the condition becomes true)
        cv.wait(thread_lock,
                [this](){ return (!queue.empty() || terminate_thread); });

        if (terminate_thread) return;

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

/** \brief            Add a new task to the queue of the generic stream
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

  if (synchronous || !thread_alive) {

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

  generic_synchronized = false;

  return my_ticket;

}

/** \brief            Synchronize with a specific task in the generic stream, 
 *                    i.e. wait for a specific task to be completed
 *
 *  \param[in]        ticket
 *                    'Ticket' number of the task to be processed.
 */
inline void Stream::sync_generic(I_t ticket) {

  PROFILING_FUNCTION_HEADER

  if (generic_synchronized) return;

  if (next_in_queue > ticket) {

    return;

  } else {

    if (synchronous || !thread_alive) {

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

  if (ticket == next_in_queue - 1) generic_synchronized = true;

}


// CUDA Stream facilities
#ifdef HAVE_CUDA
/** \brief            Synchronize with the CUDA stream
 */
inline void Stream::sync_cuda() {

  PROFILING_FUNCTION_HEADER

  if (cuda_synchronized) return;

  checkCUDA(cudaStreamSynchronize(cuda_stream));

  cuda_synchronized = true;

}

/** \brief            Create all handles for the cuda stream
 *
 *  \param[in]        device_id_
 *                    OPTIONAL: the id of the device to create the handlers 
 *                    for. If none is given, it is assumed that the handles 
 *                    are to be created for the current device.
 *
 *  \note             The function reads 'synchronous' and assumes the current
 *                    device is the one to create the handlers for
 */
inline void Stream::cuda_create(int device_id_) {


  int prev_device;
  checkCUDA(cudaGetDevice(&prev_device));
  checkCUDA(cudaSetDevice(device_id_));

  if (!synchronous) checkCUDA(cudaStreamCreate(&cuda_stream));

  checkCUBLAS(cublasCreate(&cublas_handle));
  if (!synchronous) {

    checkCUBLAS(cublasSetStream(cublas_handle, cuda_stream));

  } else { 

    checkCUBLAS(cublasSetStream(cublas_handle, NULL));

  }

  checkCUSPARSE(cusparseCreate(&cusparse_handle));
  if (!synchronous) {

    checkCUSPARSE(cusparseSetStream(cusparse_handle, cuda_stream));

  } else {
  
    checkCUSPARSE(cusparseSetStream(cusparse_handle, NULL));
  
  }

  checkCUDA(cudaSetDevice(prev_device));

  device_id = device_id_;

}
/** \overload
 */
inline void Stream::cuda_create() {

  int current_device;
  checkCUDA(cudaGetDevice(&current_device));

  cuda_create(current_device);

}

/** \brief            Destroy all handles for the cuda stream
 */
inline void Stream::cuda_destroy() {

  checkCUSPARSE(cusparseDestroy(cusparse_handle));
  checkCUBLAS(cublasDestroy(cublas_handle));
  if (!synchronous) checkCUDA(cudaStreamDestroy(cuda_stream));

}
#endif /* HAVE_CUDA */


#ifdef HAVE_MPI
# ifndef DOXYGEN_SKIP
/*  \brief              Internal helper function facilitating appending to the 
 *                      MPI stream
 *
 *  \param[in]          n_tasks
 *                      Number of tasks that will be added
 *
 *  \returns            The position at which insertion can start
 */
inline I_t Stream::_add_mpi_tasks(I_t n_tasks) {

  PROFILING_FUNCTION_HEADER

  auto current = mpi_requests.size();
  mpi_requests.resize(current + n_tasks);
  mpi_statuses.resize(current + n_tasks);

  return current;

}
# endif /* not DOXYGEN_SKIP */

/** \brief            Synchronize with the MPI stream
 */
inline void Stream::sync_mpi() {

  PROFILING_FUNCTION_HEADER

  if (mpi_synchronized) return;

  for (unsigned int i = 0; i < mpi_requests.size(); ++i) {

    MPI_Wait(&mpi_requests[i], &mpi_statuses[i]);

# ifndef LINALG_NO_CHECKS
    if (mpi_statuses[i].MPI_ERROR != MPI_SUCCESS) {

      MPI::excMPIError my_exception("MPIStream: operation %d in stream "
                                    "failed: ", i);
      my_exception.set_status(mpi_statuses[i]);

      throw my_exception;

    }
# endif

  }

  mpi_requests.clear();
  mpi_statuses.clear();

  mpi_synchronized = true;

}
#endif /* HAVE_MPI */

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
 *                    and Sparse<T> in the first place but I've chosen to stay 
 *                    with the Google C++ style guide on this point.
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

} /* namespace LinAlg */

#endif /* LINALG_STREAMS_H_ */
