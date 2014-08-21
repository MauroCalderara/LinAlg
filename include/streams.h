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

#ifdef HAVE_CUDA
#include <cuda_runtime.h> // various CUDA routines
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include "CUDA/cuda_checks.h"
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <functional> // std::function
#include <thread>     // std::thread
#include <mutex>      // std::mutex
#include <queue>      // std::deque
#include <condition_variable>
#include <atomic>     // std::atomic

#include "types.h"
#include "exceptions.h"

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

  StreamBase() : synchronous_operation(false), synchronized(true) {};

  /** \brief          Synchronous stream constructor
   */
  StreamBase(Streams stream_spec) {

    synchronous_operation = (stream_spec == Streams::Synchronous) ? true :
                                                                    false;
    synchronized = true;

  };

#ifndef DOXYGEN_SKIP
  virtual inline void sync() = 0;

  bool synchronous_operation;
  bool synchronized;
#endif

};

/** Generic stream:
 *
 *  Tasks are std::function<void()> objects. Each stream has one worker thread
 *  assigned that walks fetches tasks from the head of a deque. Appending
 *  tasks is pushing to the front to the queue. Tasks are processed in the
 *  order of being pushed to the queue.
 */
struct Stream : StreamBase {

#ifndef DOXYGEN_SKIP
  std::mutex                        lock;
  std::condition_variable           cv;
  std::thread                       worker_thread;
  std::deque<std::function<void()>> queue;
  std::atomic<I_t> next_in_queue;

  inline void worker();
  I_t next_ticket;             // next_ticket = next_in_queue -> no work;
  bool terminate;
#endif

  Stream();
  Stream(Streams stream_spec);
  ~Stream();

  inline I_t  add(std::function<void()> task);
  inline void sync(I_t ticket);
  inline void sync();
  inline void destroy();

};

/** \brief            Constructor for a stream
 */
inline Stream::Stream()
             : StreamBase(),
               next_ticket(0),
               terminate(false) {

  next_in_queue.store(0);

  // Start the worker thread
  worker_thread = std::thread(&Stream::worker, this);

};

/** \brief            Constructor for aa stream without a worker thread
 *                    (causes synchronous execution on all routines that use
 *                    this stream)
 */
inline Stream::Stream(Streams stream_spec) : StreamBase(stream_spec) {
};

/** \brief            Terminate the stream
 */
inline Stream::~Stream() {

  destroy();

};

// The routine for the worker thread
#ifndef DOXYGEN_SKIP
inline void Stream::worker() {

  // Buffer so we can release the lock before executing
  std::function<void()> my_task;

  while (!terminate) {

    {

      // Aquire the lock
      std::unique_lock<std::mutex> thread_lock(lock);

      if (queue.empty()) {

        // If the queue is empty we wait (releases the lock)
        cv.wait(thread_lock, [this](){ return (!queue.empty() || terminate); });

        if (terminate) {

          return;

        }

      }

      // Fetch work
      my_task = queue.back();
      queue.pop_back();

    } // Release lock (implicitly)

    // Execute the task
    my_task();

    // Increment counter (atomically)
    ++next_in_queue;

    // Signal to waiters
    cv.notify_all();

  }

  return;

};
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

#ifndef LINALG_NO_CHECKS
  if (synchronous_operation) {

    throw excBadArgument("Stream.add(): Can only add to asynchronous Streams");

  }
#endif

  I_t my_ticket;

  {

    std::unique_lock<std::mutex> adder_lock(lock);

    my_ticket = next_ticket;

    queue.push_front(task);

    ++next_ticket;

  }

  cv.notify_all();

  return my_ticket;

};

/** \brief            Synchronize with a specific task, i.e. wait for a
 *                    specific task to be completed
 *
 *  \param[in]        ticket
 *                    'Ticket' number of the task to be processed.
 */
inline void Stream::sync(I_t ticket) {

  if (next_in_queue > ticket) {

    return;

  } else {

    std::unique_lock<std::mutex> sync_lock(lock);

    cv.wait(sync_lock, [this, ticket](){ return next_in_queue > ticket; });

    return;

  }

};

/** \brief            Synchronize with the stream (i.e. waits till all tasks
 *                    are processed)
 */
inline void Stream::sync() {

  if (queue.empty()) {

    return;

  } else {

    std::unique_lock<std::mutex> sync_lock(lock);

    cv.wait(sync_lock, [this](){ return queue.empty(); });

    return;

  }

};

/** \brief            Signal the worker thread to exit
 */
inline void Stream::destroy() {

  {

    std::unique_lock<std::mutex> terminate_lock(lock);

    terminate = true;

  }

  cv.notify_all();

  worker_thread.join();

};




#ifdef HAVE_CUDA
/** \brief              Class to encapsulate a CUDA stream
 *
 *  The stream is bound to a device and includes a cublasHandle and
 *  cusparseHandle associated with the stream. The stream is ordered but can't
 *  be synchronized on arbitrary positions. It only supports CUDA, cuBLAS and
 *  cuSPARSE functions.
 */
struct CUDAStream : StreamBase {

#ifndef DOXYGEN_SKIP
  cudaStream_t   cuda_stream;
  cublasHandle_t cublas_handle;
  int device_id;
#endif

  CUDAStream();
  CUDAStream(Streams stream_spec);
  CUDAStream(int device_id);
  ~CUDAStream();
  inline void sync();
  inline void destroy();

};

/** \brief              Constructor for a stream on the current device
 */
inline CUDAStream::CUDAStream() {

  checkCUDA(cudaGetDevice(&device_id));
  checkCUDA(cudaStreamCreate(&cuda_stream));
  checkCUBLAS(cublasCreate(&cublas_handle));

};

/** \brief              Constructor for the default (synchronous) stream on
 *                      the current device
 */
inline CUDAStream::CUDAStream(Streams stream_spec)
                     : StreamBase(stream_spec) {

  checkCUBLAS(cublasCreate(&cublas_handle));

};

/** \brief              Constructor for a stream on a specific device
 *
 *  \param[in]          device_id
 *                      Id of the device for which to create the stream.
 */
inline CUDAStream::CUDAStream(int device_id)
                     : device_id(device_id) {

  int prev_device;
  checkCUDA(cudaGetDevice(&prev_device));

  checkCUDA(cudaSetDevice(device_id));
  checkCUDA(cudaStreamCreate(&cuda_stream));
  checkCUBLAS(cublasCreate(&cublas_handle));

  checkCUDA(cudaSetDevice(prev_device));

};

// Destructor
inline CUDAStream::~CUDAStream() {

  destroy();

};

/** \brief            Synchronize with the stream (i.e. waits till all tasks
 *                    are processed)
 */
inline void CUDAStream::sync() {

  if (synchronized || synchronous_operation) {

    return;

  }

  checkCUDA(cudaStreamSynchronize(cuda_stream));
  synchronized = true;

};

/** \brief            Destroy the stream
 */
inline void CUDAStream::destroy() {

  checkCUBLAS(cublasDestroy(cublas_handle));

  if (synchronous_operation) {

    checkCUDA(cudaStreamDestroy(cuda_stream));

  }

};

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

#ifndef DOXYGEN_SKIP
  std::vector<MPI_Request> requests;
  std::vector<MPI_Status>  statuses;
  inline I_t add_operations(I_t i);
#endif

  inline void sync();

};

/** \brief            Constructor for an asynchronous stream
 */
inline MPIStream::MPIStream() {
};

/** \brief            Constructor for a synchronous stream
 */
inline MPIStream::MPIStream(Streams stream_spec) : StreamBase(stream_spec) {
};


// Internal helper function to append to the stream
#ifndef DOXYGEN_SKIP
inline I_t MPIStream::add_operations(I_t i) {

  auto current = requests.size();
  requests.resize(current + i);
  statuses.resize(current + i);

  return current;

};
#endif

/** \brief            Synchronize with the stream (i.e. waits till all tasks
 *                    are processed)
 */
inline void MPIStream::sync() {

  if (synchronized) {

    return;

  } else {

    for (unsigned int i = 0; i < requests.size(); ++i) {

      MPI_Wait(&requests[i], &statuses[i]);

#ifndef LINALG_NO_CHECKS
      if (statuses[i].MPI_ERROR != MPI_SUCCESS) {

        MPI::excMPIError my_exception("MPIStream: operation %d in stream "
                                      "failed: ", i);
        my_exception.set_status(statuses[i]);

        throw my_exception;

      }
#endif

    }

    requests.clear();
    statuses.clear();
    synchronized = true;

    return;

  }

};

#endif /* HAVE_MPI */

} /* namespace LinAlg */

#endif /* LINALG_STREAMS_H_ */
