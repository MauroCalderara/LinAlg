/** \file             test_utilities_stream.cc
 *
 *  \brief            Test for LinAlg::Utilities::Stream
 *
 *  \date             Created:  Oct 28, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          camauro <camauro@domain.tld>
 *
 *  \version          $Revision$
 */

#include <string>
#include <thread>     // std::this_thread::sleep_for
#include <chrono>
#include <cstdio>
#include <functional>

#ifdef USE_POSIX_THREADS
#include <pthread.h>
#endif

#include <linalg.h>

using namespace std;
using namespace LinAlg;

chrono::high_resolution_clock::time_point start;

#ifdef USE_POSIX_THREADS
pthread_mutex_t global_lock;
#else
mutex global_lock;
#endif
vector<string> factual_order;
vector<size_t> factual_times;


void test_payload(size_t delay, string payload) {

  this_thread::sleep_for(chrono::seconds(delay));

# ifdef USE_POSIX_THREADS
  pthread_mutex_lock(&global_lock);
# else
  unique_lock<mutex> my_lock(global_lock);
#endif

  auto now = chrono::high_resolution_clock::now();
  long elapsed_time = 
              chrono::duration_cast<chrono::microseconds>(now - start).count();

  printf("%s (t=%ld)\n", payload.c_str(), elapsed_time);
  factual_order.push_back(payload);
  factual_times.push_back(elapsed_time);

#ifdef USE_POSIX_THREADS
  pthread_mutex_unlock(&global_lock);
#else
  // implicit release
#endif

}

/* This program should print a alphabetic sequence of capital letters with one 
 * second delay between each letter
 */
int main() {

  Stream test_stream;

  vector<string>      reference_order;
  vector<size_t>      reference_times;

  // The payloads and the references
  auto A = std::bind(test_payload, 0, "A");
  reference_order.push_back("A"); reference_times.push_back(0);
  auto B = std::bind(test_payload, 1, "B");
  reference_order.push_back("B"); reference_times.push_back(1000000);
  auto C = std::bind(test_payload, 2, "C");
  reference_order.push_back("C"); reference_times.push_back(2000000);
  auto D = std::bind(test_payload, 2, "D");
  reference_order.push_back("D"); reference_times.push_back(3000000);
  auto E = std::bind(test_payload, 1, "E");
  reference_order.push_back("E"); reference_times.push_back(4000000);


  // Load the stream
  test_stream.add(B);
  auto D_ticket = test_stream.add(D);

  // Start timing
  start = chrono::high_resolution_clock::now();

  // A: T(A) = 0
  A();

  // B: T(B) = 1
  test_stream.start();

  // C: T(A) + 2 = 2
  C();

  // D: T(B) + 2 = 3

  // E: T(D) + 1 = 4
  test_stream.sync(D_ticket);
  E();


  for (size_t i = 0; i < reference_order.size(); ++i) {
    if (reference_order[i] != factual_order[i]) {
      printf("Order invalid:\n"
             "   reference_order[%ld] = %s\n"
             "     factual_order[%ld] = %s\n", i, reference_order[i].c_str(),
             i, factual_order[i].c_str());
      return 1;
    }
  }

  printf("Order valid.\n");
  printf("Timing deviations:\n");
  for (size_t i = 0; i < reference_times.size(); ++i) {
    auto deviation = factual_times[i] - reference_times[i];
    printf("  %s  %ld us\n", reference_order[i].c_str(), deviation);
  }

  return 0;

}
