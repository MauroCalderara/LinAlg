/** \file
 *
 *  \brief            Support for the intel&copy; MIC engine
 *
 *  \date             Created:  Jul 12, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_MIC_HELPER_H_
#define LINALG_MIC_HELPER_H_

#ifdef HAVE_MIC

#include "../exceptions.h"   // LinAlg::excLinAlg

namespace LinAlg {

namespace MIC {

/** \brief            Deallocates memory on the GPGPU
 *
 *  \param[in]        device_ptr
 *                    A pointer to previously allocated memory on the MIC.
 *
 *  \param[in]        device_id
 *                    The number of the MIC on which the memory has been
 *                    allocated.
 */
template <typename T>
inline void mic_deallocate(T* device_ptr, int device_id) {

  #pragma offload_transfer target (mic:device_id) \
                           nocopy (device_ptr : alloc_if(0) free_if(1))

};

/** \brief            Allocates memory on the CPU and MIC
 *
 *  \param[in]        size
 *                    Length of the memory array to allocate on the CPU and MIC.
 *
 *  \param[in]        device_id
 *                    The number of the MIC on which to allocate the memory.
 *
 *  \return           device_ptr
 *                    A pointer to memory on the CPU/MIC.
 *
 *  \note             As far as I understand there is no way to have memory
 *                    allocated on the MIC without having a corresponding
 *                    allocation in main memory unless the memory is allocated
 *                    in an offloaded block (that is the matrix is created on
 *                    the MIC itself). For the latter case we will have to make
 *                    extra provisions.
 *
 *  \todo             Make the extra provisions to deal with the situation of
 *                    using the library directly on the MIC (that is, when we
 *                    are on the MIC and the location is specified as the
 *                    current MIC)
 */
template <typename T>
inline std::shared_ptr<T> mic_make_shared(I_t size, int device_id) {

  std::shared_ptr<T> device_ptr = host_make_shared<T>(size);
  #pragma offload_transfer target (mic:device_id) \
                           nocopy (device_ptr.get() : length(size) \
                                                alloc_if(1) \
                                                free_if(0))

  return device_ptr;

};

} /* namespace LinAlg::MIC*/

} /* namespace LinAlg */

#endif /* HAVE_MIC */

#endif /* LINALG_MIC_HELPER_H_ */
