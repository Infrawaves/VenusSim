/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

// TODO: HardwareResource.hh should be moved to the system layer.

#ifndef __HARDWARE_RESOURCE_HH__
#define __HARDWARE_RESOURCE_HH__

#include <cstdint>

#include "astra-sim/workload/ETDefinition.hh"

namespace AstraSim {

class HardwareResource {
 public:
  HardwareResource(uint32_t num_npus);
  void occupy(const std::shared_ptr<SelectETFeederNode> node);
  void release(const std::shared_ptr<SelectETFeederNode> node);
  bool is_available(const std::shared_ptr<SelectETFeederNode> node) const;

  const uint32_t num_npus;
  uint32_t num_in_flight_cpu_ops;
  uint32_t num_in_flight_gpu_comp_ops;
  uint32_t num_in_flight_gpu_comm_ops;
};

} // namespace AstraSim

#endif /* __HARDWARE_RESOURCE_HH__ */
