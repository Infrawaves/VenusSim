/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __WORKLOAD_HH__
#define __WORKLOAD_HH__

#include <memory>
#include <string>
#include <unordered_map>

#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/CommunicatorGroup.hh"
#include "astra-sim/workload/HardwareResource.hh"
#include "astra-sim/workload/ETDefinition.hh"

namespace AstraSim {

class Sys;
class DataSet;

class Workload : public Callable {
 public:
  Workload(Sys* sys, std::string et_filename, std::string comm_group_filename, std::map<std::string, int> workload_template_id={});
  ~Workload();

  // communicator groups
  void initialize_comm_group(std::string comm_group_filename);

  // event-based simulation
  void issue_dep_free_nodes();
  void issue(std::shared_ptr<SelectETFeederNode> node);
  void issue_replay(std::shared_ptr<SelectETFeederNode> node);
  void issue_remote_mem(std::shared_ptr<SelectETFeederNode> node);
  void issue_comp(std::shared_ptr<SelectETFeederNode> node);
  void issue_comm(std::shared_ptr<SelectETFeederNode> node);
  void skip_invalid(std::shared_ptr<SelectETFeederNode> node);
  void call(EventType event, CallData* data);
  void fire();
  // stats
  void report();

  SelectETFeeder* et_feeder;
  std::map<std::string, CommunicatorGroup*> comm_group_list;
  //CommunicatorGroup* comm_group;
  HardwareResource* hw_resource;
  Sys* sys;
  std::unordered_map<int, uint64_t> collective_comm_node_id_map;
  std::unordered_map<int, DataSet*> collective_comm_wrapper_map;
  bool is_finished;
  
  std::ofstream node_type_fout;
};

} // namespace AstraSim

#endif /* __WORKLOAD_HH__ */
