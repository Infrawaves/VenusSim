/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __COMMUNICATOR_GROUP_HH__
#define __COMMUNICATOR_GROUP_HH__

#include <assert.h>
#include <map>
#include <vector>
#include "astra-sim/system/topology/ComplexLogicalTopology.hh"
#include "astra-sim/system/Common.hh"

namespace AstraSim {

class Sys;
class CollectivePlan;
class CommunicatorGroup {
 public:
  CommunicatorGroup(int id, std::vector<int> involved_NPUs, Sys* generator,std::string group_name);
  CollectivePlan* get_collective_plan(ComType comm_type);
  bool if_multi_group(std::map<int, std::vector<int>>& coordinates_dict,std::vector<int>& group_list);
  void set_id(int id);
  ~CommunicatorGroup();

  std::vector<int> involved_NPUs;
  int num_streams;
  std::string group_name;
  int group_dim0_gpu_num;
  int group_gpu_num;

  int get_id(){return this->id;};

 private:
  int id;
  Sys* generator;
  std::map<ComType, CollectivePlan*> comm_plans;
};

} // namespace AstraSim

#endif /* __COMMUNICATOR_GROUP_HH__ */
