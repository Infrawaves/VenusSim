/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/CommunicatorGroup.hh"

#include <algorithm>

#include "astra-sim/system/CollectivePlan.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/topology/RingTopology.hh"
#include <cstdlib>
#include "astra-sim/system/topology/GeneralComplexSubTopology.hh"
using namespace AstraSim;

CommunicatorGroup::CommunicatorGroup(
    int id,
    std::vector<int> involved_NPUs,
    Sys* generator,
    std::string group_name) {
  set_id(id);
  this->involved_NPUs = involved_NPUs;
  this->generator = generator;
  this->group_name = group_name;
  std::sort(involved_NPUs.begin(), involved_NPUs.end());
  assert(
      std::find(involved_NPUs.begin(), involved_NPUs.end(), generator->id) !=
      involved_NPUs.end());
}

CommunicatorGroup::~CommunicatorGroup() {
  for (auto cg : comm_plans) {
    CollectivePlan* cp = cg.second;
    delete cp;
  }
}

void CommunicatorGroup::set_id(int id) {
  assert(id > 0);
  this->id = id;
  this->num_streams = id * 1000000;
}

CollectivePlan* CommunicatorGroup::get_collective_plan(ComType comm_type) {
  if (comm_plans.find(comm_type) != comm_plans.end()){
    if(this->generator->log_level == LOG_DEBUG)
      std::cout<<"found old comm_plan in get_collective_plan"<<std::endl;
    return comm_plans[comm_type];
  }
    
  if (static_cast<uint64_t>(generator->total_nodes) == involved_NPUs.size()) {
    LogicalTopology* logical_topology =
        generator->get_logical_topology(comm_type);
    std::vector<CollectiveImpl*> collective_implementation =
        generator->get_collective_implementation(comm_type);
    std::vector<bool> dimensions_involved(10, true);
    bool should_be_removed = false;
    comm_plans[comm_type] = new CollectivePlan(
        logical_topology,
        collective_implementation,
        dimensions_involved,
        should_be_removed);
    if(this->generator->log_level == LOG_DEBUG)
      std::cout<<"new comm_plan in total_node == involved_NPUs get_collective_plan"<<std::endl;
    return comm_plans[comm_type];
  } else {

    LogicalTopology* logical_topology =
        generator->get_logical_sub_topology(comm_type);
    int dims = logical_topology->get_num_of_dimensions();
    GeneralComplexSubTopology* sub_logical_topology = (GeneralComplexSubTopology*)logical_topology;
    
    for(int dim = 0;dim<dims;dim++){
      std::vector<int> sub_npus = sub_logical_topology->get_sub_dim(group_name,dim,generator->id);
      std::cout<<"dim_size:"<<sub_npus.size()<<std::endl;
      //If multi-dim conditions are not met, use the ring topology
      if(sub_npus.size()==0){
        LogicalTopology* logical_topology = new RingTopology(
          RingTopology::Dimension::Local, generator->id, involved_NPUs);
          std::vector<CollectiveImpl*> collective_implementation{
              new CollectiveImpl(CollectiveImplType::Ring)};
          std::vector<bool> dimensions_involved(1, true);
          bool should_be_removed = true;
          comm_plans[comm_type] = new CollectivePlan(
              logical_topology,
              collective_implementation,
              dimensions_involved,
              should_be_removed);
          if(this->generator->log_level == LOG_DEBUG)
            std::cout<<"new comm_plan in total_node != involved_NPUs get_collective_plan(1)"<<std::endl;
          return comm_plans[comm_type];
      }
      RingTopology* ring = new RingTopology(
          RingTopology::Dimension::Local,generator->id,sub_npus);
      sub_logical_topology->dimension_topology[dim] = ring;
    }
    
    //collective_implementation
    std::vector<CollectiveImpl*> collective_implementation =
        generator->get_collective_implementation(comm_type);
    std::vector<bool> dimensions_involved;
    if (dims > 1) {
        dimensions_involved = std::vector<bool>(10, true);
    } else {
        dimensions_involved = std::vector<bool>(1, true);
    }
    bool should_be_removed = true;
    comm_plans[comm_type] = new CollectivePlan(
        sub_logical_topology,
        collective_implementation,
        dimensions_involved,
        should_be_removed);
    if(this->generator->log_level == LOG_DEBUG)
      std::cout<<"new comm_plan in total_node != involved_NPUs get_collective_plan(2)"<<std::endl;
    return comm_plans[comm_type];
  }
  assert(false);
  return nullptr;
}

bool CommunicatorGroup::if_multi_group(std::map<int, std::vector<int>>& coordinates_dict, std::vector<int>& group_list) {
  if (coordinates_dict.empty() || group_list.empty()) return false;

  int dim = coordinates_dict.begin()->second.size(); // The total dimensionality of the coordinates
  std::vector<std::set<int>> unique_values(dim); // Record the distinct values for each dimension

  // Count the unique values in each dimension of the group_list
  for (int id : group_list) {
      const std::vector<int>& coords = coordinates_dict.at(id);
      for (int d = 0; d < dim; ++d) {
          unique_values[d].insert(coords[d]);
      }
  }

  // Identify the (n-1)th dimension, which is the dimension of the last distinct value
  int n_minus_1 = -1;
  for (int d = dim - 1; d >= 0; --d) {
      if (unique_values[d].size() > 1) {
          n_minus_1 = d;
          break;
      }
  }
  if(group_list.size()==1){
    return false;
  }
  if (n_minus_1 == -1) {
    std::cout<<"Error: commication groups have same id!"<<std::endl;
    std::exit(0);
  }; // If all coordinates are identical, directly exit.
  if (n_minus_1 == 0){
    return false;
  }
  // Count the occurrences of coordinate combinations in dimensions up to (n-1)
  std::map<std::vector<int>, int> id_times;
  for (int id : group_list) {
      const std::vector<int>& coords = coordinates_dict.at(id);
      std::vector<int> key(coords.begin(), coords.begin() + n_minus_1); // 取前 n-1 维
      id_times[key]++;
  }

  // Deduplicate the frequencies and check if all values are equal
  std::set<int> unique_counts;
  for (const auto& pair : id_times) {
      unique_counts.insert(pair.second);
  }

  return unique_counts.size() == 1 && id_times.size() > 1 *unique_counts.begin() > 1;
}
