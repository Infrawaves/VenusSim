/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/topology/GeneralComplexSubTopology.hh"

#include <cassert>
#include <iostream>

#include "astra-sim/system/topology/DoubleBinaryTreeTopology.hh"
#include "astra-sim/system/topology/RingTopology.hh"



using namespace std;
using namespace AstraSim;
GeneralComplexSubTopology::GeneralComplexSubTopology(
    int id,
    std::vector<int> dimension_size,
    std::vector<CollectiveImpl*> collective_impl) {
  int offset = 1;
  uint64_t last_dim = collective_impl.size() - 1;
  int num_of_all_npus = 1;
  assert(collective_impl.size() <= dimension_size.size());
  for (uint64_t dim = 0; dim < collective_impl.size(); dim++) {
    if (collective_impl[dim]->type == CollectiveImplType::Ring ||
        collective_impl[dim]->type == CollectiveImplType::Direct ||
        collective_impl[dim]->type == CollectiveImplType::HalvingDoubling) {
      RingTopology* ring = new RingTopology(
          RingTopology::Dimension::NA,
          id,
          dimension_size[dim],
          (id % (offset * dimension_size[dim])) / offset,
          offset);
      dimension_topology.push_back(ring);
    } else if (
        collective_impl[dim]->type == CollectiveImplType::OneRing ||
        collective_impl[dim]->type == CollectiveImplType::OneDirect ||
        collective_impl[dim]->type == CollectiveImplType::OneHalvingDoubling) {
      int total_npus = 1;
      for (int d : dimension_size) {
        total_npus *= d;
      }
      RingTopology* ring = new RingTopology(
          RingTopology::Dimension::NA, id, total_npus, id % total_npus, 1);
      dimension_topology.push_back(ring);
      return;
    } else if (
        collective_impl[dim]->type == CollectiveImplType::DoubleBinaryTree) {
      if (dim == last_dim) {
        DoubleBinaryTreeTopology* DBT = new DoubleBinaryTreeTopology(
            id, dimension_size[dim], id % offset, offset);
        dimension_topology.push_back(DBT);
      } else {
        DoubleBinaryTreeTopology* DBT = new DoubleBinaryTreeTopology(
            id,
            dimension_size[dim],
            (id - (id % (offset * dimension_size[dim]))) + (id % offset),
            offset);
        dimension_topology.push_back(DBT);
      }
    }
    offset *= dimension_size[dim];
    num_of_all_npus *= dimension_size[dim];
  }

  
  for(int npu_id = 0 ; npu_id<num_of_all_npus ; npu_id++){
    std::vector<int> inner_index;
    offset  = 1;
    for (uint64_t dim = 0; dim < collective_impl.size(); dim++){
      inner_index.push_back((npu_id % (offset * dimension_size[dim])) / offset);
      offset *= dimension_size[dim];
    }
    demention_index[npu_id]=inner_index;
  }
}

GeneralComplexSubTopology::~GeneralComplexSubTopology() {
  for (uint64_t i = 0; i < dimension_topology.size(); i++) {
    delete dimension_topology[i];
  }
}

int GeneralComplexSubTopology::get_num_of_dimensions() {
  return dimension_topology.size();
}
int GeneralComplexSubTopology::get_demention_index(int id,int dim){
  return demention_index[id][dim];
}
void GeneralComplexSubTopology::set_sub_dim(std::string group_name,int dim,int id,std::vector<int> involved_npus){
  std::string id_str = std::to_string(id);
  std::string dim_str = std::to_string(dim);
  std::string sub_group_key = group_name + "_"+ dim_str + "_"+ id_str;
  sub_dim[sub_group_key]=involved_npus;
}

std::vector<int> GeneralComplexSubTopology::get_sub_dim(std::string group_name,int dim,int id){

  std::string id_str = std::to_string(id);
  std::string dim_str = std::to_string(dim);
  std::string sub_group_key = group_name + "_" + dim_str + "_" + id_str;
  if (sub_dim.find(sub_group_key) == sub_dim.end()) return {};
  return sub_dim[sub_group_key];
}
std::vector<int> GeneralComplexSubTopology::filter_npu_ids(std::vector<int> npu_ids,int query_dim,int query_id){
    if (npu_ids.empty()) {
      return {};
    }
    std::vector<int> result;
    int dims = GeneralComplexSubTopology::get_num_of_dimensions();

    for (int id : npu_ids) {
      bool should_add = true; 
      for(int dim = 0;dim<dims;dim++){
        if(dim == query_dim) continue;
        if(GeneralComplexSubTopology::get_demention_index(id, dim)!=GeneralComplexSubTopology::get_demention_index(query_id, dim)) should_add=false;
      }
      if(should_add) {result.push_back(id);}
      
    }

    return result;
}

int GeneralComplexSubTopology::get_num_of_nodes_in_dimension(int dimension) {
  if (static_cast<uint64_t>(dimension) >= dimension_topology.size()) {
    std::cout << "dim: " << dimension
              << " requested! but max dim is: " << dimension_topology.size() - 1
              << std::endl;
  }
  assert(static_cast<uint64_t>(dimension) < dimension_topology.size());
  return dimension_topology[dimension]->get_num_of_nodes_in_dimension(0);
}

BasicLogicalTopology* GeneralComplexSubTopology::get_basic_topology_at_dimension(
    int dimension,
    ComType type) {
  return dimension_topology[dimension]->get_basic_topology_at_dimension(
      0, type);
}
