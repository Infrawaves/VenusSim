/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __GENERAL_COMPLEX_SUB_TOPOLOGY_HH__
#define __GENERAL_COMPLEX_SUB_TOPOLOGY_HH__

#include <vector>
#include <map>
#include "astra-sim/system/topology/ComplexLogicalTopology.hh"

namespace AstraSim {

class GeneralComplexSubTopology : public ComplexLogicalTopology {
 public:
  GeneralComplexSubTopology(
      int id,
      std::vector<int> dimension_size,
      std::vector<CollectiveImpl*> collective_impl);
  ~GeneralComplexSubTopology();

  int get_num_of_dimensions() override;
  int get_num_of_nodes_in_dimension(int dimension) override;
  int get_demention_index(int id,int dim);
  void set_sub_dim(std::string group_name,int dim,int id,std::vector<int> involved_npus);
  std::vector<int> filter_npu_ids(std::vector<int> npu_ids,int dim,int id);
  std::vector<int> get_sub_dim(std::string group_name,int dim,int id);
  BasicLogicalTopology* get_basic_topology_at_dimension(
      int dimension,
      ComType type) override;

  std::vector<LogicalTopology*> dimension_topology;
  std::map<int,std::vector<int>> demention_index;
  std::map<std::string,std::vector<int>> sub_dim;
};

} // namespace AstraSim

#endif /* __GENERAL_COMPLEX_TOPOLOGY_HH__ */
