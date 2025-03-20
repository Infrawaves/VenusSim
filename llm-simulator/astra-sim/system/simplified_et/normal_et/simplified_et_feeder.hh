#pragma once

#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <map>

#include "astra-sim/system/simplified_et/simplified_et_feeder_node.hh"
#include "astra-sim/system/simplified_et/template_et/simplified_et_template.hh"
#include "astra-sim/system/simplified_et/base_et/simplified_base_et.hh"

namespace SimplET{
struct CompareNodes: public std::binary_function<std::shared_ptr<SimplETFeederNode>, std::shared_ptr<SimplETFeederNode>, bool>
{
  bool operator()(const std::shared_ptr<SimplETFeederNode> lhs, const std::shared_ptr<SimplETFeederNode> rhs) const
  {
    return lhs->id() > rhs->id();
  }
};

class SimplETFeeder : public SimplETFeederBase {
 public:
  SimplETFeeder(std::string filename);
  ~SimplETFeeder() override {};

  //not used
  std::shared_ptr<SimplETTemplate> get_template(){assert(false);};
  void set_template(std::shared_ptr<SimplETTemplate> template_et){assert(false);};

  void removeNode(uint64_t node_id);
  bool hasNodesToIssue();
  std::shared_ptr<SimplETFeederNode> getNextIssuableNode();
  void pushBackIssuableNode(uint64_t node_id);
  std::shared_ptr<SimplETFeederNode> lookupNode(uint64_t node_id);
  void freeChildrenNodes(uint64_t node_id);

 private:

  void addNode(std::shared_ptr<SimplETFeederNode> node);
  void parseNodeType(std::shared_ptr<SimplETFeederNode> node);

  void readGlobalMetadata();
  std::shared_ptr<SimplETFeederNode> readNode();
  void readNextWindow();
  void resolveDep();

  std::ifstream trace_;
  const uint32_t window_size_;
  uint32_t total_node_size;
  uint32_t remain_node_size;
  bool et_complete_;

  std::unordered_map<uint64_t, std::shared_ptr<SimplETFeederNode>> dep_graph_{};
  std::unordered_set<uint64_t> dep_free_node_id_set_{};
  std::priority_queue<std::shared_ptr<SimplETFeederNode>, std::vector<std::shared_ptr<SimplETFeederNode>>, CompareNodes> dep_free_node_queue_{};
  std::unordered_set<std::shared_ptr<SimplETFeederNode>> dep_unresolved_node_set_{};
};
} // namespace SimplET

