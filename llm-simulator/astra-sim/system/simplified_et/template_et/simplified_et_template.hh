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

namespace SimplET{

class SimplETTemplate{
 public:
  SimplETTemplate(std::string filename);
  ~SimplETTemplate();

  bool dep_graph_empty(){return this->dep_graph_.empty();};
  size_t dep_graph_size(){return this->dep_graph_.size();};
  std::shared_ptr<SimplETFeederNode> lookupNode(uint64_t node_id);
  std::unordered_set<uint64_t> get_init_dep_free_node_id_set(){return std::unordered_set<uint64_t>(init_dep_free_node_id_set_);};

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
  bool use_template_;
  std::unordered_map<uint64_t, std::shared_ptr<SimplETFeederNode>> dep_graph_{};
  std::unordered_set<std::shared_ptr<SimplETFeederNode>> dep_unresolved_node_set_{};
  std::unordered_set<uint64_t> init_dep_free_node_id_set_{};
};
} // namespace SimplET

