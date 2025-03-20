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

class SimplETTemplate;
class SimplTempETFeeder : public SimplETFeederBase {
 public:
  SimplTempETFeeder(std::string filename, bool is_template=false, uint64_t self_id=0, uint64_t prev_id=0, uint64_t next_id=0);
  ~SimplTempETFeeder() override {};

  std::shared_ptr<SimplETTemplate> get_template();
  void set_template(std::shared_ptr<SimplETTemplate> template_et);

  void removeNode(uint64_t node_id);
  bool hasNodesToIssue();
  std::shared_ptr<SimplETFeederNode> getNextIssuableNode();
  void pushBackIssuableNode(uint64_t node_id);
  std::shared_ptr<SimplETFeederNode> lookupNode(uint64_t node_id);
  void freeChildrenNodes(uint64_t node_id);

 private:

  // void addNode(std::shared_ptr<SimplETFeederNode> node);
  // void parseNodeType(std::shared_ptr<SimplETFeederNode> node);

  // void readGlobalMetadata();
  // std::shared_ptr<SimplETFeederNode> readNode();
  // void readNextWindow();
  // void resolveDep();

  void set_dep_free_node_id_queue(std::unordered_set<uint64_t> init_dep_free_node_id_set);
  std::shared_ptr<SimplETFeederNode> generate_node_from_template(std::shared_ptr<SimplETFeederNode> node_template);

  uint64_t self_id, prev_id, next_id;
  uint64_t total_remove_node_;
  std::map<uint64_t, std::set<uint64_t>> monitoring_nodes_{};
  std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<uint64_t>> dep_free_node_id_queue_{};
  std::shared_ptr<SimplETTemplate> template_et_;

};
} // namespace SimplET

