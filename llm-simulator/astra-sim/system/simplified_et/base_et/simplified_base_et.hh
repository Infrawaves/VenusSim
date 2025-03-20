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

namespace SimplET{

class SimplETTemplate;
class SimplETFeederBase {
 public:

  virtual ~SimplETFeederBase(){};
  virtual std::shared_ptr<SimplETTemplate> get_template() = 0;
  virtual void set_template(std::shared_ptr<SimplETTemplate> template_et) = 0;

  virtual void removeNode(uint64_t node_id) = 0;
  virtual bool hasNodesToIssue() = 0;
  virtual std::shared_ptr<SimplETFeederNode> getNextIssuableNode() = 0;
  virtual void pushBackIssuableNode(uint64_t node_id) = 0;
  virtual std::shared_ptr<SimplETFeederNode> lookupNode(uint64_t node_id) = 0;
  virtual void freeChildrenNodes(uint64_t node_id) = 0;
};
} // namespace SimplET

