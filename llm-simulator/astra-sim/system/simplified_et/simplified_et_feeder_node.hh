#pragma once

#include <memory>
#include <unordered_set>
#include <vector>
#include <string>
#include <limits>
#include <set>
#include <iostream>
#include <algorithm>

namespace SimplET{
enum CollectiveCommType : int {
  ALL_REDUCE = 0,
  REDUCE = 1,
  ALL_GATHER = 2,
  GATHER = 3,
  SCATTER = 4,
  BROADCAST = 5,
  ALL_TO_ALL = 6,
  REDUCE_SCATTER = 7,
  REDUCE_SCATTER_BLOCK = 8,
  BARRIER = 9,
  CollectiveCommType_INT_MIN_SENTINEL_DO_NOT_USE_ = INT32_MIN,
  CollectiveCommType_INT_MAX_SENTINEL_DO_NOT_USE_ = INT32_MAX
};

enum NodeType : int {
  INVALID_NODE = 0,
  METADATA_NODE = 1,
  MEM_LOAD_NODE = 2,
  MEM_STORE_NODE = 3,
  COMP_NODE = 4,
  COMM_SEND_NODE = 5,
  COMM_RECV_NODE = 6,
  COMM_COLL_NODE = 7,
  COMP_REPLAY_NODE = 8,
  NodeType_INT_MIN_SENTINEL_DO_NOT_USE_ = INT32_MIN,
  NodeType_INT_MAX_SENTINEL_DO_NOT_USE_ = INT32_MAX
};

class SimplETFeederNode{
public:
  SimplETFeederNode(uint64_t id, std::string name);
  ~SimplETFeederNode();

  void addChild(std::shared_ptr<SimplETFeederNode> node);
  std::vector<std::shared_ptr<SimplETFeederNode>> getChildren();

  void addDepUnresolvedParentID(uint64_t node_id);
  std::vector<uint64_t> getDepUnresolvedParentIDs();
  void setDepUnresolvedParentIDs(std::vector<uint64_t> const& dep_unresolved_parent_ids);

  void addCtrlDeps(uint64_t ctrl_deps_id);
  void removeCtrlDeps(uint64_t ctrl_deps_id);
  size_t get_ctrl_deps_size();
  std::set<uint64_t> get_ctrl_deps();

  void set_type(NodeType type);
  void set_comm_type(CollectiveCommType comm_type);

  void form_comp_node(
    uint64_t num_ops,
    uint64_t tensor_size
  );

  void form_comm_coll_node(
    std::string comm_group, 
    uint64_t comm_size
  );

  void form_comm_sendrecv_node(
    uint64_t comm_size, 
    uint64_t comm_src, 
    uint64_t comm_dst, 
    uint64_t comm_tag
  );

  void form_comp_replay_node(
    uint64_t duration
  );

  uint64_t id();
  std::string name();
  SimplET::NodeType type();

  uint64_t num_ops();
  uint64_t tensor_size();

  SimplET::CollectiveCommType comm_type();
  std::string comm_group();
  uint64_t comm_size();
  uint32_t comm_src();
  uint32_t comm_dst();
  uint32_t comm_tag();
  uint32_t comm_priority(){return 0;};
  uint64_t runtime(){return 0;};
  uint64_t duration();

private:

  //std::unordered_set<std::shared_ptr<SimplETFeederNode>> children_set_{};
  std::vector<std::shared_ptr<SimplETFeederNode>> children_vec_{};
  std::vector<uint64_t> dep_unresolved_parent_ids_{};

  std::set<uint64_t> ctrl_deps{};
  uint64_t id_;
  std::string name_;
  SimplET::NodeType type_;
  std::string comm_group_;

  // for comp node:
  //    [0] for num_ops_
  //    [1] for tensor_size_
  // for comp replay node
  //    [0] for duration
  // for comm coll node:
  //    [0] for comm_size_
  //    [1] for comm_type_
  // for comm send recv node: (with template node, [1] and [2] are not used)
  //    [0] for comm_size_
  //    [1] for comm_src_
  //    [2] for comm_dst_
  //    [3] for comm_tag_
  std::vector<uint64_t> attr_{};

  //uint64_t runtime_;
  // uint64_t num_ops_;
  // uint64_t tensor_size_;

  // SimplET::CollectiveCommType comm_type_;
  // uint64_t comm_size_;
  // uint32_t comm_src_;
  // uint32_t comm_dst_;
  // uint32_t comm_tag_;
  
};
}// namespace SimplET