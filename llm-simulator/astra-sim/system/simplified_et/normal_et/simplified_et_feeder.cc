#include "simplified_et_feeder.hh"

using namespace std;
using namespace SimplET;

const map<string, CollectiveCommType> comm_type_mapping = {
  {"allreduce", CollectiveCommType::ALL_REDUCE}, 
  {"alltoall", CollectiveCommType::ALL_TO_ALL}, 
  {"allgather", CollectiveCommType::ALL_GATHER}, 
  {"reducescatter", CollectiveCommType::REDUCE_SCATTER}, 
  {"broadcast", CollectiveCommType::BROADCAST}
};

string string_replace(string target, string to_search, string replacement){
  string str = target;
  size_t start_pos;
  while((start_pos = str.find(to_search)) != std::string::npos) {
    str.replace(start_pos, to_search.length(), replacement);
  }
  return str;
}


SimplETFeeder::SimplETFeeder(string filename)
  : window_size_(4096 * 256), et_complete_(false) {

  this->trace_.open(filename);
  if(!this->trace_.is_open()){
    std::cerr<<"SimplETFeeder open workload failed. "<<filename<<endl;
    exit(1);
  }
  readGlobalMetadata();
  readNextWindow();
}

void SimplETFeeder::addNode(shared_ptr<SimplETFeederNode> node) {
  dep_graph_[node->id()] = node;
}

void SimplETFeeder::removeNode(uint64_t node_id) {
  dep_graph_.erase(node_id);

  if (!et_complete_
      && (dep_free_node_queue_.size() < window_size_)) {
    readNextWindow();
  }
}

bool SimplETFeeder::hasNodesToIssue() {
  return !(dep_graph_.empty() && dep_free_node_queue_.empty());
}

shared_ptr<SimplETFeederNode> SimplETFeeder::getNextIssuableNode() {
  if (dep_free_node_queue_.size() != 0) {
    shared_ptr<SimplETFeederNode> node = dep_free_node_queue_.top();
    dep_free_node_id_set_.erase(node->id());
    dep_free_node_queue_.pop();
    return node;
  } else {
    return nullptr;
  }
}

void SimplETFeeder::pushBackIssuableNode(uint64_t node_id) {
  shared_ptr<SimplETFeederNode> node = dep_graph_[node_id];
  dep_free_node_id_set_.emplace(node_id);
  dep_free_node_queue_.emplace(node);
}

shared_ptr<SimplETFeederNode> SimplETFeeder::lookupNode(uint64_t node_id) {
  return dep_graph_[node_id];
}

void SimplETFeeder::freeChildrenNodes(uint64_t node_id) {
  shared_ptr<SimplETFeederNode> node = dep_graph_[node_id];
  for (auto child: node->getChildren()) {
    child->removeCtrlDeps(node->id());
    if(child->get_ctrl_deps_size() == 0){
      dep_free_node_id_set_.emplace(child->id());
      dep_free_node_queue_.emplace(child);
    }
  }
}

void SimplETFeeder::readGlobalMetadata() {
  this->trace_>>this->total_node_size;
  this->remain_node_size = this->total_node_size;
}

shared_ptr<SimplETFeederNode> SimplETFeeder::readNode() {
  if(this->remain_node_size <= 0)
    return nullptr;
  
  // base info
  uint64_t id;
  int64_t ctrl_deps_num;
  std::string name;
  this->trace_>>id>>name>>ctrl_deps_num;

  shared_ptr<SimplETFeederNode> node = make_shared<SimplETFeederNode>(id, name);

  // ctrl deps
  uint64_t parent_id;
  bool dep_unresolved = false;
  for (int i = 0; i < ctrl_deps_num; ++i) {
    this->trace_>>parent_id;
    node->addCtrlDeps(parent_id);
    auto parent_node = dep_graph_.find(parent_id);
    if (parent_node != dep_graph_.end()) {
      parent_node->second->addChild(node);
    } else {
      dep_unresolved = true;
      node->addDepUnresolvedParentID(parent_id);
    }
  }

  // attr
  this->parseNodeType(node);
  if(node->type() == COMP_NODE){
    uint64_t num_ops, tensor_size;
    this->trace_>>num_ops>>tensor_size;
    node->form_comp_node(num_ops, tensor_size);
  }
  else if(node->type() == COMM_COLL_NODE){
    uint64_t num_elem, elem_bytes;
    string comm_group;
    this->trace_>>num_elem>>elem_bytes>>comm_group;
    node->form_comm_coll_node(comm_group, num_elem*elem_bytes);
  }
  else if(node->type() == COMM_SEND_NODE || node->type() == COMM_RECV_NODE){
    uint64_t num_elem, elem_bytes, src, dst, tag;
    this->trace_>>num_elem>>elem_bytes>>src>>dst>>tag;
    node->form_comm_sendrecv_node(num_elem*elem_bytes, src, dst, tag);
  }
  else if(node->type() == COMP_REPLAY_NODE){
    uint64_t duration;
    this->trace_>>duration;
    node->form_comp_replay_node(duration);
  }
  else {
    cerr<<"Undefined type: "<<node->type()<<" in node"<<node->name()<<endl;
    exit(1);
  }

  if (dep_unresolved) {
    dep_unresolved_node_set_.emplace(node);
  }

  this->remain_node_size--;
  return node;
}

void SimplETFeeder::parseNodeType(std::shared_ptr<SimplETFeederNode> node){
  // get node type
  if(node->name().find("sendrecv") != string::npos)
    node->set_type(COMM_SEND_NODE);
  else if(node->name().find("send") != string::npos)
    node->set_type(COMM_SEND_NODE);
  else if(node->name().find("recv") != string::npos)
    node->set_type(COMM_RECV_NODE);
  else if(node->name().find("ncclKernel") != string::npos ||
          node->name().find("ncclDevKernel") != string::npos ||
          node->name().find("c10d::") != string::npos ||
          node->name().find("nccl:") != string::npos)
    node->set_type(COMM_COLL_NODE);
  else if(node->name().find("comp_replay") != string::npos)
    node->set_type(COMP_REPLAY_NODE);
  else 
    node->set_type(COMP_NODE);
  
  // get comm type for COMM_COLL_NODE
  if(node->type() == COMM_COLL_NODE){
    string normalized_name = string_replace(string_replace(node->name(), "_", ""), "-", "");
    for(auto comm_type : comm_type_mapping){
      if(node->name().find(comm_type.first) != string::npos){
        node->set_comm_type(comm_type.second);
        break;
      }
    }
  }
}

void SimplETFeeder::resolveDep() {
  for (auto it = dep_unresolved_node_set_.begin();
      it != dep_unresolved_node_set_.end();) {
    shared_ptr<SimplETFeederNode> node = *it;
    vector<uint64_t> dep_unresolved_parent_ids = node->getDepUnresolvedParentIDs();
    for (auto inner_it = dep_unresolved_parent_ids.begin();
        inner_it != dep_unresolved_parent_ids.end();) {
      auto parent_node = dep_graph_.find(*inner_it);
      if (parent_node != dep_graph_.end()) {
        parent_node->second->addChild(node);
        inner_it = dep_unresolved_parent_ids.erase(inner_it);
      } else {
        ++inner_it;
      }
    }
    if (dep_unresolved_parent_ids.size() == 0) {
      it = dep_unresolved_node_set_.erase(it);
    } else {
      node->setDepUnresolvedParentIDs(dep_unresolved_parent_ids);
      ++it;
    }
  }
}

void SimplETFeeder::readNextWindow() {
  //assert(this->remain_node_size <= this->total_node_size);
  uint32_t num_read = 0;
  do {
    shared_ptr<SimplETFeederNode> new_node = readNode();
    if (new_node == nullptr) {
      et_complete_ = true;
      break;
    }

    addNode(new_node);
    ++num_read;

    resolveDep();
  } while ((num_read < window_size_)
      || (dep_unresolved_node_set_.size() != 0));

  for (auto node_id_node: dep_graph_) {
    uint64_t node_id = node_id_node.first;
    shared_ptr<SimplETFeederNode> node = node_id_node.second;
    if ((dep_free_node_id_set_.count(node_id) == 0)
        && (node->get_ctrl_deps_size() == 0)) {
      dep_free_node_id_set_.emplace(node_id);
      dep_free_node_queue_.emplace(node);
    }
  }
}
