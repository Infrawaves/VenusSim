#include "simplified_template_et_feeder.hh"
#include <cassert>

using namespace std;
using namespace SimplET;

SimplTempETFeeder::SimplTempETFeeder(
  string filename, bool is_template, 
  uint64_t self_id, uint64_t prev_id, uint64_t next_id
){
  this->self_id = self_id;
  this->prev_id = prev_id;
  this->next_id = next_id;
  this->total_remove_node_ = 0;
  monitoring_nodes_.clear();

  if(is_template){
    auto template_et = make_shared<SimplETTemplate>(filename);
    this->set_template(template_et);
  }
  else
    this->template_et_ = nullptr;

  // readGlobalMetadata();
  // readNextWindow();
}

// void SimplTempETFeeder::addNode(shared_ptr<SimplETFeederNode> node) {
//   dep_graph_[node->id()] = node;
// }

void SimplTempETFeeder::set_dep_free_node_id_queue(std::unordered_set<uint64_t> init_dep_free_node_id_set){
  for(uint64_t node_id : init_dep_free_node_id_set){
    dep_free_node_id_queue_.push(node_id);
  }
}

shared_ptr<SimplETTemplate> SimplTempETFeeder::get_template(){
  assert(template_et_ != nullptr);
  return template_et_;
}

void SimplTempETFeeder::set_template(shared_ptr<SimplETTemplate> template_et){
  template_et_ = template_et;
  auto dep_free_node_id_set = this->template_et_->get_init_dep_free_node_id_set();
  this->set_dep_free_node_id_queue(dep_free_node_id_set);
}

std::shared_ptr<SimplETFeederNode> SimplTempETFeeder::generate_node_from_template(
  std::shared_ptr<SimplETFeederNode> node_template
){
  shared_ptr<SimplETFeederNode> node = make_shared<SimplETFeederNode>(node_template->id(), node_template->name());
  node->set_type(node_template->type());
  if(node_template->type() == COMP_NODE){
    node->form_comp_node(node_template->num_ops(), node_template->tensor_size());
  }
  else if(node->type() == COMM_COLL_NODE){
    node->set_comm_type(node_template->comm_type());
    node->form_comm_coll_node(node_template->comm_group(), node_template->comm_size());
  }
  else if(node->type() == COMM_SEND_NODE || node->type() == COMM_RECV_NODE){
    uint64_t src, dst;
    
    if(node_template->comm_src() == 0)
      src = this->self_id;
    else if(node_template->comm_src() == -1)
      src = this->prev_id;
    else if(node_template->comm_src() == 1)
      src = this->next_id;
    else {//shouldn't reach here
      std::cerr<<"illegal src found. "<<src<<std::endl;
      exit(1);
    }

    if(node_template->comm_dst() == 0)
      dst = this->self_id;
    else if(node_template->comm_dst() == -1)
      dst = this->prev_id;
    else if(node_template->comm_dst() == 1)
      dst = this->next_id;
    else {//shouldn't reach here
      std::cerr<<"illegal dst found. "<<dst<<std::endl;
      exit(1);
    }
    
    node->form_comm_sendrecv_node(node_template->comm_size(), src, dst, node_template->comm_tag());
  }
  else if(node->type() == COMP_REPLAY_NODE){
    node->form_comp_replay_node(node_template->duration());
  }
  else {
    cerr<<"Undefined type: "<<node->type()<<" in node"<<node->name()<<endl;
    exit(1);
  }

  return node;
}

void SimplTempETFeeder::removeNode(uint64_t node_id) {
  // dep_graph_.erase(node_id);

  // if (!et_complete_
  //     && (dep_free_node_queue_.size() < window_size_)) {
  //   readNextWindow();
  // }
  total_remove_node_ += 1;
}

bool SimplTempETFeeder::hasNodesToIssue() {
  return !(dep_free_node_id_queue_.empty() && total_remove_node_ >= template_et_->dep_graph_size());
}

shared_ptr<SimplETFeederNode> SimplTempETFeeder::getNextIssuableNode() {
  if (dep_free_node_id_queue_.size() != 0) {
    uint64_t node_id = dep_free_node_id_queue_.top();
    shared_ptr<SimplETFeederNode> node_template = template_et_->lookupNode(node_id);
    shared_ptr<SimplETFeederNode> node = generate_node_from_template(node_template);
    dep_free_node_id_queue_.pop();
    // shared_ptr<SimplETFeederNode> node = dep_free_node_queue_.top();
    // dep_free_node_id_set_.erase(node->id());
    // dep_free_node_queue_.pop();
    return node;
  } else {
    return nullptr;
  }
}

void SimplTempETFeeder::pushBackIssuableNode(uint64_t node_id) {
  dep_free_node_id_queue_.emplace(node_id);
  // shared_ptr<SimplETFeederNode> node = dep_graph_[node_id];
  // dep_free_node_id_set_.emplace(node_id);
  // dep_free_node_queue_.emplace(node);
}

shared_ptr<SimplETFeederNode> SimplTempETFeeder::lookupNode(uint64_t node_id) {
  shared_ptr<SimplETFeederNode> node_template = template_et_->lookupNode(node_id);
  shared_ptr<SimplETFeederNode> node = generate_node_from_template(node_template);
  return node;
  // return dep_graph_[node_id];
}

void SimplTempETFeeder::freeChildrenNodes(uint64_t node_id) {
  // shared_ptr<SimplETFeederNode> node = dep_graph_[node_id];
  shared_ptr<SimplETFeederNode> node = template_et_->lookupNode(node_id);
  for (auto child: node->getChildren()) {
    if(monitoring_nodes_.find(child->id()) == monitoring_nodes_.end()){
      monitoring_nodes_[child->id()] = std::set<uint64_t>(child->get_ctrl_deps());
    }
    monitoring_nodes_[child->id()].erase(node->id());
    if(monitoring_nodes_[child->id()].empty()){
      dep_free_node_id_queue_.emplace(child->id());
      monitoring_nodes_.erase(child->id());
    }
    //child->removeCtrlDeps(node->id());
    // if(child->get_ctrl_deps_size() == 0){
    //   dep_free_node_id_set_.emplace(child->id());
    //   dep_free_node_queue_.emplace(child);
    // }
  }
}
