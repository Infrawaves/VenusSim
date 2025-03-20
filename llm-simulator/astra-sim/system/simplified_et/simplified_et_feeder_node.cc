#include "simplified_et_feeder_node.hh"

using namespace std;
using namespace SimplET;

SimplETFeederNode::SimplETFeederNode(uint64_t id, string name) {
  this->id_ = id;
  this->name_ = name;
}

SimplETFeederNode::~SimplETFeederNode(){
  //this->children_set_.clear();
  this->children_vec_.clear();
  this->dep_unresolved_parent_ids_.clear();
}

void SimplETFeederNode::addChild(shared_ptr<SimplETFeederNode> node) {
  // Avoid adding the same child node multiple times
  // addChild is called multiple times to resolve dependencies
  // if (children_set_.find(node) != children_set_.end()) {
  //   return;
  // }
  // children_set_.emplace(node);
  if (find(children_vec_.begin(), children_vec_.end(), node) != children_vec_.end()){
   return; 
  }
  children_vec_.emplace_back(node);
}

vector<shared_ptr<SimplETFeederNode>> SimplETFeederNode::getChildren() {
  return children_vec_;
}

void SimplETFeederNode::addDepUnresolvedParentID(uint64_t node_id) {
  dep_unresolved_parent_ids_.emplace_back(node_id);
}

vector<uint64_t> SimplETFeederNode::getDepUnresolvedParentIDs() {
  return dep_unresolved_parent_ids_;
}

void SimplETFeederNode::setDepUnresolvedParentIDs(
    vector<uint64_t> const& dep_unresolved_parent_ids) {
  dep_unresolved_parent_ids_ = dep_unresolved_parent_ids;
}

void SimplETFeederNode::addCtrlDeps(uint64_t ctrl_deps_id){
  this->ctrl_deps.insert(ctrl_deps_id);
}

void SimplETFeederNode::removeCtrlDeps(uint64_t ctrl_deps_id){
  this->ctrl_deps.erase(ctrl_deps_id);
}

size_t SimplETFeederNode::get_ctrl_deps_size(){
  return this->ctrl_deps.size();
}

std::set<uint64_t> SimplETFeederNode::get_ctrl_deps(){
  return std::set<uint64_t>(this->ctrl_deps);
}

void SimplETFeederNode::set_type(NodeType type){
  this->type_ = type;
}

void SimplETFeederNode::form_comp_node(uint64_t num_ops, uint64_t tensor_size){
  if(this->type_ != COMP_NODE){
    cerr<<"Wrong type in node "<<this->name_<<" when form_comp_node."<<endl;
    exit(1);
  }
  this->attr_.push_back(num_ops);
  this->attr_.push_back(tensor_size);
  // this->num_ops_ = num_ops;
  // this->tensor_size_ = tensor_size;
}

void SimplETFeederNode::form_comm_coll_node(
  std::string comm_group, 
  uint64_t comm_size
){
  if(this->type_ != COMM_COLL_NODE){
    cerr<<"Wrong type in node "<<this->name_<<" when form_comm_coll_node."<<endl;
    exit(1);
  }
  if(this->attr_.size() == 0){
    this->attr_.push_back(comm_size);
    //prepare for comm_type
    this->attr_.push_back(0);
  }
  else if(this->attr_.size() == 2)
    this->attr_[0] = comm_size;
  this->comm_group_ = comm_group;
  // this->comm_size_ = comm_size;
}

void SimplETFeederNode::form_comm_sendrecv_node(
  uint64_t comm_size, 
  uint64_t comm_src, 
  uint64_t comm_dst, 
  uint64_t comm_tag
){
  if(this->type_ != COMM_SEND_NODE && this->type_ != COMM_RECV_NODE){
    cerr<<"Wrong type in node "<<this->name_<<" when form_comm_sendrecv_node."<<endl;
    exit(1);
  }
  this->attr_.push_back(comm_size);
  this->attr_.push_back(comm_src);
  this->attr_.push_back(comm_dst);
  this->attr_.push_back(comm_tag);
  // this->comm_size_ = comm_size;
  // this->comm_src_ = comm_src;
  // this->comm_dst_ = comm_dst;
  // this->comm_tag_ = comm_tag;
}

void SimplETFeederNode::form_comp_replay_node(uint64_t duration){
  if(this->type_ != COMP_REPLAY_NODE){
    cerr<<"Wrong type in node "<<this->name_<<" when form_comp_replay_node."<<endl;
    exit(1);
  }
  this->attr_.push_back(duration);
}

void SimplETFeederNode::set_comm_type(CollectiveCommType comm_type){
  if(this->type_ != COMM_COLL_NODE){
    cerr<<"Wrong type in node "<<this->name_<<" when set_comm_type."<<endl;
    exit(1);
  }
  if(this->attr_.size() == 0){
    //prepare for comm_size_
    this->attr_.push_back(0);
    this->attr_.push_back((uint64_t)(comm_type));
  }
  else if(this->attr_.size() == 2)
    this->attr_[1] = (uint64_t)(comm_type);
  //this->comm_type_ = comm_type;
}

uint64_t SimplETFeederNode::id() {
  return id_;
}

string SimplETFeederNode::name() {
  return name_;
}

string SimplETFeederNode::comm_group() {
  return comm_group_;
}

SimplET::NodeType SimplETFeederNode::type() {
  return this->type_;
}

uint64_t SimplETFeederNode::num_ops() {
  return this->attr_[0];
  //return num_ops_;
}

uint64_t SimplETFeederNode::tensor_size() {
  return this->attr_[1];
  //return tensor_size_;
}

SimplET::CollectiveCommType SimplETFeederNode::comm_type() {
  return (SimplET::CollectiveCommType)this->attr_[1];
  //return comm_type_;
}

uint64_t SimplETFeederNode::comm_size() {
  return this->attr_[0];
  //return comm_size_;
}

uint32_t SimplETFeederNode::comm_src() {
  return this->attr_[1];
  //return comm_src_;
}

uint32_t SimplETFeederNode::comm_dst() {
  return this->attr_[2];
  //return comm_dst_;
}

uint32_t SimplETFeederNode::comm_tag() {
  return this->attr_[3];
  //return comm_tag_;
}

uint64_t SimplETFeederNode::duration(){
  return this->attr_[0];
}