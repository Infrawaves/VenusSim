/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/workload/Workload.hh"

#include <json/json.hpp>
#include "astra-sim/system/IntData.hh"
#include "astra-sim/system/MemEventHandlerData.hh"
#include "astra-sim/system/RecvPacketEventHandlerData.hh"
#include "astra-sim/system/SendPacketEventHandlerData.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/system/topology/GeneralComplexSubTopology.hh"
#include <stdlib.h>
#include <unistd.h>
#include <iostream>

using namespace std;
using namespace AstraSim;
using json = nlohmann::json;

std::map<std::string, int> comm_group_id_map = {};
int comm_group_id_generator = 0;

int get_unique_comm_group_id(){
  comm_group_id_generator += 1;
  return comm_group_id_generator;
}

string node_type_to_string(SelectNodeType type){
  switch (type)
    {
    case 0:
      return "INVALID_NODE";
    case 1:
      return "METADATA_NODE";
    case 2:
      return "MEM_LOAD_NODE";
    case 3:
      return "MEM_STORE_NODE";
    case 4:
      return "COMP_NODE";
    case 5:
      return "COMM_SEND_NODE";
    case 6:
      return "COMM_RECV_NODE";
    case 7:
      return "COMM_COLL_NODE";
    case 8:
      return "COMP_NODE";
    default:
      return "TYPE_NOT_EXIST";
    }
}

void node_type_output(ofstream &fout, string output_info){
  fout<<output_info<<endl;
}

Workload::Workload(Sys* sys, string et_filename, string comm_group_filename, std::map<std::string, int> workload_template_id) {
  string workload_filename = "";
  #ifdef Use_Chakra
    workload_filename = et_filename + "." + to_string(sys->id) + ".et";
      // Check if workload filename exists
      if (access(workload_filename.c_str(), R_OK) < 0) {
        string error_msg;
        if (errno == ENOENT) {
          error_msg = "workload file: " + workload_filename + " does not exist";
        } else if (errno == EACCES) {
          error_msg =
              "workload file: " + workload_filename + " exists but is not readable";
        } else {
          error_msg =
              "Unknown workload file: " + workload_filename + " access error";
        }
        cerr << error_msg << endl;
        exit(EXIT_FAILURE);
      }
      this->et_feeder = new SelectETFeeder(workload_filename);
  #else
    assert(sys->simplifed_trace);
    if(sys->template_trace){
      workload_filename = et_filename + "~template~." + to_string(workload_template_id["template_id"]) + ".et";
      // Check if workload filename exists
      if (access(workload_filename.c_str(), R_OK) < 0) {
        string error_msg;
        if (errno == ENOENT) {
          error_msg = "workload file: " + workload_filename + " does not exist";
        } else if (errno == EACCES) {
          error_msg =
              "workload file: " + workload_filename + " exists but is not readable";
        } else {
          error_msg =
              "Unknown workload file: " + workload_filename + " access error";
        }
        cerr << error_msg << endl;
        exit(EXIT_FAILURE);
      }
      this->et_feeder = new SimplET::SimplTempETFeeder(
        workload_filename, 
        workload_template_id["template_id"] == sys->id,
        sys->id, 
        workload_template_id["prev_rank"], 
        workload_template_id["next_rank"]
      );
      if(workload_template_id["template_id"] != sys->id){
        auto template_et = sys->all_sys[workload_template_id["template_id"]]->workload->et_feeder->get_template();
        this->et_feeder->set_template(template_et);
      }
    }
    else{
      workload_filename = et_filename + "." + to_string(sys->id) + ".et";
      // Check if workload filename exists
      if (access(workload_filename.c_str(), R_OK) < 0) {
        string error_msg;
        if (errno == ENOENT) {
          error_msg = "workload file: " + workload_filename + " does not exist";
        } else if (errno == EACCES) {
          error_msg =
              "workload file: " + workload_filename + " exists but is not readable";
        } else {
          error_msg =
              "Unknown workload file: " + workload_filename + " access error";
        }
        cerr << error_msg << endl;
        exit(EXIT_FAILURE);
      }
      this->et_feeder = new SimplET::SimplETFeeder(workload_filename);
    }
  #endif
  this->comm_group_list.clear();
  // TODO: parametrize the number of available hardware resources
  this->hw_resource = new HardwareResource(1);
  this->sys = sys;
  initialize_comm_group(comm_group_filename);
  this->is_finished = false;
  this->node_type_fout.open(et_filename + "_node_type_output.csv");
  if(!this->node_type_fout.is_open()){
    string error_msg;
    error_msg = "node-type output file" + et_filename + "_node_type_output.csv open failed.";
    cerr << error_msg << endl;
    exit(EXIT_FAILURE);
  }
}

Workload::~Workload() {
  // if (this->comm_group != nullptr)
  //   delete this->comm_group;
  if(!this->comm_group_list.empty()){
    for(auto it : this->comm_group_list){
      delete it.second;
    }
  }
  if (this->et_feeder != nullptr){
    if(sys->simplifed_trace){
      if(sys->template_trace){

      }
    }
    delete this->et_feeder;
  }
  if (this->hw_resource != nullptr)
    delete this->hw_resource;
  this->node_type_fout.close();
}

void Workload::initialize_comm_group(string comm_group_filename) {
  if(this->sys->log_level == LOG_DEBUG){
    std::cout<<">> initialize comm_group for sys "<<sys->id<<", ";
    std::cout<<"comm_group_filename is "<<comm_group_filename<<std::endl;
  }
  // communicator group input file is not given
  comm_group_list.clear();
  if (comm_group_filename.find("empty") != std::string::npos) {
    //comm_group = nullptr;
    return;
  }

  ifstream inFile;
  json j;
  inFile.open(comm_group_filename);
  inFile >> j;

  //for different comm group (tp/dp/pp...)
  for (json::iterator it = j.begin(); it != j.end(); ++it) {
    //for different small comm group in same comm group(tp [0, 1]/[2, 3]/[4, 5]...)
    for (auto id_list : it.value()) {
      
      //find sys->id
      bool in_comm_group = false;
      for(auto id : id_list){
        if (id == sys->id) {
          in_comm_group = true;
          break;
        }
      }

      //find small comm group which contains sys->id
      if(in_comm_group){
        if(comm_group_list.find(it.key()) != comm_group_list.end()){
          string error_msg = "find rank " + to_string(sys->id) + " has repeat comm group: " + it.key();
          cerr << error_msg << endl;
          exit(EXIT_FAILURE);
        }

        std::vector<int> involved_NPUs = {};
        for(auto id : id_list){
          involved_NPUs.push_back(id);
        }
        
        //get unique group name
        string comm_group_name = it.key() + "_";
        for(auto id : involved_NPUs)
          comm_group_name += to_string(id) + "_";
        
        //get namd-id map
        int comm_group_id = -1;
        if(comm_group_id_map.find(comm_group_name) != comm_group_id_map.end()){
          comm_group_id = comm_group_id_map[comm_group_name];
        }
        else{
          comm_group_id = get_unique_comm_group_id();
          comm_group_id_map[comm_group_name] = comm_group_id;
          if(this->sys->log_level == LOG_DEBUG){
            std::cout<<">>> get new comm_group: "<<comm_group_name<<"id:"<<comm_group_id<<std::endl;
          }
        }
        //create new group
        // Note: All NPUs should create comm group with identical ids if they want
        // to communicate with each other
        comm_group_list[it.key()] = new CommunicatorGroup(comm_group_id, involved_NPUs, sys, it.key());
        std::map<int, std::vector<int>> coordinates_dict = sys->logical_sub_topologies["AllReduce"]->demention_index;
        if(comm_group_list[it.key()]->if_multi_group(coordinates_dict, involved_NPUs)){
          std::cout<<"multi_group == true "<<std::endl;
          int dims = sys->logical_sub_topologies["AllReduce"]->get_num_of_dimensions();
          for (int dim=0;dim<dims;dim++){
            std::vector<int> involved_sub_npus = sys->logical_sub_topologies["AllReduce"]->filter_npu_ids(involved_NPUs,dim,sys->id);
            sys->logical_sub_topologies["AllReduce"]->set_sub_dim(it.key(),dim,sys->id,involved_sub_npus);
            sys->logical_sub_topologies["AllToAll"]->set_sub_dim(it.key(),dim,sys->id,involved_sub_npus);
            sys->logical_sub_topologies["ReduceScatter"]->set_sub_dim(it.key(),dim,sys->id,involved_sub_npus);
            sys->logical_sub_topologies["AllGather"]->set_sub_dim(it.key(),dim,sys->id,involved_sub_npus);
            if(dim==0){
              comm_group_list[it.key()]->group_dim0_gpu_num = involved_sub_npus.size();
            }
          }
          comm_group_list[it.key()]->group_gpu_num = involved_NPUs.size();
        }
        
        
        
        if(this->sys->log_level == LOG_DEBUG){
          std::cout<<">>> "<<sys->id<<" join comm_group: "<<comm_group_name<<std::endl;
        }
      }
    }
  }
}

void Workload::issue_dep_free_nodes() {
  std::queue<shared_ptr<SelectETFeederNode>> push_back_queue;
  shared_ptr<SelectETFeederNode> node = et_feeder->getNextIssuableNode();
  while (node != nullptr) {
    if (hw_resource->is_available(node)) {
      issue(node);
    } else {
      push_back_queue.push(node);
    }
    node = et_feeder->getNextIssuableNode();
  }

  while (!push_back_queue.empty()) {
    shared_ptr<SelectETFeederNode> node = push_back_queue.front();
    et_feeder->pushBackIssuableNode(node->id());
    push_back_queue.pop();
  }
}

void Workload::issue(shared_ptr<SelectETFeederNode> node) {
  // if(this->sys->log_level == LOG_DEBUG){
  //   cout  << "--------------------------issue,sys->id=" << sys->id << ",tick=" << Sys::boostedTick()
  //   << ",node->id=" << node->id() << ",node->name=" << node_type_to_string(node->type()) + "_" + node->name()
  //   << endl;
  // }

  if (sys->replay_only) {
    hw_resource->occupy(node);
    issue_replay(node);
    if (sys->trace_enabled && sys->trace_output_mapping_id != -1){
      *(sys->trace_output)  << "issue,sys->id=" << sys->id << ",tick=" << Sys::boostedTick()
                          << ",node->id=" << node->id() << ",node->name=" << node_type_to_string(node->type()) + "_" + node->name()
                          << endl;
    }
  } else {

    //node_type_output(node_type_fout, node->name() + ", " + node_type_to_string(node->type()));

    if ((node->type() == SelectNodeType::MEM_LOAD_NODE) ||
        (node->type() == SelectNodeType::MEM_STORE_NODE)) {
      if (sys->trace_enabled && sys->trace_output_mapping_id != -1) {
        *(sys->trace_output)  << "issue,sys->id=" << sys->trace_output_mapping_id << ",tick=" << Sys::boostedTick()
                              << ",node->id=" << node->id() << ",node->name=" << node_type_to_string(node->type()) + "_" + node->name()
                              << endl;
      }
      issue_remote_mem(node);
    } else if (
        // node->is_cpu_op() ||
        // (!node->is_cpu_op() && node->type() == SelectNodeType::COMP_NODE)
        node->type() == SelectNodeType::COMP_NODE ||
        node->type() == SelectNodeType::COMP_REPLAY_NODE
        
        ) {
      if ((node->runtime() == 0) && (node->num_ops() == 0)) {
        skip_invalid(node);
      } else {
        if (sys->trace_enabled && sys->trace_output_mapping_id != -1) {
          // cout<<"----------------"<<sys->id<<", "<<node->id()<<", "<<node_type_to_string(node->type()) + "_" + node->name()
          //     <<"--------------"<<node->runtime()<<","<<node->num_ops()<<endl;
          
          *(sys->trace_output)  << "issue,sys->id=" << sys->trace_output_mapping_id << ",tick=" << Sys::boostedTick()
                                << ",node->id=" << node->id() << ",node->name=" << node_type_to_string(node->type()) + "_" + node->name()
                                << endl;
        }
        issue_comp(node);
      }
    } else if (
        //!node->is_cpu_op() &&
        (node->type() == SelectNodeType::COMM_COLL_NODE ||
         (node->type() == SelectNodeType::COMM_SEND_NODE) ||
         (node->type() == SelectNodeType::COMM_RECV_NODE))) {
      if (sys->trace_enabled && sys->trace_output_mapping_id != -1) {
        *(sys->trace_output)  << "issue,sys->id=" << sys->trace_output_mapping_id << ",tick=" << Sys::boostedTick()
                              << ",node->id=" << node->id() << ",node->name=" << node_type_to_string(node->type()) + "_" + node->name()
                              << endl;
      }
      issue_comm(node);
    } else if (node->type() == SelectNodeType::INVALID_NODE) {
      skip_invalid(node);
    }
  }
}

void Workload::issue_replay(shared_ptr<SelectETFeederNode> node) {
  WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
  wlhd->node_id = node->id();
  uint64_t runtime = 1ul;
  if (node->runtime() != 0ul)
    // chakra runtimes are in microseconds and we should convert it into
    // nanoseconds
    runtime = node->runtime() * 1000;
  sys->register_event(this, EventType::General, wlhd, runtime);
}

void Workload::issue_remote_mem(shared_ptr<SelectETFeederNode> node) {
  hw_resource->occupy(node);

  WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
  wlhd->sys_id = sys->id;
  wlhd->workload = this;
  wlhd->node_id = node->id();
  sys->remote_mem->issue(node->tensor_size(), wlhd);
}

void Workload::issue_comp(shared_ptr<SelectETFeederNode> node) {
  hw_resource->occupy(node);

  bool replay_node = false;
  if (sys->roofline_enabled) {
    if(node->type() == SelectNodeType::COMP_NODE && node->tensor_size() != 0 && node->num_ops() != 0){
      WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
      wlhd->node_id = node->id();

      double operational_intensity = static_cast<double>(node->num_ops()) /
          static_cast<double>(node->tensor_size());
      double perf = sys->roofline->get_perf(operational_intensity, sys->ignore_bandwidth);
      double elapsed_time = static_cast<double>(node->num_ops()) / (perf/1E9); //NS
      // min delta time is 1ns
      elapsed_time = max(1.0, elapsed_time);
      uint64_t runtime = static_cast<uint64_t>(elapsed_time);
      sys->register_event(this, EventType::General, wlhd, runtime);
    }
    else if(node->type() == SelectNodeType::COMP_REPLAY_NODE && node->duration() != 0){
      WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
      wlhd->node_id = node->id();
      uint64_t runtime = static_cast<uint64_t>(node->duration());
      sys->register_event(this, EventType::General, wlhd, runtime);
    }
    else
      replay_node = true;
  } else 
    replay_node = true;

  if(replay_node)
    // advance this node forward the recorded "replayed" time specificed in the
    // ET.
    issue_replay(node);
}

void Workload::issue_comm(shared_ptr<SelectETFeederNode> node) {
  hw_resource->occupy(node);

  // do not use this involved_dim
  vector<bool> involved_dim = {};
  // cout<<"node->involved_dim_size(): "<<node->involved_dim_size()<<endl;
  // for (int i = 0; i < node->involved_dim_size(); i++) {
  //   involved_dim.push_back(node->involved_dim(i));
  // }

  CommunicatorGroup* involved_comm_group = nullptr;
  if(this->sys->log_level == LOG_DEBUG){
    cout<<"-----------------------------comm_group is "<<node->comm_group()<<endl;
  }
  if(comm_group_list.find(node->comm_group()) != comm_group_list.end()){
    involved_comm_group = comm_group_list[node->comm_group()];
    if(this->sys->log_level == LOG_DEBUG){
      cout<<"-----------------------------comm_group_list_size:"<<comm_group_list.size()<<std::endl;
      cout<<"-----------------------------comm_group is found"<<endl;
      cout<<"---------------------------------comm type "<<(int)node->comm_type()<<endl;
      cout<<"-----------------------------comm_size is:"<<node->comm_size()<<endl;
    }
  }
  else if(node->type() == SelectNodeType::COMM_COLL_NODE){
    std::cerr<<"comm_group for node: "<<node->name()<<"is not found."<<std::endl;
  }

  //if (!node->is_cpu_op() && (node->type() == SelectNodeType::COMM_COLL_NODE)) {
  //if (node->is_cpu_op() && (node->type() == SelectNodeType::COMM_COLL_NODE)) {
  if (node->type() == SelectNodeType::COMM_COLL_NODE) {
    if (node->comm_type() == SelectCollectiveCommType::ALL_REDUCE) {
      DataSet* fp = sys->generate_all_reduce(
          node->comm_size(), involved_dim, involved_comm_group, node->comm_priority());
      collective_comm_node_id_map[fp->my_id] = node->id();
      collective_comm_wrapper_map[fp->my_id] = fp;
      std::cout<<fp->my_id<<std::endl;
      fp->set_notifier(this, EventType::CollectiveCommunicationFinished);

    } else if (node->comm_type() == SelectCollectiveCommType::ALL_TO_ALL) {
      DataSet* fp = sys->generate_all_to_all(
          node->comm_size(), involved_dim, involved_comm_group, node->comm_priority());
      collective_comm_node_id_map[fp->my_id] = node->id();
      collective_comm_wrapper_map[fp->my_id] = fp;
      fp->set_notifier(this, EventType::CollectiveCommunicationFinished);

    } else if (node->comm_type() == SelectCollectiveCommType::ALL_GATHER) {
      DataSet* fp = sys->generate_all_gather(
          node->comm_size(), involved_dim, involved_comm_group, node->comm_priority());
      collective_comm_node_id_map[fp->my_id] = node->id();
      collective_comm_wrapper_map[fp->my_id] = fp;
      fp->set_notifier(this, EventType::CollectiveCommunicationFinished);

    } else if (node->comm_type() == SelectCollectiveCommType::REDUCE_SCATTER) {
      DataSet* fp = sys->generate_reduce_scatter(
          node->comm_size(), involved_dim, involved_comm_group, node->comm_priority());
      collective_comm_node_id_map[fp->my_id] = node->id();
      collective_comm_wrapper_map[fp->my_id] = fp;
      fp->set_notifier(this, EventType::CollectiveCommunicationFinished);

    } else if (node->comm_type() == SelectCollectiveCommType::BROADCAST) {
      // broadcast colelctive has not been implemented in ASTRA-SIM yet.
      // So, we just use its real system mesurements
      uint64_t runtime = 1ul;
      if (node->runtime() != 0ul)
        // chakra runtimes are in microseconds and we should convert it into
        // nanoseconds
        runtime = node->runtime() * 1000;
      DataSet* fp = new DataSet(1);
      fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
      collective_comm_node_id_map[fp->my_id] = node->id();
      collective_comm_wrapper_map[fp->my_id] = fp;
      sys->register_event(
          fp,
          EventType::General,
          nullptr,
          // chakra runtimes are in microseconds and we should convert it into
          // nanoseconds
          runtime);
      fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
    }
  } else if (node->type() == SelectNodeType::COMM_SEND_NODE) {
    sim_request snd_req;
    snd_req.srcRank = node->comm_src();
    snd_req.dstRank = node->comm_dst();
    snd_req.reqType = UINT8;
    SendPacketEventHandlerData* sehd = new SendPacketEventHandlerData;
    sehd->callable = this;
    sehd->wlhd = new WorkloadLayerHandlerData;
    sehd->wlhd->node_id = node->id();
    sehd->event = EventType::PacketSent;
    sys->front_end_sim_send(
        0,
        Sys::dummy_data,
        node->comm_size(),
        UINT8,
        node->comm_dst(),
        node->comm_tag(),
        &snd_req,
        &Sys::handleEvent,
        sehd);
  } else if (node->type() == SelectNodeType::COMM_RECV_NODE) {
    sim_request rcv_req;
    RecvPacketEventHandlerData* rcehd = new RecvPacketEventHandlerData;
    rcehd->wlhd = new WorkloadLayerHandlerData;
    rcehd->wlhd->node_id = node->id();
    rcehd->workload = this;
    rcehd->event = EventType::PacketReceived;
    sys->front_end_sim_recv(
        0,
        Sys::dummy_data,
        node->comm_size(),
        UINT8,
        node->comm_src(),
        node->comm_tag(),
        &rcv_req,
        &Sys::handleEvent,
        rcehd);
  } else {
    cerr << "Unknown communication node type" << endl;
    exit(EXIT_FAILURE);
  }
}

void Workload::skip_invalid(shared_ptr<SelectETFeederNode> node) {
  et_feeder->freeChildrenNodes(node->id());
  et_feeder->removeNode(node->id());
}

void Workload::call(EventType event, CallData* data) {
  if (is_finished) {
    return;
  }

  if (event == EventType::CollectiveCommunicationFinished) {
    IntData* int_data = (IntData*)data;
    uint64_t node_id = collective_comm_node_id_map[int_data->data];
    shared_ptr<SelectETFeederNode> node = et_feeder->lookupNode(node_id);

    if (sys->trace_enabled && sys->trace_output_mapping_id != -1) {
      *(sys->trace_output)  << "callback,sys->id=" << sys->trace_output_mapping_id << ",tick=" << Sys::boostedTick()
                            << ",node->id=" << node->id() << ",node->name=" << node_type_to_string(node->type()) + "_" + node->name()
                            << endl;
    }

    hw_resource->release(node);

    et_feeder->freeChildrenNodes(node_id);

    issue_dep_free_nodes();

    et_feeder->removeNode(node_id);

    // The Dataset class provides statistics that should be used later to dump
    // more statistics in the workload layer
    delete collective_comm_wrapper_map[int_data->data];
    collective_comm_wrapper_map.erase(int_data->data);

  } else {
    if (data == nullptr) {
      issue_dep_free_nodes();
    } else {
      WorkloadLayerHandlerData* wlhd = (WorkloadLayerHandlerData*)data;
      shared_ptr<SelectETFeederNode> node =
          et_feeder->lookupNode(wlhd->node_id);

      if (sys->trace_enabled && sys->trace_output_mapping_id != -1) {
        *(sys->trace_output)  << "callback,sys->id=" << sys->trace_output_mapping_id << ",tick=" << Sys::boostedTick()
                              << ",node->id=" << node->id() << ",node->name=" << node_type_to_string(node->type()) + "_" + node->name()
                              << endl;
      }

      hw_resource->release(node);

      et_feeder->freeChildrenNodes(node->id());

      issue_dep_free_nodes();

      et_feeder->removeNode(wlhd->node_id);
      delete wlhd;
    }
  }

  if (!et_feeder->hasNodesToIssue() &&
      (hw_resource->num_in_flight_cpu_ops == 0) &&
      (hw_resource->num_in_flight_gpu_comp_ops == 0) &&
      (hw_resource->num_in_flight_gpu_comm_ops == 0)) {
    report();
    is_finished = true;
  }
}

void Workload::fire() {
  call(EventType::General, NULL);
}

void Workload::report() {
  Tick curr_tick = Sys::boostedTick();
  cout << "sys[" << sys->id << "] finished, " << curr_tick << " cycles" << endl;
}
