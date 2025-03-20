/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_unaware/BasicTopology.hh"
#include <cassert>
#include "common/NetworkFunction.hh"
#include <iostream>
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionUnaware;
using namespace std;
BasicTopology::BasicTopology(
    const int npus_count,
    const Bandwidth bandwidth,
    const Latency latency) noexcept
    : latency(latency),
      basic_topology_type(TopologyBuildingBlock::Undefined),
      Topology() {
  assert(npus_count > 0);
  assert(bandwidth > 0);
  assert(latency >= 0);

  // set topology shape
  this->npus_count = npus_count;
  npus_count_per_dim.push_back(npus_count);
  dims_count = 1;
  this->bandwidth = bandwidth;
  bandwidth_per_dim.push_back(bandwidth);

  // translate bandwidth from GB/s to B/ns
  bandwidth_Bpns = bw_GBps_to_Bpns(bandwidth);

  // init random
  this->gen_ptr = new std::mt19937(this->rd());
  this->random_comm_delay = new std::normal_distribution<double>(0.0, 0.01);
}

// default destructor
BasicTopology::~BasicTopology(){
  delete this->gen_ptr;
  this->gen_ptr = nullptr;
  delete this->random_comm_delay;
  this->random_comm_delay = nullptr;
}

EventTime BasicTopology::send(
    const DeviceId src,
    const DeviceId dest,
    const ChunkSize chunk_size) const noexcept {
  assert(0 <= src && src < npus_count);
  assert(0 <= dest && dest < npus_count);
  assert(src != dest);
  assert(chunk_size > 0);

  // get hops count
  auto hops_count = compute_hops_count(src, dest);

  // return communication delay
  return compute_communication_delay(hops_count, chunk_size);
}

EventTime BasicTopology::compute_communication_delay(
    const int hops_count,
    const ChunkSize chunk_size) const noexcept {
  assert(hops_count > 0);
  assert(chunk_size > 0);

  // compute link delay and serialization delay
  auto link_delay = hops_count * latency;
  auto serialization_delay = static_cast<double>(chunk_size) / bandwidth_Bpns;

  // comms_delay is the summation of the two
  float dim0_npu_num =  std::get<0>(nccl_correct_factor);
  float group_npu_num = std::get<1>(nccl_correct_factor);
  bool align_to_nccl = std::get<2>(nccl_correct_factor);
  float aligh_factor = (group_npu_num-1)/(group_npu_num-dim0_npu_num);
  auto comms_delay = link_delay + serialization_delay;
  if(align_to_nccl&&group_npu_num>1&&((int)group_npu_num>(int)dim0_npu_num)){
    comms_delay = link_delay+int(serialization_delay*aligh_factor);
  }
  double ramdom = (*this->random_comm_delay)((*this->gen_ptr));
  comms_delay = comms_delay * (1 + fabs((*this->random_comm_delay)((*this->gen_ptr))));

  // return EventTime type of comms_delay
  return static_cast<EventTime>(comms_delay);
}

TopologyBuildingBlock BasicTopology::get_basic_topology_type() const noexcept {
  assert(basic_topology_type != TopologyBuildingBlock::Undefined);

  return basic_topology_type;
}
