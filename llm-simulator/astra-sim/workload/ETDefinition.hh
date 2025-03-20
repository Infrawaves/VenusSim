#ifndef ET_DEFINITION
#define ET_DEFINITION

#include "extern/graph_frontend/chakra/et_feeder/et_feeder.h"
#include "astra-sim/system/simplified_et/base_et/simplified_base_et.hh"
#include "astra-sim/system/simplified_et/normal_et/simplified_et_feeder.hh"
#include "astra-sim/system/simplified_et/template_et/simplified_template_et_feeder.hh"


//#define Use_Chakra
#ifdef Use_Chakra
  typedef Chakra::ETFeeder SelectETFeeder;
  typedef Chakra::ETFeederNode SelectETFeederNode;
  typedef ChakraProtoMsg::NodeType SelectNodeType;
  typedef ChakraProtoMsg::CollectiveCommType SelectCollectiveCommType;
#else
  typedef SimplET::SimplETFeederBase SelectETFeeder;
  typedef SimplET::SimplETFeederNode SelectETFeederNode;
  typedef SimplET::NodeType SelectNodeType;
  typedef SimplET::CollectiveCommType SelectCollectiveCommType;
#endif

#endif