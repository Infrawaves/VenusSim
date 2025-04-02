#include <fstream>
#include "common/CommonTraceParser.hh"

TraceParser::TraceParser(
    std::string trace_output_file, 
    std::string trace_output_rule, 
    std::string trace_template_mapping,
    int npus_count
){
    this->npus_count = npus_count;

    //get trace output mapping rule
    std::ifstream fin;
    this->trace_output_mapping_id.resize(npus_count, -1);
    fin.open(trace_output_rule);
    if(fin.is_open()){
        uint64_t size, index;
        fin>>size;
        for(int i=0;i<size;i++){
            fin>>index;
            this->trace_output_mapping_id[index] = i;
        }
    }
    fin.close();

    //get trace template mapping rule
    this->trace_template_mapping_id.resize(npus_count, {{"template_id", -1}, {"prev_rank", -1}, {"next_rank", -1}});
    fin.open(trace_template_mapping);
    if(fin.is_open()){
        uint64_t size, index, mapping_id, prev_rank, next_rank;
        fin>>size;
        for(int i=0;i<size;i++){
            fin>>index>>mapping_id>>prev_rank>>next_rank;
            this->trace_template_mapping_id[index]["template_id"] = mapping_id;
            this->trace_template_mapping_id[index]["prev_rank"] = prev_rank;
            this->trace_template_mapping_id[index]["next_rank"] = next_rank;
        }
    }
    fin.close();

    //get trace output file
    this->trace_output = std::make_shared<std::ofstream>(trace_output_file);
    if(!this->trace_output->is_open()){
        std::cout<<trace_output_file<<" open fail."<<std::endl;
        exit(1);
    }
}

TraceParser::~TraceParser(){
    if(this->trace_output->is_open())
        this->trace_output->close();
}

bool TraceParser::check_npu_id(int npu_id){
    return npu_id < this->npus_count && npu_id >=0;
}

std::map<std::string, int> TraceParser::get_trace_template_mapping(int npu_id){
    check_npu_id(npu_id);
    return this->trace_template_mapping_id[npu_id];
}

int TraceParser::get_trace_output_mapping(int npu_id){
    check_npu_id(npu_id);
    return this->trace_output_mapping_id[npu_id];
}

std::shared_ptr<std::ofstream> TraceParser::get_trace_output(int npu_id){
    check_npu_id(npu_id);
    return this->trace_output;
}
