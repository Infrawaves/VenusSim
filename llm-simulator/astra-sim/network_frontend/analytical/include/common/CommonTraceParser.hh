#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <cstring>
#include <memory>

class TraceParser{
    public:
        TraceParser(
            std::string trace_output_file, 
            std::string trace_output_rule, 
            std::string trace_template_mapping,
            int npus_count
        );
        ~TraceParser();
        std::map<std::string, int> get_trace_template_mapping(int npu_id);
        int get_trace_output_mapping(int npu_id);
        std::shared_ptr<std::ofstream> get_trace_output(int npu_id);
    private:
        bool check_npu_id(int npu_id);
        int npus_count = 0;
        std::vector<std::map<std::string, int>> trace_template_mapping_id = {};
        std::vector<int> trace_output_mapping_id = {};
        std::shared_ptr<std::ofstream> trace_output = nullptr;
};
