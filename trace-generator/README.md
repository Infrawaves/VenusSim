# Trace-Generator
## Setup
Please ensure that the following preparations have been completed
### Virtual Env
 Setup the virtual env created in [chakra_linker_visualizer](http://183.207.7.174:8081/moon/llm-simulator/chakra_linker_visualizer.git) and enable it
```bash
$ source path/to/vir_env/bin/activate
```

### LLM-Simulator
 Complete the relevant settings for the [llm-simulator](http://183.207.7.174:8081/moon/llm-simulator/llm-simulator), successfully built it to obtain AstraSim_Analytical_Congestion_Aware and get relevant config files


## Run
```bash
$ path/to/vir_env/bin/python3 path/to/trace-generator/llm_sim.py --config-file path/to/llm-sim-config.json
```

## Output file
- Generated-Trace file will be created in path/in/llm-sim-config.json:"trace_generator_config":"output_filepath"/
- A timeline json will be created in path/in/llm-sim-config.json:"simulator_config":"trace_output"/
- Several log files will be created.