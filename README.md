# VenusSim

Welcome to **VenusSim**! This is a simulator specifically designed for LLM training, developed by the **Infrawaves** team. We focus on performance and flexibility, aiming to provide a **lightweight** and **accurate** simulation environment to help developers efficiently conduct research and development work. Let's work together to enhance productivity and explore more possibilities.

[简体中文]: README_CN.md	"简体中文"

## User Guide

The usage process of VenusSim is as follows:

1. Install VenusSim.
2. Modify the configuration file.
3. Run the simulation.
4. View the simulation results.

### 1. Install VenusSim

#### 1.1 Container Construction

1. Create Container

```Plain
# Create Base Container
docker run -it --name llm-sim ubuntu:latest /bin/bash
```

2. Copy the following content to /home/build.sh

```
apt update
apt install \
    build-essential \
    python3 python3-dev python3-pip python3-venv \
    gcc g++ make cmake \
    libboost-dev libboost-program-options-dev \
    libprotobuf-dev protobuf-compiler \
    git
#Pull VenusSim
git clone git@github.com:cq-eng/VenusSim.git
#Change Directory
mv VenusSim llm-sim-project

#astra-sim
cd /home/llm-sim-project
pip3 install --upgrade pip
pip3 install protobuf==3.6.1 pydot
cd llm-simulator
chmod 777 ./build/astra_analytical/build.sh
./build/astra_analytical/build.sh

# chakra_linker_visualizer
cd /home/llm-sim-project
python3 -m venv chakra_env
source chakra_env/bin/activate
cd chakra_linker_visualizer
pip install .
cd extern/et_replay
pip install .
```

3. Run build.sh

```
bash build.sh
```

#### 1.2 File Structure

After completing the container construction, the /home directory should have the following file structure:

```Python
/home/llm-sim-project
    # Python virtual environment, used for chakra_lingker_visualizer and generator.
    /chakra_env
    # Chakra native trace_linker and timeline_visualizer.
    /chakra_linker_visualizer
    # astra-sim
    /llm-simulator
    # trace generator
    /trace-generator
```

After confirming the file structure is complete, the installation is finished.

### 2. Modify Configuration File

The configuration file is located in trace-generator/demo/llm-sim-example.

- llm-sim-config.json
- Sim-config
  - system.json
  - network.yml
  - memory.json
  - comm-group.json

#### 2.1 llm-sim-config.json

The `llm-sim-config.json` consists of three parts: **`trace_generator_config`**, **`simulator_config`**  and **`time_line_visualizer_config`**

**trace_generator_config**

This configuration file is used to configure the generation of traces. The specific parameter descriptions are as follows:

| Megatron config params                  |                                                              |
| --------------------------------------- | ------------------------------------------------------------ |
| `mlp-worker-num`                        | Machine Count                                                |
| `world-size`                            | Total GPU Count                                              |
| `pipeline-model-parallel-size`          | PP size                                                      |
| `context-parallel-size`                 | CP size                                                      |
| `tensor-model-parallel-size`            | TP size                                                      |
| `sequence-parallel`                     | SP size                                                      |
| `vpp-enable`                            | 0: Disabled  1: Enabled                                      |
| `num-layers-per-virtual-pipeline-stage` | Number of model layers included in each PP stage             |
| `global-batch-size-mlp-multiplier`      | global-batch-size = mlp-worker-num * global-batch-size-mlp-multiplier |
| `micro-batch-size`                      | Micro batch size                                             |
| `forward_only`                          | 0: Disabled  1: Enabled                                      |
| `no-overlap-p2p-communication`          | 1: Disabled  0: Enabled                                      |
| `use-distributed-optimizer`             | 0: Disabled  1: Enabled                                      |
| `overlap-grad-reduce`                   | 0: Disabled  1: Enabled                                      |
| `no-delay-grad-reduce`                  | 1: Disabled  0: Enabled                                      |
| `overlap-param-gather`                  | 0: Disabled  1: Enabled                                      |
| `delay-param-gather`                    | 0: Disabled  1: Enabled                                      |
| `train-iters`                           | Number of training iterations                                |
| **Model config params**                 |                                                              |
| `model-type`                            |                                                              |
| `seq-length`                            | Sequence Length                                              |
| `decoder-seq-length`                    | Decoder Sequence Length                                      |
| `hidden-size`                           | Hidden Layer Size                                            |
| `ffn-hidden-size`                       | FFN Hidden Layer Size                                        |
| `num-attention-heads`                   | Number of Attention Heads                                    |
| `num-query-groups`                      | Number of MQA Groups                                         |
| `num-layers`                            | Number of Model Layers                                       |
| `vocab-size`                            | Vocabulary Size                                              |
| `swiglu`                                | Swish-Gated Linear Unit    0: Disabled   1: Enabled          |
| `untie-embeddings-and-output-weights`   | Whether Input `embedding` and Output `embedding` Share Parameters  0: Not Shared   1: Shared |
| **Trace generation config params**      |                                                              |
| `output_filepath`                       | Trace output path                                            |
| `output_filename`                       | Trace output name                                            |
| `log_filename`                          | Log file name during trace generation                        |
| `trace_generate`                        | When set to 0, the trace generation phase will be skipped, typically used when the trace has already been generated and the simulator needs to be run multiple times |
| `use_chakra`                            | Use Chakra as the trace input for the simulator. When using it, you need to uncomment the line `#define Use_Chakra` in `llm-simulator/astra-sim/workload/ETDefinition.hh` from the `llm-simulator` and recompile the `llm-simulator`. |
| `use_simplified_trace`                  | Use simplified traces as the trace input for the simulator (saving significant memory compared to Chakra). When using it, you need to comment out the line `#define Use_Chakra` in `llm-simulator/astra-sim/workload/ETDefinition.hh` from the `llm-simulator` and recompile the `llm-simulator` (**this configuration is the default for the project and has been compiled accordingly during installation**). |
| `use_simplified_template_trace`         | Use simplified template traces as the trace input for the simulator (saving memory compared to simplified traces but slightly increasing runtime). When using it, you need to enable `use_simplified_trace` and meet the runtime conditions for simplified traces. |
| `total_file_size_gb_limit`              | The total size limit for trace files. If this size is exceeded, the system will delete the generated `JSON` files and retain only the `.et` files used for simulation. |
| `keep_json_files`                       | Ignore `total_file_size_gb_limit` during use and force saving `JSON` files. |
| `use_step_snapshot`                     | When set to 1, it will forcibly use the data in `step_snapshot_data` as the computation time for forward and backward passes. |
| `step_snapshot_data`                    | There are three data points, representing the sum of forward and backward times for the pp-0, pp-mid, and pp-end stages (forward: backward = 1:2). |

**simulator_config**

| binary        | The compiled executable file of `llm-simulator`, usually located in the llm-simulator/build/astra_analytical/build/bin path. |
| ------------- | ------------------------------------------------------------ |
| system        | The absolute path of `system.json` in sim-config.            |
| network       | The absolute path of `system.json` in sim-config             |
| remote_memory | The absolute path of `memory.json` in sim-config             |
| comm_group    | The absolute path of `comm_group.json` in sim-config         |
| trace_output  | The output file path for `llm-simulator`'s execution results |

**time_line_visualizer_config**

| output_filename | The output file path after visualization                     |
| --------------- | ------------------------------------------------------------ |
| single_pp_group | When set to 1, only the visualization result of one PP group will be output. When there are too many ranks, this will significantly reduce runtime and minimize redundant output. |

#### 2.2 Sim-config

- system.json
- network.yml
- memory.json （No modification required）
- comm-group.json (Automatically generated, no modification required）

For the above files, only `system.json` and `network.yml` need to be modified according to the actual simulation scenario. `system.json` contains some system parameters of the simulator, while `network.yml` is used to configure the network parameters of the simulator. `system.json` generally remains in its default configuration. For the specific meaning of `network.yml`, please refer to [Astra-Sim](https://astra-sim.github.io/astra-network-analytical-docs/input-format/input-format.html). 

Below is a specific demo:

```yaml
# Dim0 FullyConnected, Dim1 Ring.
topology: [ FullyConnected, Ring ] 

# 16 NPUs：Dim0 8 NPUs，Dim1 2 NPUs
npus_count: [ 8, 2 ]  # number of NPUs

# link Bandwidth (per direction)
bandwidth: [ 450.0, 50.0 ]  # GB/s

# link Latency
latency: [ 1000.0, 6000.0 ]  # ns
```

### 3. Run the simulation.

After confirming the installation and configuration files are correct, you can enter the following command to run the demo simulation.

```Python
# Enter the virtual environment
source /home/llm-sim-project/chakra_env/bin/activate

# Switch to the trace-generator directory
cd /home/llm-sim-project/trace-generator

# Run the simulation program, and fill in the path to llm-sim-config.json according to the actual path.
python3 llm_sim.py --config-file demo/llm-sim-example/llm-sim-config.json
```

### 4. View the simulation results

1. Open <https://ui.perfetto.dev/> in your browser.
2. Open the visualization output file specified by `time_line_visualizer_config:output_filename` in `llm-sim-config.json` at the above URL to view the visual simulation results.