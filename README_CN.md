# VenusSim

欢迎使用**VenusSim**！这是由**Infrawaves**团队开发的一款专为LLM训练设计的仿真模拟器。我们注重性能与灵活性，旨在提供一个**轻量**且**精确**的模拟环境，帮助开发者高效地进行研究和开发工作。让我们一起提升工作效率，探索更多可能性。

## 使用指南

VenusSim的使用流程如下：

1. 安装VenusSim。
2. 修改配置文件。
3. 运行仿真。
4. 查看仿真结果。

### 1. 安装VenusSim

#### 1.1 容器构建

1. 创建容器

```Plain
# 创建基础容器
docker run -it --name llm-sim ubuntu:latest /bin/bash
```

2. 将以下内容拷贝到/home/build.sh

```
apt update
apt install \
    build-essential \
    python3 python3-dev python3-pip python3-venv \
    gcc g++ make cmake \
    libboost-dev libboost-program-options-dev \
    libprotobuf-dev protobuf-compiler \
    git
#拉取VenusSim
git clone git@github.com:cq-eng/VenusSim.git

#astra-sim
cd /home/VenusSim
pip3 install --upgrade pip
pip3 install protobuf==3.6.1 pydot
cd llm-simulator
chmod 777 ./build/astra_analytical/build.sh
./build/astra_analytical/build.sh

# chakra_linker_visualizer
cd /home/VenusSim
python3 -m venv chakra_env
source chakra_env/bin/activate
cd chakra_linker_visualizer
pip install .
cd extern/et_replay
pip install .
```

3. 运行build.sh

```
bash build.sh
```

#### 1.2 文件结构

完成容器构建后，/home目录下应有如下文件结构：

```Python
/home/VenusSim
    # python虚拟环境，用于chakra_lingker_visualizer和generator
    /chakra_env
    # chakra原生trace_linker和timeline_visualizer
    /chakra_linker_visualizer
    # astra-sim
    /llm-simulator
    # trace生成器
    /trace-generator
```

确认文件结构完整后，即安装完成。

### 2. 修改配置文件

配置文件位于trace-generator/demo/VenusSim中

- llm-sim-config.json
- Sim-config
  - system.json
  - network.yml
  - memory.json （无需修改）
  - comm-group.json (自动生成，无需修改）

#### 2.1 llm-sim-config.json

llm-sim-config.json由**trace_generator_config**, **simulator_config**和**time_line_visualizer_config**三部分组成

**trace_generator_config**

该配置文件用来配置trace的生成，具体参数说明如下：

| **megatron参数配置**                    |                                                              |
| --------------------------------------- | ------------------------------------------------------------ |
| `mlp-worker-num`                        | 仿真机器数目                                                 |
| `world-size`                            | 总GPU数目                                                    |
| `pipeline-model-parallel-size`          | PP size                                                      |
| `context-parallel-size`                 | CP size                                                      |
| `tensor-model-parallel-size`            | TP size                                                      |
| `sequence-parallel`                     | SP size                                                      |
| `vpp-enable`                            | 是否开启 VPP     0：不开启   1：开启                         |
| `num-layers-per-virtual-pipeline-stage` | 开启VPP时，每个PP stage包含的的模型层数                      |
| `global-batch-size-mlp-multiplier`      | global-batch-size-mlp-multiplier含义为：global-batch-size = mlp-worker-num * global-batch-size-mlp-multiplier |
| `micro-batch-size`                      | Micro batch size                                             |
| `forward_only`                          | megatron是否只开启forward    0：不开启   1：开启             |
| `no-overlap-p2p-communication`          | 是否开启p2p overlap    1：不开启   0：开启                   |
| `use-distributed-optimizer`             | 是否开启distributed-optimizer    0：不开启   1：开启         |
| `overlap-grad-reduce`                   | 是否开启grad reduce overlap   0：不开启   1：开启            |
| `no-delay-grad-reduce`                  | 是否开启grad-reduce延时    1：不开启   0：开启               |
| `overlap-param-gather`                  | 是否开启param-gathe overlap    0：不开启   1：开启           |
| `delay-param-gather`                    | 是否开启param-gather延时    0：不开启   1：开启              |
| `train-iters`                           | 训练迭代数目                                                 |
| **模型参数配置**                        |                                                              |
| `model-type`                            |                                                              |
| `seq-length`                            | 序列长度                                                     |
| `decoder-seq-length`                    | 解码器序列长度                                               |
| `hidden-size`                           | 隐藏层大小                                                   |
| `ffn-hidden-size`                       | FFN隐藏层大小                                                |
| `num-attention-heads`                   | 注意力头数目                                                 |
| `num-query-groups`                      | MQA组数                                                      |
| `num-layers`                            | 模型层数                                                     |
| `vocab-size`                            | 词表大小                                                     |
| `swiglu`                                | 是否开启Swish-Gated Linear Unit    0：不开启   1：开启       |
| `untie-embeddings-and-output-weights`   | 输入`embedding`和输出`embedding`是否共享参数   0：不共享   1：共享 |
| **trace生成参数配置**                   |                                                              |
| `output_filepath`                       | trace输出路径                                                |
| `output_filename`                       | trace输出名称                                                |
| `log_filename`                          | 生成trace时的log文件名称                                     |
| `trace_generate`                        | 设置为0时将跳过trace生成阶段，一般在已经生成过trace想多次运行simulaor时使用。 |
| `use_chakra`                            | 使用chakra作为simulator的trace输入。使用时需将[llm-simulator](http://183.207.7.174:8081/moon/llm-simulator/llm-simulator)中的：llm-simulator/astra-sim/workload/ETDefinition.hh中的“#define Use_Chakra”取消注释并重新编译llm-simulator。 |
| `use_simplified_trace`                  | 使用简化trace作为simulator的trace输入（相比chakra节省大量内存）使用时需将[llm-simulator](http://183.207.7.174:8081/moon/llm-simulator/llm-simulator)中的：llm-simulator/astra-sim/workload/ETDefinition.hh中的“#define Use_Chakra”注释并重新编译llm-simulator(**项目默认使用此配置，安装时已按该配置编译**)。 |
| `use_simplified_template_trace`         | 使用简化的模板trace作为simulator的trace输入（相比简化trace节省内存，但略微增加运行时间）使用时需开启use_simplified_trace并满足简化trace的运行条件。 |
| `total_file_size_gb_limit`              | trace文件的总大小限制，超过此大小系统将删除生成的json文件，只保留用于仿真的.et文件。 |
| `keep_json_files`                       | 使用时忽略total_file_size_gb_limit，强制保存json文件         |
| `use_step_snapshot`                     | 设置为1时，将强制使用step_snapshot_data中的数据作为前后向的计算时间 |
| `step_snapshot_data`                    | 有三个数据，分别为pp-0, pp-mid, pp-end三个阶段的前后向时间之和（前向：后向=1：2） |

**simulator_config**

| binary        | llm-simulator编译后的可执行文件，一般在llm-simulator/build/astra_analytical/build/bin路径中 |
| ------------- | ------------------------------------------------------------ |
| system        | sim-config中的system.json的绝对路径                          |
| network       | sim-config中的network.yml的绝对路径                          |
| remote_memory | sim-config中的memory.json的绝对路径                          |
| comm_group    | sim-config中的comm_group.json的绝对路径                      |
| trace_output  | llm-simulator的运行结果输出文件路径                          |

**time_line_visualizer_config**

| output_filename | 可视化后的输出文件路径                                       |
| --------------- | ------------------------------------------------------------ |
| single_pp_group | 使用时将只输出一个pp组的可视化结果，当rank过多时，将显著减少运行时间，并减少重复性输出 |

#### 2.2 Sim-config

- system.json
- network.yml
- memory.json （无需修改）
- comm-group.json (自动生成，无需修改）

上述文件只要根据实际仿真场景修改`system.json`和`network.yml`。`system.json`包含仿真器的一些系统参数，`network.yml`用来配置仿真器的网络参数。`system.json` 一般保持默认配置，`network.yml`的具体含义可参考[Astra-Sim](https://astra-sim.github.io/astra-network-analytical-docs/input-format/input-format.html),以下为一个具体demo：

```yaml
# 一维FullyConnected，二维Ring
topology: [ FullyConnected, Ring ] 

# 16 NPUs：一维8卡，二维2卡
npus_count: [ 8, 2 ]  # number of NPUs

# 每一维 link Bandwidth (per direction)
bandwidth: [ 450.0, 50.0 ]  # GB/s

# 每一维 link Latency
latency: [ 1000.0, 6000.0 ]  # ns
```

### 3. 运行仿真

确认安装和配置文件无误后，可输入下列指令进行demo的仿真

```Python
# 进入虚拟环境
source /home/VenusSim/chakra_env/bin/activate

# 进入trace-generator目录
cd /home/VenusSim/trace-generator

# 运行仿真程序,llm-sim-config.json的路径根据实际路径填写
python3 llm_sim.py --config-file demo/llm-sim-example/llm-sim-config.json
```

### 4. 查看仿真结果

1. 在浏览器中打开https://ui.perfetto.dev/
2. 将`llm-sim-config.json`中`time_line_visualizer_config:output_filename`指定的可视化输出文件在以上网址中打开，即可查看可视化仿真结果。
