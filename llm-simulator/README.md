# Dependencies Setup
## Install required system packages:
```bash
$ sudo apt update
$ sudo apt install \
    gcc g++ make cmake \
    libboost-dev libboost-program-options-dev \
    libprotobuf-dev protobuf-compiler \
    python3 python3-pip git
```

## .pip3
You can install required Python packages natively using pip3.

```bash
$ pip3 install --upgrade pip
$ pip3 install protobuf==3.6.1 pydot
```

# Build llm-simulator
## Clone Repository
```bash
$ git clone --recurse-submodules git@183.207.7.174:moon/llm-simulator/llm-simulator.git
$ cd llm-simulator
# set all detached head in submodules into branch recording in .gitmodules
$ git submodule foreach -q --recursive 'git checkout $(git config -f $toplevel/.gitmodules submodule.$name.branch)'
```

## Compile Program
```bash
# For Analytical Network Backend, -l for cleanup, -d for debug
$ ./build/astra_analytical/build.sh

# For NS3 Network Backend
$ ./build/astra_ns3/build.sh -c
```

## Update and pull submodule
### Update submodule
```bash
$ cd path/to/submodule
$ git add ....
$ git commit ...
$ git push ...
$ cd path/to/toplevel-module
$ git commit submodule commit id changes
$ git push ...
```

### Pull changes in submodule
```bash
$ cd path/to/toplevel-module
# pull all submodule new update
$ git submodule update --remote
# pull submodule commit id changes
$ git pull origin main
```
# [Wiki](https://astra-sim.github.io/astra-sim-docs/index.html)
