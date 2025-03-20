import json
import os
import time
from typing import Any, Dict, List, Set
import enum

@enum.unique
class NodeType(enum.Enum):
    BASE_NODE = 0
    CPU_NODE = 1
    GPU_NODE = 2
    COMP_NODE = 3
    COMM_SEND_NODE = 4
    COMM_RECV_NODE = 5
    COMM_COLL_NODE = 6
    REPLAY_NODE = 7

class ToJsonObj:
    def get_json(self) -> Dict[str, Any]:
        var_dict = vars(self)

        # remove all private attr
        remove_keys = []
        for key in var_dict.keys():
            if key.find("__") == 0:
                remove_keys.append(key)
        
        for key in remove_keys:
            var_dict.pop(key)

        return vars(self)

class TraceNodeAttr(ToJsonObj):
    def __init__(self, name: str, value: Any) -> None:
        self.name = name
        self.type = ""
        self.value = value
        if type(value) is str:
            self.type = "string"
        elif type(value) is int:
            self.type = "Uint64"
        else:
            assert False, f"value '{name}' has wrong type '{type(value)}' in trace_node_attr: __init__."
        
class TraceNode(ToJsonObj):
    def __init__(self, name: str) -> None:
        self.id: int = 0
        self.name: str = name
        self.ctrl_deps: List[int] = []
        self.inputs: Dict[str, List[Any]] = {"values": [], "shapes":[], "types":[]}
        self.outputs: Dict[str, List[Any]]  = {"values": [], "shapes":[], "types":[]}
        self.attrs: List[Dict[str, Any]]  = []
        self.__node_type__: Set[NodeType]  = {NodeType.BASE_NODE}

        self.__attrs__: Dict[str, TraceNodeAttr] = {}
        self.add_attr(TraceNodeAttr("op_schema", ""))

    def add_attr(self, attr: TraceNodeAttr) -> None:
        assert attr.name not in self.__attrs__.keys(), f"Duplicated attr in node {self.name}: {attr.name}"
        self.__attrs__[attr.name] = attr
        
    def set_ctrl_deps(self, node: "TraceNode") -> None:
        self.ctrl_deps.append(node.id)
    
    def form_cpu_node(self) -> "TraceNode":
        self.__node_type__.add(NodeType.CPU_NODE)
        return self

    def form_gpu_node(self) -> "TraceNode":
        self.cat = "X"
        self.__node_type__.add(NodeType.GPU_NODE)
        return self
    
    def form_comm_coll_node(self, num_elem: int, elem_bytes: int, comm_group: str) -> "TraceNode":
        # check 1
        if (
            "ncclKernel" not in self.name and 
            "ncclDevKernel" not in self.name and
            "c10d::" not in self.name and 
            "nccl:" not in self.name
        ):
            assert False, f"Illegal name {self.name} for comm-coll node."
            exit(1)
        # check 2
        comm_type_mapping = [
            "allreduce",
            "alltoall",
            "allgather",
            "reducescatter",
            "broadcast",
        ]
        normalized_name = self.name.replace("_", "").replace("-", "").lower()
        found_flag: bool = False
        for key in comm_type_mapping:
            if key in normalized_name:
                found_flag = True
                break
        assert found_flag, f"Illegal name {self.name} for comm-coll node."
        self.__node_type__.add(NodeType.COMM_COLL_NODE)
        self.add_comm_coll_tensor(num_elem, elem_bytes)
        self.add_attr(TraceNodeAttr("comm_group", comm_group))
        return self
    
    def add_comm_coll_tensor(self, num_elem: int, elem_bytes: int) -> "TraceNode":
        assert NodeType.COMM_COLL_NODE in self.__node_type__, \
               f"node {self.name} is not a comm_coll_node, please use func form_comm_coll_node first."
        self.inputs["values"].append([0, 0, 0, num_elem, elem_bytes])
        self.inputs["types"].append(["Tensor"])
        return self
    
    def form_comm_send_node(self, comm_src: int, comm_dst: int, comm_tag: int, num_elem: int, elem_bytes: int) -> "TraceNode":
        if "send" not in self.name.lower():
            assert False, f"Illegal name {self.name} for send node."
            exit(1)
        self.inputs["values"].append([0, 0, 0, num_elem, elem_bytes])
        self.inputs["types"].append(["Tensor"])
        self.add_attr(TraceNodeAttr("comm_src", comm_src))
        self.add_attr(TraceNodeAttr("comm_dst", comm_dst))
        self.add_attr(TraceNodeAttr("comm_tag", comm_tag))
        self.__node_type__.add(NodeType.COMM_SEND_NODE)
        return self
    
    def form_comm_recv_node(self, comm_src: int, comm_dst: int, comm_tag: int, num_elem: int, elem_bytes: int) -> "TraceNode":
        if "recv" not in self.name.lower():
            assert False, f"Illegal name {self.name} for recv node."
            exit(1)
        self.inputs["values"].append([0, 0, 0, num_elem, elem_bytes])
        self.inputs["types"].append(["Tensor"])
        self.add_attr(TraceNodeAttr("comm_src", comm_src))
        self.add_attr(TraceNodeAttr("comm_dst", comm_dst))
        self.add_attr(TraceNodeAttr("comm_tag", comm_tag))
        self.__node_type__.add(NodeType.COMM_RECV_NODE)
        return self

    def form_comp_node(self, num_ops: int, tensor_size: int) -> "TraceNode":
        self.add_attr(TraceNodeAttr("num_ops", num_ops))
        self.add_attr(TraceNodeAttr("tensor_size", tensor_size))
        self.__node_type__.add(NodeType.COMP_NODE)
        return self

    def form_replay_node(self, ns: int) -> "TraceNode":
        self.add_attr(TraceNodeAttr("dur", ns))
        self.__node_type__.add(NodeType.REPLAY_NODE)
        return self

    def get_json(self, is_simplified=False) -> Dict[str, Any]:
        if is_simplified:
            if NodeType.REPLAY_NODE in self.__node_type__ :
                self.simplified_attrs: List[Any] = [self.__attrs__["dur"].value]
            elif NodeType.COMP_NODE in self.__node_type__ :
                num_ops = self.__attrs__["num_ops"].value
                tensor_size = self.__attrs__["tensor_size"].value
                self.simplified_attrs: List[Any] = [num_ops, tensor_size]
            elif NodeType.COMM_COLL_NODE in self.__node_type__:
                num_elem = self.inputs["values"][0][3]
                elem_bytes = self.inputs["values"][0][4]
                comm_group = self.__attrs__["comm_group"].value
                self.simplified_attrs: List[Any] = [num_elem, elem_bytes, comm_group]
            elif NodeType.COMM_SEND_NODE in self.__node_type__ or \
                 NodeType.COMM_RECV_NODE in self.__node_type__:
                num_elem = self.inputs["values"][0][3]
                elem_bytes = self.inputs["values"][0][4]
                comm_src = self.__attrs__["comm_src"].value
                comm_dst = self.__attrs__["comm_dst"].value
                comm_tag = self.__attrs__["comm_tag"].value
                self.simplified_attrs: List[Any] = [num_elem, elem_bytes, comm_src, comm_dst, comm_tag]
            else:
                assert False, f"NodeType not found for node {self.name}"
            del self.inputs
            del self.outputs
            del self.attrs
            return super().get_json()
        else:
            self.attrs = [attr.get_json() for attr in self.__attrs__.values()]
            return super().get_json()

class PytorchPlusTrace(ToJsonObj):
    def __init__(self) -> None:
        self.schema: str = "1.0.2-chakra.0.0.4"
        self.pid: int = 1
        self.time: int = time.strftime("%Y-%m-%d %X", time.localtime())
        self.start_ts: int = int(1E9)
        self.nodes: List[Dict[str, Any]] = {}
        self.finish_ts: int = int(2E9)

        self.__nodes__: List[TraceNode] = []
        self.__nodes_map__: Dict[str, int] = {}
        self.__unique_id_counter__: int = 0
        self.__sendrecv_list__: List[List[int]] = []
    
    def __remove_dangling_nodes__(self) -> None:
        """
        Remove any dangling nodes from the self.__nodes__ dictionary.

        A node is considered dangling if it has no parents and no children.
        """
        parent_ids = set()
        for node in self.__nodes__:
            parent_ids.update(node.ctrl_deps)

        dangling_nodes = [
            node_id for node_id in range(len(self.__nodes__)) \
                if node_id not in parent_ids and not self.__nodes__[node_id].ctrl_deps
        ]
        for node_id in dangling_nodes:
            del self.__nodes__[node_id]

        if dangling_nodes:
            print(f"Identified and removed {len(dangling_nodes)} dangling nodes:")
            for node_id in dangling_nodes:
                print(f" - Node ID {node_id}")

    def __identify_cyclic_dependencies__(self) -> None:
        """
        Identify if there are any cyclic dependencies among Chakra nodes.

        This method checks for cycles in the graph of Chakra nodes using a depth-first search (DFS) algorithm. It logs
        an error message and raises an exception if a cycle is detected, ensuring the graph is a Directed Acyclic Graph
        (DAG).

        Raises
            Exception: If a cyclic dependency is detected among the Chakra nodes.
        """
        visited = set()
        stack = set()

        def dfs(node_id: int, path: List[int]) -> bool:
            """
            Depth-first search to detect cycles.

            Args:
                node_id (int): The node ID to start the DFS from.
                path (List[int]): The path traversed so far, for tracing the cycle.

            Returns:
                bool: True if a cycle is detected, False otherwise.
            """
            if node_id in stack:
                cycle_nodes = " -> ".join([self.__nodes__[n].name for n in path + [node_id]])
                print(f"Cyclic dependency detected: {cycle_nodes}")
                return True
            if node_id in visited:
                return False

            visited.add(node_id)
            stack.add(node_id)
            path.append(node_id)
            for child_id in self.__nodes__[node_id].ctrl_deps:
                if dfs(child_id, path.copy()):
                    return True
            stack.remove(node_id)
            path.pop()
            return False

        for node_id in range(len(self.__nodes__)):
            if dfs(node_id, []):
                raise Exception(f"Cyclic dependency detected starting from node {self.__nodes__[node_id].name}")

    def add_node(self, node: TraceNode) -> None:
        # send recv check
        if(NodeType.COMM_SEND_NODE in node.__node_type__ or 
           NodeType.COMM_RECV_NODE in node.__node_type__):
            check_sendecv: List[int] = [node.__attrs__["comm_src"].value, node.__attrs__["comm_dst"].value, node.__attrs__["comm_tag"].value]
            # for send node, check[3] = 0. for recv node, check[3] = 1
            check_sendecv.append(0 if NodeType.COMM_SEND_NODE in node.__node_type__ else 1)
            assert check_sendecv not in self.__sendrecv_list__, \
                   f"Duplicated sendrecv found. src: {check_sendecv[0]}, " \
                                             +f"dst: {check_sendecv[1]}, " \
                                             +f"tag: {check_sendecv[2]}, " \
                                             +("send node." if check_sendecv[3]==0 else "recv node.")
            self.__sendrecv_list__.append(check_sendecv)
        node.id = self.__unique_id_counter__
        self.__unique_id_counter__ += 1
        self.__nodes__.append(node)
        assert node.name not in self.__nodes_map__.keys(), f"Duplicated node found. {node.name}"
        self.__nodes_map__[node.name] = len(self.__nodes__) - 1
    
    def add_nodes(self, node_list: List[TraceNode]) -> None:
        for node in node_list:
            self.add_node(node)
    
    def add_node_with_ctrl_deps(self, node: TraceNode, ctrl_deps: List[str] = []) -> None:
        if len(ctrl_deps) != 0:
            for parent_node_name in ctrl_deps:
                assert parent_node_name in self.__nodes_map__.keys(), \
                    "parent node is not found. f{parent_node_name}"
                node.set_ctrl_deps(self.__nodes__[self.__nodes_map__[parent_node_name]])


            # for parent_node in self.__nodes__:
            #     if parent_node.name in ctrl_deps:
            #         node.set_ctrl_deps(parent_node)
        elif len(self.__nodes__) > 0:
            node.set_ctrl_deps(self.__nodes__[-1])
        self.add_node(node)
    
    def get_last_node_name(self) -> str:
        return self.__nodes__[-1].name

    def contain_node(self, node_name: str) -> bool:
        return node_name in self.__nodes_map__.keys()

    def trace_legel_check(self) -> None:
        self.__remove_dangling_nodes__()
        self.__identify_cyclic_dependencies__()

    def get_json(self) -> Dict[str, Any]:
        self.nodes = [node.get_json(is_simplified=False) for node in self.__nodes__]
        return super().get_json()
    
    def get_simplified_trace(self) -> str:
        self.trace_legel_check()
        self.nodes = [node.get_json(is_simplified=True) for node in self.__nodes__]
        simplified_trace: str = str(len(self.nodes))+"\n"
        for node in self.nodes:
            simplified_node: str = str(node["id"]) + "\n" + node["name"] + "\n"
            if len(node["ctrl_deps"]) == 0:
                simplified_node = simplified_node + "-1\n"
            else:
                simplified_node = \
                    simplified_node + \
                    str(len(node["ctrl_deps"])) + " " + \
                    " ".join(str(i) for i in node["ctrl_deps"]) + "\n"
            simplified_node = simplified_node + " ".join(str(i) for i in node["simplified_attrs"]) + "\n"
            simplified_trace = simplified_trace + simplified_node
        return simplified_trace

def get_example(comp1: int, ar1_name: str, ar1: int, comp2: int, ar2_name: str, ar2: int) -> PytorchPlusTrace:
    trace = PytorchPlusTrace()

    root_node = TraceNode("[pytorch|profiler|execution_graph|thread]").form_cpu_node().form_comp_node(1024, 1024)
    trace.add_node_with_ctrl_deps(root_node)

    comp1_node = TraceNode("comp1").form_cpu_node().form_comp_node(65536, comp1*128)
    trace.add_node_with_ctrl_deps(comp1_node)

    ar1_node = TraceNode(ar1_name).form_comm_coll_node(ar1, 1, "dp_group")
    trace.add_node_with_ctrl_deps(ar1_node)

    comp2_node = TraceNode("comp2").form_cpu_node().form_comp_node(65536, comp2*128)
    trace.add_node_with_ctrl_deps(comp2_node)

    ar2_node = TraceNode(ar2_name).form_comm_coll_node(ar2, 1, "tp_group")
    trace.add_node_with_ctrl_deps(ar2_node)

    comp3_node = TraceNode("comp3").form_cpu_node().form_comp_node(1024, 1024)
    trace.add_node_with_ctrl_deps(comp3_node)

    return trace

def get_overlap_example() -> PytorchPlusTrace:
    trace = PytorchPlusTrace()
    root_node = TraceNode("[pytorch|profiler|execution_graph|thread]").form_cpu_node().form_comp_node(1024, 1024)
    trace.add_node_with_ctrl_deps(root_node)

    ar1_node = TraceNode("nccl:allreduce1").form_comm_coll_node(4000, 1, "dp_group")
    trace.add_node_with_ctrl_deps(ar1_node, ["[pytorch|profiler|execution_graph|thread]"])

    comp1_node = TraceNode("comp1").form_cpu_node().form_comp_node(65536, 500*128)
    trace.add_node_with_ctrl_deps(comp1_node,["[pytorch|profiler|execution_graph|thread]"])

    return trace

def get_send_recv_example(is_send: bool = True) -> PytorchPlusTrace:
    trace = PytorchPlusTrace()
    root_node = TraceNode("[pytorch|profiler|execution_graph|thread]").form_cpu_node().form_comp_node(1024, 1024)
    trace.add_node_with_ctrl_deps(root_node)

    if is_send:
        send_node = TraceNode("test_send").form_comm_send_node(0, 1, 0, 65536, 8)
        trace.add_node_with_ctrl_deps(send_node)
    else:
        recv_node = TraceNode("test_recv").form_comm_recv_node(0, 1, 0, 65536, 8)
        trace.add_node_with_ctrl_deps(recv_node)

    return trace

def example():
    path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(path + "/demo/trace"):
        os.mkdir(path + "/demo/trace")

    trace = get_example(500, "nccl:allreduce1", 4000, 10000, "nccl:allreduce2", 6000)
    with open(path + "/demo/trace/npu0.json", "w") as f:
        f.write(json.dumps(trace.get_json(), indent=4))

    trace = get_example(10000, "nccl:allreduce1", 4000, 15, "nccl:allreduce4", 400000)
    with open(path + "/demo/trace/npu1.json", "w") as f:
        f.write(json.dumps(trace.get_json(), indent=4))

    trace = get_example(250, "nccl:allreduce3", 1000, 500, "nccl:allreduce2", 6000)
    with open(path + "/demo/trace/npu2.json", "w") as f:
        f.write(json.dumps(trace.get_json(), indent=4))

    trace = get_example(130, "nccl:allreduce3", 1000, 15, "nccl:allreduce4", 400000)
    with open(path + "/demo/trace/npu3.json", "w") as f:
        f.write(json.dumps(trace.get_json(), indent=4))

def overlap_example():
    path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(path + "/demo/trace"):
        os.mkdir(path + "/demo/trace")
    trace = get_overlap_example()
    with open(path + "/demo/trace/overlap_example0.json", "w") as f:
        f.write(json.dumps(trace.get_json(), indent=4))
    with open(path + "/demo/trace/overlap_example1.json", "w") as f:
        f.write(json.dumps(trace.get_json(), indent=4))

def send_recv_example():
    path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(path + "/demo/trace"):
        os.mkdir(path + "/demo/trace")
    trace = get_send_recv_example(True)
    with open(path + "/demo/trace/send_recv_example0.json", "w") as f:
        f.write(json.dumps(trace.get_json(), indent=4))
    trace = get_send_recv_example(False)
    with open(path + "/demo/trace/send_recv_example1.json", "w") as f:
        f.write(json.dumps(trace.get_json(), indent=4))

# send_recv_example()