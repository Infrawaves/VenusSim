# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from __future__ import annotations

import json

from typing import List, Tuple

from param_bench.train.comms.pt import comms_utils
from param_bench.train.comms.pt.comms_utils import commsArgs
from param_bench.train.comms.pt.pytorch_backend_utils import supportedP2pOps

from param_bench.train.compute.python.tools.execution_trace import ExecutionTrace

tensorDtypeMap = {
    "Tensor(int)": "int",
    "Tensor(float)": "float",
    "Tensor(bool)": "bool",
    "Tensor(long)": "long",
    "Tensor(long int)": "long",
    "Tensor(double)": "double",
    "Tensor(half)": "half",
    "Tensor(byte)": "byte",
    "Tensor(c10::Half)": "half",
    "Tensor(c10::BFloat16)": "bfloat16",
    "Tensor(unsigned char)": "char",
}


def parseTrace(
    in_trace: List, trace_type: str, target_rank: int, total_ranks: int
) -> List:
    """
    Parse trace files to be compatible with PARAM replay-mode.
    Currently supports: Basic Trace, Kineto Unitrace, and PyTorch ET trace.

    Args:
        in_trace: Trace file to be parsed.
        trace_type: Trace type to be parsed with
        target_rank: The current rank of the device.
        total_ranks: Total number of ranks.
    Returns:
        parsed_trace: Parsed trace that is compatible with PARAM replay-mode.
    """

    if trace_type == "basic":  # Basic Trace
        parsed_trace = _parseBasicTrace(in_trace)
    elif trace_type == "et":  # Execution Trace (e.g. PyTorch ET, Chakra)
        parsed_trace = _parseExecutionTrace(ExecutionTrace(in_trace), total_ranks)
    elif trace_type == "kineto":  # Kineto Unitrace
        parsed_trace = _parseKinetoUnitrace(in_trace, target_rank)
    else:
        raise ValueError("Unrecognized trace format.")

    return parsed_trace


def _parseBasicTrace(in_trace: List):
    """
    Convert Basic Trace to comms trace format.
    """
    newCommsTrace = []
    for cnt, curComm in enumerate(in_trace):

        newComm = commsArgs()
        newComm.id = cnt
        newComm.markerStack = curComm.get("markers")
        if "comms" in curComm:
            _parseBasicTraceComms(curComm, newComm)

        elif "compute" in curComm:
            _parseBasicTraceCompute(curComm, newComm)

        if newComm.comms is not None or newComm.compute is not None:
            newCommsTrace.append(newComm)
        else:
            raise ValueError(
                "Trace file contains an element that is not a supported in PARAM! Please format all elements as comms or compute for replay."
            )

    return newCommsTrace


def _parseBasicTraceComms(curComm, newComm: commsArgs) -> None:

    newComm.comms = comms_utils.paramToCommName(curComm["comms"].lower())
    if newComm.markerStack is None:
        newComm.markerStack = [newComm.comms]
    newComm.req = curComm.get("req")
    newComm.startTimeNs = curComm.get("startTime_ns")
    newComm.worldSize = curComm.get("world_size")
    newComm.root = curComm.get("root")
    newComm.pgId = curComm.get("pg_id")
    newComm.groupRanks = curComm.get("global_ranks")

    if newComm.comms not in ("wait", "barrier", "init", "batch_isend_irecv"):
        newComm.inMsgSize = curComm["in_msg_size"]
        newComm.outMsgSize = curComm["out_msg_size"]
        newComm.dtype = curComm["dtype"].lower()

    if newComm.comms == "all_to_allv":
        newComm.inSplit = curComm["in_split"]
        newComm.outSplit = curComm["out_split"]

    if newComm.comms in supportedP2pOps:
        newComm.src_rank = curComm["src_rank"]
        newComm.dst_rank = curComm["dst_rank"]
        newComm.batch_p2p = curComm["use_batch"]


def _parseBasicTraceCompute(curComm, newComm: commsArgs) -> None:
    newComm.compute = curComm["compute"].lower()
    if newComm.markerStack is None:
        newComm.markerStack = [newComm.compute]
    # count = number of times to call the compute kernel
    if "count" in curComm:
        newComm.count = curComm["count"]
    # if no count is specified, assume 1
    else:
        newComm.count = 1
    if newComm.compute == "gemm":
        if "mm_dim" in curComm:
            newComm.mm0_dim0 = curComm.get("mm_dim")
            newComm.mm0_dim1 = curComm.get("mm_dim")
            newComm.mm1_dim0 = curComm.get("mm_dim")
            newComm.mm1_dim1 = curComm.get("mm_dim")
        else:
            newComm.mm0_dim0 = curComm.get("mm0_dim0")
            newComm.mm0_dim1 = curComm.get("mm0_dim1")
            newComm.mm1_dim0 = curComm.get("mm1_dim0")
            newComm.mm1_dim1 = curComm.get("mm1_dim1")
        newComm.dtype = curComm.get("dtype").lower()
    elif newComm.compute == "emb_lookup":
        if "direction" in curComm:
            newComm.direction = curComm["direction"]
        else:
            newComm.direction = "forward"
        newComm.emb_dim = curComm.get("emb_dim")
        newComm.num_embs = curComm.get("num_embs")
        newComm.batch_size = curComm.get("batch_size")
        newComm.num_emb_tables_per_device = curComm.get("num_emb_tables")
        newComm.num_emb_tables_batched = -1
        newComm.bag_size = curComm.get("bag_size")
    else:
        raise ValueError(
            f"Trace file contains {str(newComm.compute)} compute element that is not supported in PARAM!"
        )


def _parseKinetoUnitrace(in_trace: List, target_rank: int) -> List:
    """
    Convert the Kineto unitrace w/ comms metadata to the clean common trace format for replay.
    """
    newCommsTrace = []
    commsCnt = 0
    for entry in in_trace:
        # TODO: figure the current marker stack if present
        marker = "unknown"
        pass

        if (
            "name" in entry
            and entry["name"] == "record_param_comms"
            and entry["args"]["rank"] == target_rank
        ):

            newComm = commsArgs()
            newComm.comms = comms_utils.paramToCommName(entry["args"]["comms"].lower())
            newComm.id = commsCnt
            newComm.inMsgSize = entry["args"]["in_msg_size"]
            newComm.outMsgSize = entry["args"]["out_msg_size"]
            newComm.dtype = entry["args"]["dtype"].lower()
            newComm.inSplit = entry["args"]["in_split"]
            newComm.outSplit = entry["args"]["out_split"]
            newComm.markerStack = marker

            newCommsTrace.append(newComm)
            commsCnt += 1

    return newCommsTrace


def _getTensorInfoFromPyTorchETEntry(
    tensor_container: List, container_type: str
) -> Tuple[int, int, str]:
    """
    Extract message size, tensor count, type from PyTorch ET entry inputs/outputs field.
    NOTE: This format can be changed at anytime. TODO: When an extract/parsing tool is available in ATC, switch to it.
    """
    list_count = container_type.count("GenericList")
    tensors = []
    if list_count == 2:
        # GenericList[GenericList[Tensor(), Tensor()]]
        tensors = tensor_container[0][0]
        dtype = container_type.replace("GenericList[", "").split(",", 1)[0]
    elif list_count == 1:
        # GenericList[Tensor()]
        tensors = tensor_container[0]
        dtype = container_type.replace("GenericList[", "").replace("]", "")
    else:
        tensors.append(tensor_container[0])
        dtype = container_type

    msg_size = 0
    for tensor in tensors:
        msg_size += tensor[3]

    return msg_size, dtype


def _parseExecutionTrace(in_trace: ExecutionTrace, total_ranks: int) -> List:
    """
    Convert the Execution Trace comms metadata to the common trace format for replay.

    """

    initOps = []
    newCommsTrace = []
    backendIdToGlobalRanks = {}
    backendIdToPgid = {}
    commsPerbackendId = {}

    # Parse PG info from ET
    for node in in_trace.nodes.values():
        if "process_group:init" in node.name:
            pgJson = node.inputs[0]
            pgObj = json.loads(pgJson)
            for pg in pgObj:
                pgId = pg["pg_name"] if "pg_name" in pg else pg["pg_id"]
                backendId = pg["backend_id"]
                ranks = pg["ranks"]
                backendIdToGlobalRanks[backendId] = [int(rank) for rank in ranks.keys()]
                backendIdToPgid[backendId] = pgId
                commsPerbackendId[backendId] = 0
            break  # only one process_group init node per trace

    # Parse comms nodes
    for node in in_trace.nodes.values():
        if node.name == "record_param_comms":
            shift = (
                0 if len(node.inputs) == 8 else 1
            )  # wait/barrier ops do not have an input tensor (len=7), shift index one over
            newComm = commsArgs()
            newComm.id = node.id
            newComm.comms = comms_utils.paramToCommName(
                node.inputs[4 - shift].lower()
            )  # 5th value of inputs is colName
            if newComm.comms == "init":
                continue
            newComm.req = node.inputs[
                1 - shift
            ]  # 2nd value of inputs is the req id of the collective

            backendId = node.inputs[
                2 - shift
            ]  # 3rd value of inputs is the backend id of the collective
            if backendId in backendIdToGlobalRanks:
                # Assign pg_id info for PGs that were created.
                newComm.pgId = backendIdToPgid[backendId]
                newComm.groupRanks = backendIdToGlobalRanks[backendId]
                newComm.worldSize = len(newComm.groupRanks)
                commsPerbackendId[backendId] += 1

            if newComm.comms not in ("wait", "barrier"):
                (
                    newComm.inMsgSize,
                    inMsgType,
                ) = _getTensorInfoFromPyTorchETEntry(node.inputs, node.input_types[0])
                (
                    newComm.outMsgSize,
                    _,
                ) = _getTensorInfoFromPyTorchETEntry(node.outputs, node.output_types[0])
                newComm.dtype = tensorDtypeMap[
                    inMsgType
                ]  # 1st value of input_types is the data type for the tensors

            if newComm.comms == "all_to_allv":
                # 6th value of inputs is in_split, split evenly if not provided
                if not newComm.worldSize:
                    # if no pg info provided, use total ranks as world size
                    newComm.worldSize = total_ranks
                newComm.inSplit = (
                    node.inputs[5]
                    if node.inputs[5]
                    else [int(newComm.inMsgSize / newComm.worldSize)]
                    * newComm.worldSize
                )
                # 7th value of inputs is out_split, split evenly if not provided
                newComm.outSplit = (
                    node.inputs[6]
                    if node.inputs[6]
                    else [int(newComm.outMsgSize / newComm.worldSize)]
                    * newComm.worldSize
                )
            newCommsTrace.append(newComm)
    newCommsTrace.sort(key=lambda x: x.req)

    # Build init node
    initOps = []
    for backend_id, global_ranks in backendIdToGlobalRanks.items():
        if commsPerbackendId[backend_id] == 0:
            continue
        newComm = commsArgs()
        newComm.comms = "init"
        newComm.pgId = backendIdToPgid[backend_id]
        newComm.req = -1
        newComm.groupRanks = global_ranks
        newComm.worldSize = len(global_ranks)
        initOps.append(newComm)

    return initOps + newCommsTrace
