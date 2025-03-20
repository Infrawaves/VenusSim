import argparse
import sys
import traceback
import json

from chakra_convert_standalone.converter.pytorch_converter import PyTorchConverter
from chakra_convert_standalone.converter.text_converter import TextConverter
from chakra_convert_standalone.converter.converter import get_logger

def main() -> None:
    parser = argparse.ArgumentParser(description="Execution Trace Converter")
    parser.add_argument("--input_type", type=str, default=None, required=True, help="Input execution trace type")
    parser.add_argument(
        "--input_filename", type=str, default=None, required=True, help="Input execution trace filename"
    )
    parser.add_argument(
        "--output_filename", type=str, default=None, required=True, help="Output Chakra execution trace filename"
    )
    parser.add_argument(
        "--num_npus", type=int, default=None, required="Text" in sys.argv, help="Number of NPUs in a system"
    )
    parser.add_argument(
        "--num_passes", type=int, default=None, required="Text" in sys.argv, help="Number of training passes"
    )
    parser.add_argument("--log_filename", type=str, default="debug.log", help="Log filename")
    args = parser.parse_args()

    logger = get_logger(args.log_filename)
    logger.debug(" ".join(sys.argv))

    try:
        if args.input_type == "Text":
            converter = TextConverter(args.input_filename, args.output_filename, args.num_npus, args.num_passes, logger)
            converter.convert()
        elif args.input_type == "PyTorch":
            converter = PyTorchConverter(args.input_filename, args.output_filename, logger)
            converter.convert()
        else:
            supported_types = ["Text", "PyTorch"]
            logger.error(
                f"The input type '{args.input_type}' is not supported. "
                f"Supported types are: {', '.join(supported_types)}."
            )
            sys.exit(1)
    except Exception:
        traceback.print_exc()
        logger.debug(traceback.format_exc())
        sys.exit(1)

# def main() -> None:
#     parser = argparse.ArgumentParser(description="Execution Trace Generator")
#     parser.add_argument("--config_file", type=str, default=None, required=True, help="configuration file")
#     args = parser.parse_args()
#     args = json.load(open(args.config_file, 'r'))

#     logger = get_logger(args.log_filename)
#     logger.debug(" ".join(sys.argv))
#     for i in range(json_output_files):
#         json_output_file = json_output_files[i]
#         try:
#             et_output_file = args.et_output_filename + f".{i}.et"
#             converter = PyTorchConverter(json_output_file, et_output_file, logger)
#             converter.convert()
#         except Exception:
#             traceback.print_exc()
#             logger.debug(traceback.format_exc())
#             sys.exit(1)

if __name__ == "__main__":
    main()