import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

from dpnegf.entrypoints.run import run
from dpnegf.utils.loggers import set_log_handles

from dpnegf import __version__



def get_ll(log_level: str) -> int:
    """Convert string to python logging level.

    Parameters
    ----------
    log_level : str
        allowed input values are: DEBUG, INFO, WARNING, ERROR, 3, 2, 1, 0

    Returns
    -------
    int
        one of python logging module log levels - 10, 20, 30 or 40
    """
    if log_level.isdigit():
        int_level = (4 - int(log_level)) * 10
    else:
        int_level = getattr(logging, log_level)

    return int_level

def main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DPNEGF: A NEGF Python package compatible to DeePTB method for efficient quantum transport simulations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('-v', '--version', 
                        action='version', version=f'%(prog)s {__version__}', help="show the DPNEGF's version number and exit")


    subparsers = parser.add_subparsers(title="Valid subcommands", dest="command")

    # log parser
    parser_log = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_log.add_argument(
        "-ll",
        "--log-level",
        choices=["DEBUG", "3", "INFO", "2", "WARNING", "1", "ERROR", "0"],
        default="INFO",
        help="set verbosity level by string or number, 0=ERROR, 1=WARNING, 2=INFO "
             "and 3=DEBUG",
    )

    parser_log.add_argument(
        "-lp",
        "--log-path",
        type=str,
        default=None,
        help="set log file to log messages to disk, if not specified, the logs will "
             "only be output to console",
    )


    # run parser
    parser_run = subparsers.add_parser(
        "run",
        parents=[parser_log],
        help="run the TB with a model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser_run.add_argument(
        "INPUT", help="the input parameter file for postprocess run in json format",
        type=str,
        default=None
    )

    parser_run.add_argument(
        "-i",
        "--init-model",
        type=str,
        default=None,
        help="Initialize the model by the provided checkpoint.",
    )

    parser_run.add_argument(
        "-stu",
        "--structure",
        type=str,
        default=None,
        help="the structure file name wiht its suffix of format, such as, .vasp, .cif etc., prior to the model_ckpt tags in the input json. "
    )

    parser_run.add_argument(
        "-o",
        "--output",
        type=str,
        default="./",
        help="The output files in postprocess run."
    )

    return parser

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse arguments and convert argument strings to objects.

    Parameters
    ----------
    args: List[str]
        list of command line arguments, main purpose is testing default option None
        takes arguments from sys.argv

    Returns
    -------
    argparse.Namespace
        the populated namespace
    """
    parser = main_parser()
    parsed_args = parser.parse_args(args=args)
    if parsed_args.command is None:
        parser.print_help()
    else:
        parsed_args.log_level = get_ll(parsed_args.log_level)

    return parsed_args

def main():
    args = parse_args()

    if args.command not in (None, "run"):
        set_log_handles(args.log_level, Path(args.log_path) if args.log_path else None)

    dict_args = vars(args)
    

    if args.command == 'run':
        run(**dict_args)


