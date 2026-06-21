#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

from __future__ import annotations

import argparse
import argcomplete
import typing

if typing.TYPE_CHECKING:
    import torch
    from src.config import Config

def setup_environment(config: Config, verbose: bool) -> torch.device:
    """ Configures logging level, device assignment, and random seed.
    """
    import torch
    from src.logger import logger

    if verbose:
        logger.setLevel("DEBUG")
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.debug(f"Using device: {device}")

    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)

    return device


def main() -> None:
    parser = argparse.ArgumentParser(description="Fermi-LAT electron/proton classifier.")
    parser.add_argument("--merit", action="store_true", help="Use the merit variables model.")
    parser.add_argument("--single-branch", action="store_true", help="Use the single-branch CNN model.")
    parser.add_argument("--train", action="store_true", help="Run the training loop on a newly instantiated model.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose option (set logger to debug mode).")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    from src.config import Config
    from src.model import FermiMultiBranchCNN, FermiSingleBranchCNN, FermiMeritVarsNN
    from src.pipelines import ImagingPipeline, MeritPipeline

    config = Config()
    device = setup_environment(config, args.verbose)

    if args.single_branch:
        model = FermiSingleBranchCNN()
        model_save_path = config.single_branch_model_save_path
    else:
        model = FermiMultiBranchCNN()
        model_save_path = config.model_save_path

    imaging_pipeline = ImagingPipeline(model=model, config=config, device=device, train=args.train, save_path=model_save_path)
    imaging_pipeline.run()

    if args.merit:
        merit_model = FermiMeritVarsNN()
        merit_pipeline = MeritPipeline(model=merit_model, config=config, device=device, train=args.train)
        merit_pipeline.run()


if __name__ == '__main__':
    main()