from __future__ import annotations

import argparse
import sys

from .inference_deployment_common import ensure_undeployed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Undeploy Modal inference app for one experiment")
    parser.add_argument("model_cli_name", type=str)
    parser.add_argument("model_api_name", type=str)
    parser.add_argument("config_nickname", type=str)
    parser.add_argument("epoch", type=int)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    app_name, return_code = ensure_undeployed(
        args.model_cli_name,
        args.model_api_name,
        args.config_nickname,
        args.epoch,
    )
    if return_code == 0:
        print(f"Modal inference app already absent or stopped, or stopped now: {app_name}", flush=True)
    else:
        print(f"Failed to stop Modal inference app: {app_name}", file=sys.stderr, flush=True)
    print(f"DEPLOYMENT_NAME={app_name}", flush=True)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
