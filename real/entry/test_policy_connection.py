"""
Smoke-test the websocket policy server: connect, send dummy observations, print outputs.

Start the policy (same as ``inference.py``), for example:

    uv run training/serve_policy.py policy:checkpoint \\
        --policy.config=xarm_scoop \\
        --policy.dir=/path/to/checkpoint \\
        --default-prompt='scoop up the rocks'

Then run (no robot or cameras):

    uv run real/entry/test_policy_connection.py --host <server_ip> --port 8000

Dummy observation keys match the xarm policy stack (``XarmInputs`` / ``collect_demo.py``):
``observation/exterior_image_1_left``, ``observation/wrist_image_left``, ``observation/state``,
and ``prompt``. Images use the same H×W as the recorded demos (240×320); the server resizes to 224.
"""

import dataclasses
import logging
import time

import numpy as np
import tyro
from openpi_client import websocket_client_policy


# Same defaults as ``real/entry/collect_demo.py`` / ``inference.py`` task string.
TASK_DESCRIPTION = "scoop up the rocks"

# RealSense color dimensions used when collecting xarm demos.
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320
STATE_DIM = 6


@dataclasses.dataclass
class Args:
    """Websocket policy server (``training/serve_policy.py``)."""

    host: str = "localhost"
    port: int = 8000
    api_key: str | None = None
    prompt: str = TASK_DESCRIPTION
    num_infers: int = 3
    seed: int = 0


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    rng = np.random.default_rng(seed=args.seed)

    def make_observation() -> dict:
        exterior = rng.integers(0, 256, size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        wrist = rng.integers(0, 256, size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        state = rng.standard_normal(STATE_DIM, dtype=np.float32)
        return {
            "observation/exterior_image_1_left": exterior,
            "observation/wrist_image_left": wrist,
            "observation/state": state,
            "prompt": args.prompt,
        }

    policy = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    meta = policy.get_server_metadata()
    logging.info("Connected; server metadata: %s", meta)

    for i in range(args.num_infers):
        obs = make_observation()
        t0 = time.perf_counter()
        out = policy.infer(obs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        actions = np.asarray(out["actions"])
        logging.info(
            "infer %d/%d: actions shape=%s dtype=%s client_roundtrip_ms=%.1f",
            i + 1,
            args.num_infers,
            actions.shape,
            actions.dtype,
            elapsed_ms,
        )
        if "policy_timing" in out:
            logging.info("policy_timing: %s", out["policy_timing"])
        if "server_timing" in out:
            logging.info("server_timing: %s", out["server_timing"])

    logging.info("OK — websocket inference completed successfully.")


if __name__ == "__main__":
    main(tyro.cli(Args))
