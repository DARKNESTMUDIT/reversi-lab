"""Serve the portfolio demo and the local checkpoint on loopback."""
import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from reversi_lab.game import START, VOCAB, State

ROOT = Path(__file__).resolve().parents[1]


def prediction_frame(history, model):
    from reversi_lab.model import predict

    if (
        not isinstance(history, list)
        or not 1 <= len(history) <= 96
        or any(type(token) is not int or not 0 <= token < VOCAB for token in history)
        or history[0] != START
    ):
        raise ValueError("Provide a list of move tokens beginning with START.")

    state = State()
    for token in history[1:]:
        state = state.play(token)
    return {
        "history": history,
        "board": state.board,
        "turn": state.turn,
        "legal": state.legal(),
        "probabilities": None if state.ended else predict(model, history),
        "next": None,
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, model, **kwargs):
        self.model = model
        super().__init__(*args, directory=str(ROOT / "site"), **kwargs)

    def send_json(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_json(404, {"error": "Unknown endpoint."})
            return
        if self.model is None:
            self.send_json(503, {"error": "Start the local server with a trained checkpoint."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= 4096:
                raise ValueError("Expected a JSON request under 4 KB.")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object with a history field.")
            frame = prediction_frame(body.get("history"), self.model)
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
            return
        self.send_json(200, frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import torch

    from reversi_lab.model import load_model

    torch.set_num_threads(2)
    model = load_model(args.checkpoint)
    with ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, model=model)) as server:
        print(f"Reversi Lab at http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
