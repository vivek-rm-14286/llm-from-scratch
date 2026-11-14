"""A fake OpenAI-compatible streaming server for testing clients and load-test scripts without a GPU.

    python fake_server.py --port 8001 --ttft 0.2 --tpot 0.02 --slots 4

It serves POST /v1/chat/completions (stream or not) and returns filler words. Each request waits
--ttft seconds, then emits one "token" every --tpot seconds. Only --slots requests are served at a
time; the rest queue, which is how a real server at its batch limit behaves. Standard library only.
"""
import argparse, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=8001)
p.add_argument("--ttft", type=float, default=0.2)
p.add_argument("--tpot", type=float, default=0.02)
p.add_argument("--slots", type=int, default=4)
args = p.parse_args()
slots = threading.Semaphore(args.slots)
WORDS = "the river runs down from the hills to the sea and".split()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n = int(body.get("max_tokens", 32))
        base = {"id": "fake-1", "model": body.get("model", "fake"), "created": int(time.time())}
        usage = {"prompt_tokens": 8, "completion_tokens": n, "total_tokens": 8 + n}
        with slots:
            time.sleep(args.ttft)
            if not body.get("stream"):
                time.sleep(args.tpot * (n - 1))
                text = " ".join(WORDS[i % len(WORDS)] for i in range(n))
                out = json.dumps({**base, "object": "chat.completion", "usage": usage, "choices": [
                    {"index": 0, "finish_reason": "length",
                     "message": {"role": "assistant", "content": text}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for i in range(n):
                if i:
                    time.sleep(args.tpot)
                self.send({**base, "object": "chat.completion.chunk", "choices": [
                    {"index": 0, "delta": {"content": WORDS[i % len(WORDS)] + " "}, "finish_reason": None}]})
            if (body.get("stream_options") or {}).get("include_usage"):
                self.send({**base, "object": "chat.completion.chunk", "choices": [], "usage": usage})
            self.wfile.write(b"data: [DONE]\n\n")
        self.close_connection = True

    def send(self, obj):
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        self.wfile.flush()


print(f"fake OpenAI-compatible server on http://localhost:{args.port}/v1")
ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
