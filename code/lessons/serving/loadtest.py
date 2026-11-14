"""Async load test for any OpenAI-compatible streaming endpoint (vLLM, llama-server, Ollama, SGLang).

    pip install openai
    python loadtest.py --base-url http://localhost:8000/v1 --model my-model --concurrency 8 --requests 32
"""
import argparse, asyncio, statistics, time
from openai import AsyncOpenAI

p = argparse.ArgumentParser()
p.add_argument("--base-url", default="http://localhost:8000/v1")
p.add_argument("--model", required=True)
p.add_argument("--concurrency", type=int, default=8)
p.add_argument("--requests", type=int, default=32)
p.add_argument("--max-tokens", type=int, default=128)
p.add_argument("--prompt", default="Write a short paragraph about rivers.")
args = p.parse_args()
client = AsyncOpenAI(base_url=args.base_url, api_key="none")
sem = asyncio.Semaphore(args.concurrency)

async def one_request():
    async with sem:                                   # at most --concurrency requests in flight
        t0, first, last, chunks, usage = time.perf_counter(), None, None, 0, None
        stream = await client.chat.completions.create(
            model=args.model, messages=[{"role": "user", "content": args.prompt}],
            max_tokens=args.max_tokens, stream=True, stream_options={"include_usage": True})
        async for chunk in stream:
            usage = chunk.usage or usage              # final chunk carries exact token counts
            if chunk.choices and chunk.choices[0].delta.content:
                last = time.perf_counter()
                first = first or last
                chunks += 1
        n = usage.completion_tokens if usage else chunks
        return first - t0, (last - first) / max(1, n - 1), n

def pct(xs, q):
    return sorted(xs)[min(len(xs) - 1, int(q * len(xs)))] * 1000

async def main():
    t0 = time.perf_counter()
    res = await asyncio.gather(*[one_request() for _ in range(args.requests)])
    wall = time.perf_counter() - t0
    ttft, tpot, toks = zip(*res)
    print(f"requests {len(res)}  concurrency {args.concurrency}  wall {wall:.1f} s")
    print(f"TTFT  p50 {pct(ttft, .5):.0f} ms  p99 {pct(ttft, .99):.0f} ms")
    print(f"TPOT  p50 {pct(tpot, .5):.1f} ms  p99 {pct(tpot, .99):.1f} ms  (mean {statistics.mean(tpot) * 1000:.1f})")
    print(f"throughput {sum(toks) / wall:.0f} output tokens/s  {len(res) / wall:.2f} requests/s")

asyncio.run(main())
