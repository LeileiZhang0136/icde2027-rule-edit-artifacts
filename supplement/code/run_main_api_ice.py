#!/usr/bin/env python3
"""Resumable, bounded-concurrency API ICE inference on frozen prompts only."""
import argparse
import concurrent.futures as cf
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--key-file', type=Path, required=True)
    p.add_argument('--base-url', required=True)
    p.add_argument('--model', default='gpt-6.1-sol')
    p.add_argument('--concurrency', type=int, default=8)
    p.add_argument('--max-completion-tokens', type=int, default=8192)
    p.add_argument('--reasoning-effort', default='medium')
    p.add_argument('--limit', type=int)
    p.add_argument('--timeout', type=int, default=300)
    a = p.parse_args()
    if a.concurrency < 1:
        p.error('concurrency must be positive')
    a.run_dir.mkdir(parents=True, exist_ok=True)
    lock = (a.run_dir / 'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    key = a.key_file.read_text().strip()
    inputs = rows(a.input)
    ids = [r['id'] for r in inputs]
    assert len(inputs) == 5100 and len(set(ids)) == 5100
    assert len({r['case_family_id'] for r in inputs}) == 1700
    assert all(r['split'] == 'test' and isinstance(r['prompt'], str) for r in inputs)
    output = a.run_dir / 'GPT-6.1-Sol-ICE.jsonl'
    errors = a.run_dir / 'errors.jsonl'
    attempts = a.run_dir / 'attempts.jsonl'
    raw_path = a.run_dir / 'api_responses.jsonl'
    manifest_path = a.run_dir / 'manifest.json'
    config = dict(model=a.model, base_url=a.base_url.rstrip('/'),
                  input_sha256=digest(a.input), max_completion_tokens=a.max_completion_tokens,
                  reasoning_effort=a.reasoning_effort,
                  sampling='provider default; temperature/top_p/seed omitted',
                  prompt_body='single user message; byte-identical frozen Qwen+ICE prompt',
                  method='GPT-6.1-Sol+ICE', information_condition='historical_rule+new_rule+domain_rule+case',
                  tools=False, provider_type='user-supplied third-party compatible API')
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        assert old['config'] == config, 'Configuration changed; use a new run directory'
    existing = rows(output) if output.exists() else []
    done = {r['id'] for r in existing}
    assert len(done) == len(existing) and done <= set(ids)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    started = time.time()
    usage = {}
    returned_models = {r.get('returned_model', '') for r in existing}
    for r in existing:
        for k, v in r.get('usage', {}).items():
            if isinstance(v, (int, float)): usage[k] = usage.get(k, 0) + v

    def append(path, row):
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())

    def save(status, detail=None):
        obj = dict(config=config, input_path=str(a.input.resolve()), script_sha256=digest(Path(__file__)),
                   expected_rows=len(inputs), rows=len(done), status=status, detail=detail,
                   concurrency=a.concurrency, usage=usage, updated_at_unix=time.time(),
                   elapsed_this_process_sec=round(time.time()-started, 2),
                   output_sha256=digest(output) if output.exists() else None,
                   returned_models=sorted(returned_models),
                   api_identity_note='Returned model IDs are recorded; upstream model identity is not independently attested.')
        tmp = manifest_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
        tmp.replace(manifest_path)

    def call(row):
        body = dict(model=a.model, messages=[dict(role='user', content=row['prompt'])],
                    reasoning_effort=a.reasoning_effort, max_completion_tokens=a.max_completion_tokens)
        body_bytes = json.dumps(body, ensure_ascii=False).encode()
        for attempt in range(1, 5):
            if stop.is_set(): return row, None, dict(error='stopped', fatal=False)
            t0 = time.time()
            try:
                req = Request(a.base_url.rstrip('/')+'/chat/completions', data=body_bytes,
                              headers={'Authorization':'Bearer '+key, 'Content-Type':'application/json'})
                with urlopen(req, timeout=a.timeout) as resp:
                    result = json.load(resp)
                choice = result['choices'][0]
                content = choice['message'].get('content')
                if isinstance(content, list): content = ''.join(x.get('text','') for x in content if isinstance(x,dict))
                if not isinstance(content, str) or not content.strip():
                    return row, result, dict(error='empty visible answer', fatal=True, attempt=attempt)
                if choice.get('finish_reason') != 'stop':
                    return row, result, dict(error='non-stop finish: '+str(choice.get('finish_reason')), fatal=True, attempt=attempt)
                return row, result, dict(attempt=attempt, latency_sec=round(time.time()-t0,3), request_sha256=hashlib.sha256(body_bytes).hexdigest())
            except HTTPError as e:
                detail = e.read().decode(errors='replace').replace(key,'[REDACTED]')[:1000]
                err = dict(error='HTTP '+str(e.code), detail=detail, attempt=attempt,
                           fatal=e.code in (400,401,402,403,404), latency_sec=round(time.time()-t0,3))
            except (URLError, TimeoutError, ValueError, KeyError, OSError) as e:
                err = dict(error=type(e).__name__, detail=str(e).replace(key,'[REDACTED]')[:1000], attempt=attempt, fatal=False)
            if err['fatal'] or attempt == 4: return row, None, err
            stop.wait(min(2**attempt, 20))

    pending_rows = [r for r in inputs if r['id'] not in done]
    if a.limit: pending_rows = pending_rows[:a.limit]
    save('running')
    print(json.dumps(dict(status='running', already_done=len(done), selected=len(pending_rows), concurrency=a.concurrency)), flush=True)
    failure = None
    iterator = iter(pending_rows)
    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        futures = {}
        def schedule():
            if not stop.is_set():
                row = next(iterator, None)
                if row: futures[pool.submit(call,row)] = row
        for _ in range(a.concurrency): schedule()
        while futures:
            ready, _ = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
            for f in ready:
                row = futures.pop(f)
                _, result, info = f.result()
                record = dict(id=row['id'], timestamp=time.time(), **info)
                append(attempts, record)
                if result is not None: append(raw_path, dict(id=row['id'], response=result))
                if 'error' in info:
                    append(errors, record)
                    if not failure: failure = record
                    stop.set()
                    print(json.dumps(record, ensure_ascii=False),flush=True)
                else:
                    result_row = {k:v for k,v in row.items() if k != 'prompt'}
                    result_row.update(answer=result['choices'][0]['message']['content'],
                                      method=config['method'], model=a.model, returned_model=result.get('model'),
                                      response_id=result.get('id'), usage=result.get('usage',{}),
                                      system_fingerprint=result.get('system_fingerprint'), **info)
                    append(output, result_row)
                    done.add(row['id'])
                    returned_models.add(result.get('model', ''))
                    for k,v in result.get('usage',{}).items():
                        if isinstance(v,(int,float)): usage[k] = usage.get(k,0)+v
                    if len(done) % 10 == 0 or a.limit:
                        print(json.dumps(dict(completed=len(done), total=len(inputs), usage=usage, elapsed_sec=round(time.time()-started))),flush=True)
                if len(done) % 10 == 0 or stop.is_set():
                    save('running' if not stop.is_set() else 'stopping', failure)
                schedule()
    status = 'complete' if len(done) == len(inputs) else ('failed' if failure else 'partial')
    save(status, failure)
    print(json.dumps(dict(status=status, completed=len(done), total=len(inputs))),flush=True)
    return 1 if failure else 0


if __name__ == '__main__':
    raise SystemExit(main())
