"""Offline HTTP/API regressions with tiny deterministic models; Python standard library only."""
import contextlib
import http.client
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time

binary = str(Path(sys.argv[1] if len(sys.argv) > 1 else './tachyon').resolve())


def fixture(path, output='x', thinking=False, vocab=288):
    """A one-layer Q8 model whose output is always token 256."""
    h, v = 256, vocab
    tensors, blobs, data = [], [], bytearray()

    def put(raw):
        data.extend(b'\0' * (-len(data) % 4096))
        offset = len(data)
        data.extend(raw)
        return offset

    def mat(name, rows, cols, raw):
        tensors.append(dict(name=name, type='q8', rows=rows, cols=cols,
                            offset=put(raw), data_bytes=len(raw),
                            scale_offset=put(struct.pack('<e', 1) * (rows * cols // 32)),
                            scale_bytes=rows * cols // 16))

    def vec(name):
        raw = struct.pack('<f', 1) * h
        tensors.append(dict(name=name, type='f32', rows=1, cols=h,
                            offset=put(raw), data_bytes=len(raw)))

    mat('tok_embd', v, h, b'\1' * (v * h))
    mat('output', v, h, b'\0' * (256 * h) + b'\1' * h + b'\0' * ((v - 257) * h))
    vec('output_norm')
    vec('blk.0.attn_norm')
    vec('blk.0.ffn_norm')
    mat('blk.0.wqkv', 3 * h, h, b'\0' * (3 * h * h))
    mat('blk.0.wo', h, h, b'\0' * (h * h))
    vocab = [bytes([i]) for i in range(256)] + [output.encode()] + [f'[unused{i}]'.encode() for i in range(v - 257)]
    offsets = [0]
    for token in vocab:
        offsets.append(offsets[-1] + len(token))
    for name, raw in [('tok.vocab_data', b''.join(vocab)),
                      ('tok.vocab_off', struct.pack('<' + 'I' * len(offsets), *offsets)),
                      ('tok.type', b'\0' * v), ('tok.merges', b'')]:
        blobs.append(dict(name=name, offset=put(raw), size=len(raw)))
    header = dict(config=dict(name=path.stem, arch='qwen2', hidden=h, vocab=v,
                             n_layers=1, n_heads=4, n_kv_heads=4, head_dim=64,
                             ffn=0, max_ctx=1024, wtype='q8'),
                  tokenizer=dict(n_vocab=v, thinking=thinking, think_default=False,
                                 chat_format='chatml', think_disable_block=False),
                  tensors=tensors, blobs=blobs)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack('<8sQQ', b'TACHYON2', 4096 + len(data), len(raw))
                     + b'\0' * (4096 - 24) + data + raw)


@contextlib.contextmanager
def serve(args, env):
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    with tempfile.TemporaryFile() as log:
        p = subprocess.Popen([binary, 'serve', '--port', str(port), *args], env=env, stdout=log, stderr=log)
        try:
            for _ in range(200):
                if p.poll() is not None:
                    log.seek(0)
                    raise AssertionError(log.read().decode())
                try:
                    c = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
                    c.request('GET', '/health')
                    c.getresponse().read()
                    c.close()
                    break
                except OSError:
                    time.sleep(.05)
            else:
                raise AssertionError('server did not become healthy')
            yield port
        finally:
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


def request(port, route, body=None, status=200, headers=None):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=15)
    try:
        c.request('GET' if body is None else 'POST', route,
                  None if body is None else json.dumps(body), headers or {})
        r = c.getresponse()
        raw = r.read()
        assert r.status == status, (route, r.status, raw)
        if 'ndjson' in r.getheader('Content-Type', ''):
            chunks = [json.loads(line) for line in raw.splitlines() if line.strip()]
            assert chunks and chunks[-1]['done'] is True, chunks
            assert all(x['done'] is False for x in chunks[:-1]), chunks
            return chunks
        return json.loads(raw)
    finally:
        c.close()


def suite(port, model):
    assert request(port, '/api/version')['version']
    assert request(port, '/api/tags')['models'][0]['name'] == model
    show = request(port, '/api/show', dict(model=model))
    assert 'completion' in show['capabilities'] and show['model_info']['qwen2.context_length'] == 1024
    request(port, '/api/chat', [], 400)
    request(port, '/api/chat', dict(model=model, messages='oops'), 400)
    request(port, '/api/chat', dict(model=model, messages=[dict(role='user', content=1)]), 400)
    request(port, '/api/chat', dict(model=model, messages=[], think='high'), 400)
    request(port, '/api/generate', dict(prompt='hi'), 400)
    request(port, '/api/generate', dict(model=model, prompt=4), 400)
    request(port, '/api/generate', dict(model='missing', prompt='hi'), 404)
    for opts in [dict(temperature=-1), dict(top_p=2), dict(repeat_penalty=0), dict(num_ctx=0), dict(num_ctx=2000), dict(num_predict=-3), dict(num_predict=1e99), dict(num_ctx=4.5), dict(temperature='hot')]:
        request(port, '/api/generate', dict(model=model, prompt='hi', options=opts), 400)
    for route in ['/api/chat', '/api/generate']:
        body = dict(model=model, options=dict(num_predict=3, temperature=0))
        body.update(dict(messages=[dict(role='user', content='hi')]) if route.endswith('chat') else dict(prompt='hi'))
        j = request(port, route, dict(body, stream=False))
        assert j['done'] and j['done_reason'] == 'length' and j['eval_count'] == 3, j
        assert j.get('message', {}).get('content', j.get('response')) == 'xxx', j
        for key in ['total_duration', 'load_duration', 'prompt_eval_count', 'prompt_eval_duration', 'eval_count', 'eval_duration']:
            assert isinstance(j[key], int) and j[key] >= 0, (key, j)
        chunks = request(port, route, body)  # streaming by default
        assert ''.join(x.get('message', {}).get('content', x.get('response', '')) for x in chunks) == 'xxx', chunks
        assert chunks[-1]['eval_count'] == 3
    j = request(port, '/api/generate', dict(model=model, prompt='hi', raw=True, stream=False,
                                          options=dict(num_predict=5, stop=['xx'], temperature=0)))
    assert j['response'] == '' and j['done_reason'] == 'stop', j
    j = request(port, '/api/generate', dict(model=model, prompt='hi', raw=True, stream=False,
                                          options=dict(num_predict=5, num_ctx=4, temperature=0)))
    assert j['eval_count'] == 2 and j['prompt_eval_count'] == 2, j
    j = request(port, '/api/generate', dict(model=model, prompt='hi', raw=True, stream=False,
                                          options=dict(num_predict=0, temperature=0)))
    assert j['eval_count'] == 0 and j['response'] == '', j
    j = request(port, '/v1/completions', dict(model=model, prompt=[256], max_tokens=1,
                                            presence_penalty=1000, temperature=0))
    assert j['choices'][0]['text'] != 'x', j
    body = dict(model=model, prompt='hi', raw=True, stream=False,
                options=dict(num_predict=8, temperature=1000, top_k=0, top_p=1, seed=0))
    assert request(port, '/api/generate', body)['response'] == request(port, '/api/generate', body)['response']
    request(port, '/v1/completions', dict(prompt=[-1], max_tokens=1), 400)
    assert request(port, '/v1/chat/completions', dict(messages=[dict(role='user', content='hi')], max_tokens=2, temperature=0))['choices'][0]['message']['content'] == 'xx'
    assert request(port, '/v1/messages', dict(messages=[dict(role='user', content='hi')], max_tokens=2, temperature=0))['content'][0]['text'] == 'xx'


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='tachyon-api-') as tmp:
        root = Path(tmp)
        model = root / 'tiny.tach'
        fixture(model)
        if len(sys.argv) > 2:
            subprocess.run([str(Path(sys.argv[2]).resolve()), str(model)], check=True)
        env = dict(os.environ, TACHYON_HOME=str(root / 'store'))
        with serve(['-m', str(model), '--device', 'cpu', '-c', '1024', '-t', '2', '--alias', 'tiny'], env) as port:
            suite(port, 'tiny')
            # Invalid framing must be rejected without reading an attacker-selected body length.
            for headers, body in [
                ('Content-Length: -1', ''), ('Content-Length: 2x', ''),
                ('Content-Length: 0\r\nContent-Length: 1', ''),
                ('Content-Length: 0\r\nTransfer-Encoding: chunked', ''),
                ('Transfer-Encoding: chunked', 'garbage\r\n\r\n'),
                ('Transfer-Encoding: chunked', '1\r\nx!!'),
                ('Transfer-Encoding: chunked', 'ffffffffffffffff\r\n')]:
                with socket.create_connection(('127.0.0.1', port), timeout=2) as s:
                    s.sendall(f'POST /tokenize HTTP/1.1\r\nHost: localhost\r\n{headers}\r\n\r\n{body}'.encode())
                    assert s.recv(1024).startswith((b'HTTP/1.1 400', b'HTTP/1.1 413'))
            # A chunked request followed by a pipelined request shares one connection.
            c = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
            c.request('POST', '/tokenize', iter([b'{"content":"hi"}']), encode_chunked=True)
            assert json.loads(c.getresponse().read())['tokens'] == [104, 105]
            c.request('GET', '/api/version')
            assert json.loads(c.getresponse().read())['version']
            c.close()
        subprocess.run([binary, 'import', str(model), 'tiny'], env=env, check=True, capture_output=True)
        with serve(['--no-auto-pull', '--runner-args', '--device cpu -c 1024 -t 2 --spec 0'], env) as port:
            suite(port, 'tiny')
            j = request(port, '/api/generate', dict(model='tiny', prompt='hi', keep_alive=0, stream=False,
                                                  options=dict(num_predict=2, temperature=0)))
            assert j['response'] == 'xx' and j['eval_count'] == 2, j
            assert request(port, '/api/ps')['models'] == []
            assert request(port, '/api/generate', dict(model='tiny', stream=False, keep_alive='1h'))['done_reason'] == 'load'
            assert request(port, '/api/ps')['models'][0]['unload_in_seconds'] > 3590
            assert request(port, '/api/generate', dict(model='tiny', keep_alive=0, stream=False))['done_reason'] == 'unload'
            assert request(port, '/api/ps')['models'] == []
        for name, output, thinking in [
            ('thought', '<think>why</think>ok', True),
            ('tool', '<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>', False)]:
            model = root / f'{name}.tach'
            fixture(model, output, thinking)
            with serve(['-m', str(model), '--device', 'cpu', '-c', '1024', '-t', '2', '--alias', name], env) as port:
                body = dict(model=name, messages=[dict(role='user', content='hi')],
                            options=dict(num_predict=1, temperature=0), think=thinking)
                if name == 'tool':
                    body['tools'] = [dict(type='function', function=dict(name='weather', parameters=dict(type='object')))]
                for stream in [False, True]:
                    result = request(port, '/api/chat', dict(body, stream=stream))
                    messages = [x['message'] for x in result] if stream else [result['message']]
                    if thinking:
                        assert ''.join(x.get('thinking', '') for x in messages) == 'why', messages
                        assert ''.join(x.get('content', '') for x in messages) == 'ok', messages
                    else:
                        calls = [tc for x in messages for tc in x.get('tool_calls', [])]
                        assert calls[0]['function']['arguments'] == dict(city='Paris'), calls
                        history = body['messages'] + [dict(role='assistant', content='', tool_calls=calls),
                                                     dict(role='tool', tool_name='weather', content='sunny')]
                        reply = request(port, '/api/chat', dict(body, messages=history, stream=False))
                        assert reply['message']['tool_calls'][0]['function']['name'] == 'weather', reply
        print('HTTP, OpenAI, Anthropic, Ollama and model lifecycle regressions passed')
