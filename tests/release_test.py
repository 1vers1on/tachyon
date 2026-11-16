"""Offline CLI, malformed tokenizer and model-manager release regressions."""
import copy
import http.client
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from server_test import fixture, request, serve

binary = str(Path(sys.argv[1]).resolve())
client = str(Path(sys.argv[2] if len(sys.argv) > 2 else './tachyon-client').resolve())


def cli(*args, valid=True, message=''):
    p = subprocess.run([binary, *args], env=env, capture_output=True, text=True, timeout=15)
    assert p.returncode == (0 if valid else 1), (args, p.returncode, p.stderr)
    assert message in p.stderr, (args, message, p.stderr)
    return p


with tempfile.TemporaryDirectory(prefix='tachyon-release-') as tmp:
    root = Path(tmp)
    env = dict(os.environ, TACHYON_HOME=str(root / 'store'))
    assert cli('--version').stdout.strip() == '1.1.1'
    model = root / 'tiny.tach'
    fixture(model, 'beforeFIRSTmiddleSECONDafter')
    cli('run', '-m', str(model), '--device', 'typo', valid=False, message='invalid device')
    cli('run', '-m', str(model), '--device', 'cpu', '-c', '2', '-p', 'long prompt',
        valid=False, message='exceeds context')
    cli('bench', '-m', str(model), '--prompt-len', '0', valid=False, message='must be positive')
    with serve(['-m', str(model), '--device', 'cpu', '-t', '2', '--spec', '0'], env) as port:
        assert request(port, '/api/version')['version'] == '1.1.1'
        body = dict(prompt='hi', max_tokens=1, temperature=0, stop=['SECOND', 'FIRST'])
        assert request(port, '/v1/completions', body)['choices'][0]['text'] == 'before'
        c = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
        c.request('POST', '/v1/completions', json.dumps(dict(body, stream=True)))
        response = c.getresponse()
        chunks = [json.loads(line[6:]) for line in response.read().decode().splitlines()
                  if line.startswith('data: ') and line != 'data: [DONE]']
        assert ''.join(x['choices'][0]['text'] for x in chunks) == 'before', chunks
        c.close()
    cli('import', str(model), 'qwen3:8b')
    assert 'qwen3:8b' in cli('show', 'qwen3:latest').stdout
    key = 'release-test'
    env['TACHYON_API_KEY'] = key
    with serve(['-m', str(model), '--device', 'cpu', '-t', '2', '--max-tokens', '1'], env) as port:
        for variable in ('OPENAI_API_KEY', 'TACHYON_API_KEY'):
            client_env = dict(env)
            client_env.pop('OPENAI_API_KEY', None)
            client_env.pop('TACHYON_API_KEY', None)
            client_env[variable] = key
            p = subprocess.run([client, '--url', f'http://127.0.0.1:{port}', '-p', 'hi', '-n', '1', '--temp', '0'],
                               env=client_env, capture_output=True, text=True, timeout=15)
            assert p.returncode == 0 and 'beforeFIRST' in p.stdout, (variable, p.returncode, p.stderr)
    with serve(['--no-auto-pull', '--device', 'cpu', '-t', '2'], env) as port:
        # Named CLI requests must authenticate to the manager, and child processes must not inherit its sockets.
        env['TACHYON_HOST'] = f'127.0.0.1:{port}'
        cli('run', 'qwen3:latest', '--temp', '0', '--max-tokens', '1', 'hi')
        runners = request(port, '/api/ps', headers={'Authorization': 'Bearer ' + key})['models']
        assert len(runners) == 1, runners
        manager_sockets = set()
        for line in Path('/proc/net/tcp').read_text().splitlines()[1:]:
            cols = line.split()
            if int(cols[1].split(':')[1], 16) == port and cols[3] == '0A':
                manager_sockets.add('socket:[' + cols[9] + ']')
        child_fds = Path(f'/proc/{runners[0]["pid"]}/fd')
        assert manager_sockets and not any(os.readlink(p) in manager_sockets for p in child_fds.iterdir())
        for route, method in [('/api/pull', 'DELETE'), ('/api/delete', 'POST')]:
            c = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
            c.request(method, route, json.dumps(dict(model='qwen3:8b')), {'Authorization': 'Bearer ' + key})
            r = c.getresponse()
            assert r.status == 405, (route, r.status, r.read())
            r.read(); c.close()
        cli('ps')
        cli('stop', 'qwen3:8b')
        assert request(port, '/api/ps', headers={'Authorization': 'Bearer ' + key})['models'] == []

    raw = model.read_bytes()
    _, offset, length = struct.unpack('<8sQQ', raw[:24])
    header = json.loads(raw[offset:offset + length])
    prefix = raw[:offset]

    def malformed(edit, error, command='tokenize'):
        h = copy.deepcopy(header)
        data = bytearray(prefix)
        edit(h, data)
        body = json.dumps(h).encode()
        struct.pack_into('<Q', data, 16, len(body))
        model.write_bytes(data + body)
        cli(command, '-m', str(model), '-p', 'hi', '--device', 'cpu', valid=False, message=error)

    malformed(lambda h, d: h['tokenizer'].update(n_vocab=-1), 'invalid tokenizer vocabulary size')
    malformed(lambda h, d: h['tokenizer'].update(n_vocab=2**32 + 288), 'invalid tokenizer vocabulary size')
    malformed(lambda h, d: next(b for b in h['blobs'] if b['name'] == 'tok.type').update(size=1), 'blob sizes')
    malformed(lambda h, d: struct.pack_into('<I', d, 4096 + next(b['offset'] for b in h['blobs'] if b['name'] == 'tok.vocab_off') + 4, 2**32 - 1), 'vocabulary offsets')
    malformed(lambda h, d: h['tokenizer'].update(specials=[dict(content='bad', id=9999)]), 'token id')
    malformed(lambda h, d: h['tokenizer'].update(eos=[9999]), 'token id')
    malformed(lambda h, d: h['tokenizer'].update(bos=9999), 'token id')
    malformed(lambda h, d: h['config'].update(hidden=512), 'shape/format', 'run')
    malformed(lambda h, d: h['config'].update(n_heads=0), 'attention config', 'run')
    malformed(lambda h, d: h['config'].update(n_layers=2**30), 'layer count', 'run')
    malformed(lambda h, d: h['config'].update(layers=[dict(type='attn', n_heads=4, n_kv_heads=4,
                                                         head_dim=64, rope_half=64)]), 'attention config', 'run')
    malformed(lambda h, d: h['config'].update(layers=[dict(type='attn', n_heads=4, n_kv_heads=4,
                                                         head_dim=64, kv_src=0)]), 'attention config', 'run')
print('CLI failures, earliest stops, aliases, authentication, child sockets and malformed model metadata passed')
