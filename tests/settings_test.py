"""Generation defaults, request controls and explicit server-start regressions."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

from server_test import fixture, request, serve

binary = str(Path(sys.argv[1]).resolve())
device = os.environ.get('TACH_TEST_DEVICE', 'cpu')


def check_defaults(port):
    for route, base, key in [
        ('/v1/completions', dict(prompt='hi'), 'choices'),
        ('/api/generate', dict(model='tiny', prompt='hi', stream=False), 'response'),
        ('/v1/messages', dict(messages=[dict(role='user', content='hi')]), 'content')]:
        result = request(port, route, base)
        text = result[key][0]['text'] if key in ('choices', 'content') else result[key]
        assert text == 'xx', result
        body = dict(base)
        if route.startswith('/api/'):
            body['options'] = dict(num_predict=3)
        else:
            body['max_tokens'] = 3
        result = request(port, route, body)
        text = result[key][0]['text'] if key in ('choices', 'content') else result[key]
        assert text == 'xxx', result


def check_controls(port):
    def completion(**options):
        body = dict(prompt='cache check', max_tokens=1, temperature=0)
        body.update(options)
        return request(port, '/v1/completions', body)

    completion(cache_prompt=True)
    assert completion(cache_prompt=True)['usage']['prompt_tokens_details']['cached_tokens'] > 0
    assert completion(cache_prompt=False)['usage']['prompt_tokens_details']['cached_tokens'] == 0
    assert completion(cache_prompt=False)['usage']['prompt_tokens_details']['cached_tokens'] == 0
    # The fixture's winning token (256) is penalized enough that a +100 bias can select token 65.
    result = completion(prompt=[256], presence_penalty=200, logit_bias={'65': 100})
    assert result['choices'][0]['text'] == 'A', result
    result = completion(prompt=[256], presence_penalty=1000, penalize_prompt=False)
    assert result['choices'][0]['text'] == 'x', result
    result = completion(prompt=[256], presence_penalty=1000, repeat_last_n=0)
    assert result['choices'][0]['text'] == 'x', result
    no_drafts = completion(max_tokens=20, speculative_tokens=0)
    assert no_drafts['timings']['draft_tokens'] == 0, no_drafts
    # Typical sampling must produce the same result from compact GPU candidates and full logits.
    stochastic = dict(max_tokens=8, temperature=1000, top_k=0, top_p=1, typical_p=.01,
                      repeat_last_n=0, seed=0, speculative_tokens=0)
    compact = completion(**stochastic)['choices'][0]['text']
    full = completion(**stochastic, logit_bias={'0': 0})['choices'][0]['text']
    assert compact == full and set(compact) <= {'\x00', '\x01', '\x02'}, (compact, full)
    if device == 'cuda':
        drafts = completion(max_tokens=20, speculative_tokens=4)
        assert drafts['timings']['draft_tokens'] > 0, drafts
    for options in [dict(typical_p=0), dict(repeat_last_n=-2), dict(cache_prompt='no'),
                    dict(logit_bias={'9999': 1}), dict(logit_bias={'1': 101}),
                    dict(speculative_tokens=9), dict(top_k=2.5), dict(seed='random')]:
        request(port, '/v1/completions', dict(prompt='hi', **options), 400)
        request(port, '/api/generate', dict(model='tiny', prompt='hi', options=options), 400)
        request(port, '/v1/messages', dict(messages=[dict(role='user', content='hi')], **options), 400)


with tempfile.TemporaryDirectory(prefix='tachyon-settings-') as tmp:
    root = Path(tmp)
    model = root / 'tiny.tach'
    fixture(model)
    config = root / 'sampling.json'
    config.write_text(json.dumps(dict(temperature=.2, max_tokens=2, typical_p=.9,
                                     repeat_last_n=64, cache_prompt=False)))
    env = dict(os.environ, TACHYON_HOME=str(root / 'store'))
    backend = ['--device', device, '-c', '1024', '--slots', '1', '-t', '2', '--spec', '4']
    defaults = ['--sampling-config', str(config), '--temp', '0']
    with serve(['-m', str(model), '--alias', 'tiny', *backend, *defaults], env) as port:
        settings = request(port, '/api/settings')
        assert settings['generation']['temperature'] == 0
        assert settings['generation']['max_tokens'] == 2
        assert settings['generation']['repeat_last_n'] == 64
        assert settings['backend']['num_ctx'] == 1024
        check_defaults(port)
        check_controls(port)
        # Request-level stops replace server defaults when explicitly provided.
        result = request(port, '/api/generate', dict(model='tiny', prompt='hi', stream=False,
                                                   options=dict(stop=['xx'], num_predict=3)))
        assert result['response'] == '' and result['done_reason'] == 'stop', result
    with serve(['-m', str(model), '--alias', 'tiny', *backend, '--temp', '0', '--max-tokens', '3', '--stop', 'xx'], env) as port:
        base = dict(model='tiny', prompt='hi', stream=False)
        assert request(port, '/api/generate', base)['response'] == ''
        assert request(port, '/api/generate', dict(base, options=dict(stop=[])))['response'] == 'xxx'
        assert request(port, '/v1/completions', dict(prompt='hi', stop=[]))['choices'][0]['text'] == 'xxx'
        assert request(port, '/v1/messages', dict(messages=[dict(role='user', content='hi')], stop_sequences=[]))['content'][0]['text'] == 'xxx'
        assert request(port, '/v1/completions', dict(prompt='hi', stop=[], temperature=None, max_tokens=None))['choices'][0]['text'] == 'xxx'
    with serve(['-m', str(model), '--alias', 'tiny', *backend, '--api-key', 'settings-test'], env) as port:
        request(port, '/api/settings', status=401)
        assert request(port, '/api/settings', headers={'Authorization': 'Bearer settings-test'})['model'] == 'tiny'
    p = subprocess.run([binary, 'run', '-m', str(model), '-p', 'hi', *backend, *defaults],
                       env=env, capture_output=True, text=True, timeout=30)
    assert p.returncode == 0 and p.stdout.strip() == 'xx', (p.stdout, p.stderr)
    subprocess.run([binary, 'import', str(model), 'tiny'], env=env, check=True, capture_output=True)
    with serve(['--no-auto-pull', *backend, *defaults], env) as port:
        settings = request(port, '/api/settings')
        assert settings['manager'] and settings['generation_overrides']['temperature'] == 0
        check_defaults(port)
        check_controls(port)
        client_env = dict(env, TACHYON_HOST=f'127.0.0.1:{port}', TACHYON_AUTOSTART='0')
        p = subprocess.run([binary, 'run', 'tiny', '--max-tokens', '3', '--temp', '0', 'hi'],
                           env=client_env, capture_output=True, text=True, timeout=30)
        assert p.returncode == 0 and p.stdout.strip() == 'xxx', (p.stdout, p.stderr)
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        unused = s.getsockname()[1]
        for explicit in [False, True]:
            client_env = dict(env, TACHYON_HOST=f'127.0.0.1:{unused}', TACHYON_AUTOSTART='1' if explicit else '0')
            p = subprocess.run([binary, 'run', 'tiny', *(['--no-autostart'] if explicit else []), 'hi'],
                               env=client_env, capture_output=True, text=True, timeout=10)
            assert p.returncode == 1 and 'auto-start is disabled' in p.stderr, p.stderr
    config.write_text('{"unknown_setting": 1}')
    p = subprocess.run([binary, 'serve', '-m', str(model), '--sampling-config', str(config)],
                       env=env, capture_output=True, text=True, timeout=5)
    assert p.returncode and 'unknown inference setting' in p.stderr, p.stderr
print(f'Inference settings, overrides, cache, biases and no-autostart checks passed ({device})')
