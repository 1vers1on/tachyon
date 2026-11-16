"""HTTP clients must reject malformed/truncated responses and failed generation streams."""
import contextlib
from pathlib import Path
import socketserver
import subprocess
import sys
import threading

probe, client = (str(Path(p).resolve()) for p in sys.argv[1:3])


@contextlib.contextmanager
def server(response):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            data = b''
            while b'\r\n\r\n' not in data:
                chunk = self.request.recv(4096)
                if not chunk:
                    return
                data += chunk
            if data.startswith(b'GET /v1/models '):
                body = b'{"data":[{"id":"tiny"}]}'
                raw = b'HTTP/1.1 200 OK\r\nContent-Length: ' + str(len(body)).encode() + b'\r\n\r\n' + body
            else:
                raw = response(data) if callable(response) else response
            self.request.sendall(raw)

    with socketserver.TCPServer(('127.0.0.1', 0), Handler) as srv:
        thread = threading.Thread(target=srv.serve_forever, kwargs=dict(poll_interval=.01))
        thread.start()
        try:
            yield str(srv.server_address[1])
        finally:
            srv.shutdown()
            thread.join()


checks = 0
for raw, expected in [
    (b'Content-Length: 5\r\n\r\nhello', b'hello'),
    (b'X-Content-Length: 1\r\n\r\nhello', b'hello'),
    (b'Transfer-Encoding: Chunked\r\n\r\n2;foo=bar\r\nhe\r\n3\r\nllo\r\n0\r\nX-Trailer: yes\r\n\r\n', b'hello'),
    (b'Content-Length: 0\r\n\r\n', b''),
    (b'Content-Length: 5\r\n\r\nhi', None),
    (b'Content-Length: -1\r\n\r\nhi', None),
    (b'Content-Length: 5x\r\n\r\nhello', None),
    (b'Content-Length: 5\r\nContent-Length: 5\r\n\r\nhello', None),
    (b'Content-Length: 0\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n', None),
    (b'Transfer-Encoding: chunked\r\n\r\n-1\r\nhello', None),
    (b'Transfer-Encoding: chunked\r\n\r\ngarbage\r\n', None),
    (b'Transfer-Encoding: chunked\r\n\r\nfffffffffffffffff\r\n', None),
    (b'Transfer-Encoding: chunked\r\n\r\n5\r\nhi', None),
    (b'Transfer-Encoding: chunked\r\n\r\n2\r\nhi!!0\r\n\r\n', None),
    (b'Transfer-Encoding: chunked\r\n\r\n2\r\nhi\r\n', None),
    (b'Transfer-Encoding: chunked\r\n\r\n0\r\n', None),
]:
    with server(b'HTTP/1.1 200 OK\r\n' + raw) as port:
        for mode in ('call', 'get'):
            p = subprocess.run([probe, mode, port, '/'], capture_output=True, timeout=5)
            assert p.returncode == (1 if expected is None else 0), (mode, raw, p)
            if expected is not None:
                assert p.stdout == expected, (mode, raw, p.stdout)
            checks += 1

with server(b'HTTP/1.1 103 Early Hints\r\nLink: </x>\r\n\r\nHTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok') as port:
    for mode in ('call', 'get'):
        p = subprocess.run([probe, mode, port, '/'], capture_output=True, timeout=5)
        assert p.returncode == 0 and p.stdout == b'ok', p
        checks += 1

# Authorization must stay on the original origin, including its port.
seen = []


def redirected(data):
    seen.append(data)
    return b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok'


with server(redirected) as other:
    with server(b'HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:' + other.encode() + b'/final\r\nContent-Length: 0\r\n\r\n') as port:
        p = subprocess.run([probe, 'get-auth', port, '/'], capture_output=True, timeout=5)
        assert p.returncode == 0 and p.stdout == b'ok', p
        assert len(seen) == 1 and b'Authorization:' not in seen[0], seen
        assert b'Host: 127.0.0.1:' + other.encode() in seen[0], seen
        checks += 1

for payload, valid in [
    ('data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n', True),
    ('data: {"error":{"message":"generation failed"}}\n\n', False),
    ('data: {"choices":[{"delta":{"content":"partial"}}]}\n\n', False),
]:
    raw = payload.encode()
    with server(b'HTTP/1.1 200 OK\r\nContent-Length: ' + str(len(raw)).encode() + b'\r\n\r\n' + raw) as port:
        p = subprocess.run([client, '--url', 'http://127.0.0.1:' + port, '-p', 'hi'],
                           capture_output=True, timeout=5)
        assert p.returncode == (0 if valid else 1), (payload, p)
        checks += 1
print(f'HTTP client framing and stream completion: {checks} checks passed')
