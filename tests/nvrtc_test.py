"""Compile real kernels for RTX 3060 and 5090 using only the packaged NVRTC and headers; no GPU needed."""
import ctypes as c
from pathlib import Path
import sys

cuda = Path(sys.argv[1]).resolve()
root = Path(__file__).resolve().parents[1]
nv = c.CDLL(str(cuda / 'lib64/libnvrtc.so'))
nv.nvrtcGetErrorString.restype = c.c_char_p
# Match the production loader: preload the versioned builtins beside NVRTC.
major, minor = c.c_int(), c.c_int()
assert nv.nvrtcVersion(c.byref(major), c.byref(minor)) == 0
builtins = c.CDLL(str(cuda / f'lib64/libnvrtc-builtins.so.{major.value}.{minor.value}'))


def check(code):
    if code:
        raise RuntimeError(nv.nvrtcGetErrorString(code).decode())


source = (root / 'src/kernels/kernels.cu').read_text().replace(
    '// TACH_PACKED_SOURCE', (root / 'src/packed.h').read_text()).encode()
names = ['gemv2<3,256,1,1,0>', 'gemv_tc<3,256,0>', 'gemm<3,0,128>', 'gemm<3,0,64>', 'gemm<2,2,64>',
         'rope_t<128,0>', 'attn_decode<128,4,8,0>', 'flash_prefill<128>', 'flash_wide<256>',
         'delta_rule<128,128>', 'dn_prep<128,128>', 'dn_chunk<128,128>',
         'causal_conv<0>', 'gemv_ind<3,256>', 'gemv_grp<3,256>',
         'attn_decode<128,1,16,0>', 'attn_decode<96,2,16,0>', 'attn_decode<512,1,16,2>',
         'gemm<1,0,64>', 'gemm<5,0,64>', 'gemm<7,1,128>', 'gemm<8,2,64>',
         'gemm<9,2,64>', 'gemm<10,1,128>', 'gemm<11,0,128>', 'packed_mv<5>', 'pmv2<6,1,4>', 'pmv2<11,1,1>', 'pmv2<11,8,2>', 'pmv_ind<9,1>']
for arch in (86, 120):
    for kv8 in (0, 1):
        program = c.c_void_p()
        check(nv.nvrtcCreateProgram(c.byref(program), source, b'kernels.cu', 0, None, None))
        try:
            for name in names:
                check(nv.nvrtcAddNameExpression(program, name.encode()))
            options = [f'--gpu-architecture=sm_{arch}', '-std=c++17', '-default-device',
                       '--extra-device-vectorization', '-DHID=256', '-DEPS=0.00001f',
                       '-DEQ=3', f'-DKV8={kv8}', '-I' + str(cuda / 'include')]
            args = (c.c_char_p * len(options))(*(s.encode() for s in options))
            code = nv.nvrtcCompileProgram(program, len(options), args)
            if code:
                size = c.c_size_t()
                check(nv.nvrtcGetProgramLogSize(program, c.byref(size)))
                log = c.create_string_buffer(size.value)
                check(nv.nvrtcGetProgramLog(program, log))
                print(log.value.decode(), file=sys.stderr)
                check(code)
            size = c.c_size_t()
            check(nv.nvrtcGetCUBINSize(program, c.byref(size)))
            assert size.value > 0
            print(f'Packaged NVRTC: sm_{arch}, KV8={kv8}, real CUDA kernels compiled ({size.value} bytes)', flush=True)
        finally:
            check(nv.nvrtcDestroyProgram(c.byref(program)))
