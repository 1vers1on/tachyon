// Tachyon - runtime loading of libcuda (driver) and libnvrtc (kernel compiler)
#include "cuda_api.h"
#include <dlfcn.h>
#include <link.h>
#include <sys/stat.h>
#include <cstdlib>
#include <mutex>
#include <vector>
#include <unistd.h>
#include <limits.h>

#ifndef TACH_CUDA_DIR
#define TACH_CUDA_DIR "/usr/local/cuda"
#endif

namespace {

#define TACH_STR2(x) #x
#define TACH_STR(x) TACH_STR2(x)

bool is_dir(const std::string& p) { struct stat st; return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode); }

void* open_first(const std::vector<std::string>& names, std::string& tried) {
    for (auto& n : names) {
        if (void* h = dlopen(n.c_str(), RTLD_NOW | RTLD_LOCAL)) return h;
        tried += (tried.empty() ? "" : ", ") + n;
    }
    return nullptr;
}

// toolkit directories to search, most specific first
std::vector<std::string> cuda_dirs() {
    std::vector<std::string> d;
    for (const char* v : {"TACHYON_CUDA", "CUDA_PATH", "CUDA_HOME"})
        if (const char* e = getenv(v); e && *e) d.push_back(e);
    // Release layout: libexec/tachyon-<isa> beside cuda/{lib64,include}.
    // Resolve the running executable, including when invoked through an install symlink.
    char executable[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
    if (n > 0) {
        executable[n] = 0;
        std::string p = executable;
        d.push_back(p.substr(0, p.rfind('/')) + "/../cuda");
    }
    d.push_back(TACH_CUDA_DIR);
    d.push_back("/usr/local/cuda");
    return d;
}

void load(CudaApi& a) {
    std::string tried;
    void* cu = open_first({"libcuda.so.1", "libcuda.so", "/usr/lib/wsl/lib/libcuda.so.1"}, tried);
    if (!cu) { a.error = "NVIDIA driver library not found (tried " + tried + ")"; return; }

    // Prefer explicit toolkit settings and the matching bundled headers/libraries before system NVRTC.
    std::vector<std::string> names;
    if (const char* e = getenv("TACHYON_NVRTC"); e && *e) names.push_back(e);
    static const char* sonames[] = {"libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11.2"};
    for (auto& dir : cuda_dirs())
        for (const char* sub : {"/lib64/", "/lib/", "/targets/x86_64-linux/lib/"})
            for (auto s : sonames) names.push_back(dir + sub + s);
    for (auto s : sonames) names.push_back(s);
    tried.clear();
    void* nv = open_first(names, tried);
    if (!nv) { a.error = "NVRTC library (libnvrtc.so, part of the CUDA toolkit) not found; set TACHYON_NVRTC or CUDA_PATH"; return; }

    std::string missing;
#define TACH_LOAD(h, f) a.f = (decltype(a.f))dlsym(h, TACH_STR(f)); if (!a.f) missing += std::string(missing.empty() ? "" : ", ") + TACH_STR(f);
#define TACH_LOAD_CU(f) TACH_LOAD(cu, f)
#define TACH_LOAD_NV(f) TACH_LOAD(nv, f)
    TACH_CU_FUNCS(TACH_LOAD_CU)
    TACH_NVRTC_FUNCS(TACH_LOAD_NV)
    if (!missing.empty()) { a.error = "CUDA libraries are too old (missing " + missing + ")"; return; }

    // NVRTC may search for its builtins by soname rather than beside an explicitly
    // dlopened compiler. Preload the matching library so relocated bundles work
    // without LD_LIBRARY_PATH (including CUDA 12.8 on Blackwell).
    int major = 0, minor = 0;
    if (a.nvrtcVersion(&major, &minor) != NVRTC_SUCCESS) { a.error = "Cannot query NVRTC version"; return; }
    std::string builtins = "libnvrtc-builtins.so." + std::to_string(major) + "." + std::to_string(minor);
    link_map* compiler = nullptr;
    std::vector<std::string> builtin_names;
    if (dlinfo(nv, RTLD_DI_LINKMAP, &compiler) == 0 && compiler && compiler->l_name) {
        std::string path = compiler->l_name;
        if (path.rfind('/') != std::string::npos) builtin_names.push_back(path.substr(0, path.rfind('/') + 1) + builtins);
    }
    builtin_names.push_back(builtins);
    tried.clear();
    if (!open_first(builtin_names, tried)) { a.error = "Matching NVRTC builtins library not found (tried " + tried + ")"; return; }

    // toolkit root of the loaded libnvrtc (…/lib64/libnvrtc.so -> …), used for NVRTC include paths
    link_map* lm = nullptr;
    if (dlinfo(nv, RTLD_DI_LINKMAP, &lm) == 0 && lm && lm->l_name) {
        std::string p = lm->l_name;
        for (int up = 0; up < 2 && p.rfind('/') != std::string::npos; up++) p = p.substr(0, p.rfind('/'));
        if (size_t t = p.rfind("/targets/"); t != std::string::npos) p = p.substr(0, t);
        if (is_dir(p + "/include")) a.cuda_dir = p;
    }
    if (a.cuda_dir.empty())
        for (auto& dir : cuda_dirs()) if (is_dir(dir + "/include")) { a.cuda_dir = dir; break; }
    a.ok = true;
}

}  // namespace

const CudaApi& cuda_api() {
    static CudaApi api;
    static std::once_flag once;
    std::call_once(once, [] { load(api); });
    return api;
}
