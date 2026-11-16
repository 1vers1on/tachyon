// Baseline x86-64 entry point for relocatable releases. GCC's feature checks include OS XSAVE support.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cerrno>
#include <climits>
#include <unistd.h>

int main(int argc, char** argv) {
    (void)argc;
    __builtin_cpu_init();
    const bool avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    const bool avx512 = avx2 && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
                       __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vnni");
    std::string cpu = avx512 ? "avx512" : avx2 ? "avx2" : "scalar";
    if (const char* e = getenv("TACHYON_CPU"); e && *e && strcmp(e, "auto")) cpu = e;
    if (cpu != "scalar" && cpu != "avx2" && cpu != "avx512") {
        fprintf(stderr, "tachyon: TACHYON_CPU must be auto, scalar, avx2 or avx512\n");
        return 1;
    }
    if ((cpu == "avx2" && !avx2) || (cpu == "avx512" && !avx512)) {
        fprintf(stderr, "tachyon: this CPU/OS does not support TACHYON_CPU=%s\n", cpu.c_str());
        return 1;
    }
    char executable[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
    if (n < 0 || n == sizeof(executable) - 1) {
        fprintf(stderr, "tachyon: cannot resolve /proc/self/exe\n");
        return 1;
    }
    executable[n] = 0;
    std::string path = executable;
    path = path.substr(0, path.rfind('/')) + "/../libexec/tachyon-" + cpu;
    execv(path.c_str(), argv);
    fprintf(stderr, "tachyon: cannot execute %s: %s\n", path.c_str(), strerror(errno));
    return 1;
}
