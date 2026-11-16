#include "model_file.h"
#include "safetensors.h"
#include "quant.h"

int g_log_level = 0;

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    if (std::string(argv[1]) == "tach") {
        ModelFile m; m.open(argv[2]);
        // Reopening must discard tensors and mappings from the previous file.
        m.open(argv[2]);
    } else if (std::string(argv[1]) == "imatrix") {
        IMatrix::load(argv[2]);
    } else {
        SafeTensors s; s.open_dir(argv[2]);
        if (std::string(argv[1]) == "read") {
            float out;
            s.get("w")->to_float(&out, 4, 1);
        }
    }
}
