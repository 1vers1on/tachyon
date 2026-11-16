# Tachyon build. Host code is plain C++ (g++). CUDA kernels are compiled at runtime by NVRTC,
# so no nvcc / host-compiler compatibility is needed.
CXX      ?= g++
CUDA     ?= /usr/local/cuda
CPU      ?= scalar
CPU_scalar := -march=x86-64 -mtune=generic
CPU_avx2   := -march=x86-64 -mtune=generic -mavx2 -mfma -mf16c
CPU_avx512 := $(CPU_avx2) -mavx512f -mavx512bw -mavx512vl -mavx512vnni
CPU_native := -march=native -mtune=native
ifeq ($(filter $(CPU),scalar avx2 avx512 native),)
$(error CPU must be scalar, avx2, avx512 or native)
endif
OPT      ?= -O3 $(CPU_$(CPU))
CXXFLAGS ?= $(OPT) -std=gnu++20 -g -Wall -Wno-unused-function -Wno-sign-compare -Wno-unused-variable -Wno-array-bounds -fno-math-errno -pthread
LDFLAGS  ?=
INCS     := -I$(CUDA)/include -I$(BUILD)/generated -Isrc -DTACH_CUDA_DIR='"$(CUDA)"'
# libcuda and libnvrtc are loaded at runtime (src/cuda_api.cpp): only the CUDA headers are needed to build, and the
# binary runs on the CPU backend where no NVIDIA driver is installed
LDCUDA   := -ldl

BUILD ?= build/$(CPU)
OUT ?= tachyon
CLIENT_OUT ?= tachyon-client
PYTHON ?= python3
CORE_SRCS := src/json.cpp src/http.cpp src/https.cpp src/chat.cpp src/runner.cpp src/model_file.cpp src/safetensors.cpp src/quant.cpp src/convert.cpp src/tokenizer.cpp \
             src/cpu_backend.cpp src/cuda_api.cpp src/cuda_backend.cpp src/sampler.cpp src/engine.cpp src/server.cpp src/main.cpp
CORE_OBJS := $(CORE_SRCS:src/%.cpp=$(BUILD)/%.o)

all: tachyon $(CLIENT_OUT)

$(BUILD)/generated/kernels.inc: src/kernels/kernels.cu tools/embed.py
	$(PYTHON) tools/embed.py $< $@

$(BUILD)/generated/calib.inc: data/calib.txt tools/embed.py
	$(PYTHON) tools/embed.py $< $@

$(BUILD)/cuda_backend.o: $(BUILD)/generated/kernels.inc
$(BUILD)/runner.o: $(BUILD)/generated/calib.inc

$(BUILD)/config: FORCE
	@mkdir -p $(BUILD)
	@printf '%s\n' '$(CXX) $(CXXFLAGS) $(INCS) $(LDFLAGS)' > $@.tmp
	@cmp -s $@.tmp $@ || mv $@.tmp $@
	@rm -f $@.tmp

$(BUILD)/%.o: src/%.cpp src/*.h $(BUILD)/config
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(INCS) -MMD -c $< -o $@

ifneq ($(OUT),tachyon)
tachyon: $(OUT)

$(OUT): $(CORE_OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDCUDA)
else
tachyon: $(CORE_OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDCUDA)
endif

$(CLIENT_OUT): $(BUILD)/client.o $(BUILD)/json.o $(BUILD)/http.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

$(BUILD)/file-probe: tests/file_probe.cpp $(BUILD)/model_file.o $(BUILD)/safetensors.o $(BUILD)/json.o $(BUILD)/quant.o
	$(CXX) $(CXXFLAGS) $(INCS) -o $@ $^

$(BUILD)/kernel-test: tests/kernel_test.cpp $(BUILD)/cpu_backend.o $(BUILD)/model_file.o $(BUILD)/json.o $(BUILD)/quant.o
	$(CXX) $(CXXFLAGS) $(INCS) -o $@ $^

$(BUILD)/quant-test: tests/quant_test.cpp $(BUILD)/quant.o
	$(CXX) $(CXXFLAGS) $(INCS) -o $@ $^

$(BUILD)/text-test: tests/text_test.cpp $(BUILD)/tokenizer.o $(BUILD)/model_file.o $(BUILD)/json.o $(BUILD)/sampler.o
	$(CXX) $(CXXFLAGS) $(INCS) -o $@ $^

$(BUILD)/http-probe: tests/http_probe.cpp $(BUILD)/http.o $(BUILD)/https.o
	$(CXX) $(CXXFLAGS) $(INCS) -o $@ $^ $(LDCUDA)

test: tachyon $(CLIENT_OUT) $(BUILD)/file-probe $(BUILD)/kernel-test $(BUILD)/quant-test $(BUILD)/text-test $(BUILD)/http-probe
	python3 tests/file_test.py $(BUILD)/file-probe
	$(BUILD)/kernel-test
	$(BUILD)/quant-test
	python3 tests/server_test.py $(abspath $(OUT)) $(BUILD)/text-test
	python3 tests/settings_test.py $(abspath $(OUT))
	python3 tests/http_test.py $(BUILD)/http-probe $(abspath $(CLIENT_OUT))
	python3 tests/release_test.py $(abspath $(OUT)) $(abspath $(CLIENT_OUT))

$(BUILD)/inference-probe: tests/inference_probe.cpp $(BUILD)/cpu_backend.o $(BUILD)/cuda_backend.o $(BUILD)/cuda_api.o $(BUILD)/model_file.o $(BUILD)/json.o $(BUILD)/quant.o
	$(CXX) $(CXXFLAGS) $(INCS) -o $@ $^ $(LDCUDA)

$(BUILD)/cuda-sampling-test: tests/cuda_sampling_test.cpp $(BUILD)/cpu_backend.o $(BUILD)/cuda_backend.o $(BUILD)/cuda_api.o $(BUILD)/model_file.o $(BUILD)/json.o $(BUILD)/quant.o
	$(CXX) $(CXXFLAGS) $(INCS) -o $@ $^ $(LDCUDA)

$(BUILD)/bench-sampling: tools/bench_sampling.cpp $(BUILD)/cuda_api.o
	$(CXX) $(CXXFLAGS) $(INCS) -o $@ $^ $(LDCUDA)

test-cuda: $(BUILD)/cuda-sampling-test
	python3 -c 'import sys; from pathlib import Path; sys.path.insert(0, "tests"); from server_test import fixture; fixture(Path("$(BUILD)/sampling.tach")); fixture(Path("$(BUILD)/sampling-wide.tach"), vocab=1056)'
	$(BUILD)/cuda-sampling-test $(BUILD)/sampling.tach
	$(BUILD)/cuda-sampling-test $(BUILD)/sampling-wide.tach

# Optional reference tests: needs torch, transformers, numpy and safetensors (no downloads).
PYTHON ?= python3
test-models: tachyon $(BUILD)/inference-probe
	$(PYTHON) tests/model_test.py $(abspath $(OUT)) $(BUILD)/inference-probe

clean:
	rm -rf $(BUILD) tachyon tachyon-client

.PHONY: all clean test test-models test-cuda FORCE

-include $(CORE_OBJS:.o=.d) $(BUILD)/client.d
