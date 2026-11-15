# Tachyon build. Host code is plain C++ (g++). CUDA kernels are compiled at runtime by NVRTC,
# so no nvcc / host-compiler compatibility is needed.
CXX      ?= g++
CUDA     ?= /usr/local/cuda
OPT      ?= -O3 -march=native -mtune=native
CXXFLAGS ?= $(OPT) -std=gnu++23 -g -Wall -Wno-unused-function -Wno-sign-compare -Wno-unused-variable -Wno-array-bounds -fno-math-errno -pthread
INCS     := -I$(CUDA)/include -Isrc
LDCUDA   := -L$(CUDA)/lib64 -L/usr/lib/wsl/lib -Wl,-rpath,$(CUDA)/lib64 -Wl,-rpath,/usr/lib/wsl/lib -lcuda -lnvrtc

BUILD ?= build
OUT ?= tachyon
CORE_SRCS := src/json.cpp src/http.cpp src/chat.cpp src/runner.cpp src/model_file.cpp src/safetensors.cpp src/quant.cpp src/convert.cpp src/tokenizer.cpp \
             src/cpu_backend.cpp src/cuda_backend.cpp src/sampler.cpp src/engine.cpp src/server.cpp src/main.cpp
CORE_OBJS := $(CORE_SRCS:src/%.cpp=$(BUILD)/%.o)

all: tachyon tachyon-client

$(BUILD)/%.o: src/%.cpp src/*.h src/kernels/*.cu data/calib.txt
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(INCS) -MMD -c $< -o $@

ifneq ($(OUT),tachyon)
tachyon: $(OUT)

$(OUT): $(CORE_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDCUDA)
else
tachyon: $(CORE_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDCUDA)
endif

tachyon-client: $(BUILD)/client.o $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILD)/file-probe: tests/file_probe.cpp $(BUILD)/model_file.o $(BUILD)/safetensors.o $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $(INCS) -o $@ $^

$(BUILD)/kernel-test: tests/kernel_test.cpp $(BUILD)/cpu_backend.o $(BUILD)/model_file.o $(BUILD)/json.o $(BUILD)/quant.o
	$(CXX) $(CXXFLAGS) $(INCS) -o $@ $^

test: $(BUILD)/file-probe $(BUILD)/kernel-test
	python3 tests/file_test.py $(BUILD)/file-probe
	$(BUILD)/kernel-test

$(BUILD)/inference-probe: tests/inference_probe.cpp $(BUILD)/cpu_backend.o $(BUILD)/cuda_backend.o $(BUILD)/model_file.o $(BUILD)/json.o $(BUILD)/quant.o
	$(CXX) $(CXXFLAGS) $(INCS) -o $@ $^ $(LDCUDA)

# Optional reference tests: needs torch, transformers, numpy and safetensors (no downloads).
PYTHON ?= python3
test-models: tachyon $(BUILD)/inference-probe
	$(PYTHON) tests/model_test.py ./tachyon $(BUILD)/inference-probe

clean:
	rm -rf $(BUILD) tachyon tachyon-client

.PHONY: all clean test test-models
