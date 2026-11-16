#pragma once
#include <string>
// Lossless import of explicitly supported GGUF architectures and block layouts.
int convert_gguf(const std::string& input, const std::string& output);
