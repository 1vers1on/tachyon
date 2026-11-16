#pragma once
#include "json.h"
#include <string>
std::string render_checkpoint_template(const std::string& source, const Json& variables);
