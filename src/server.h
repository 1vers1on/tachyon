#pragma once
#include "engine.h"

struct ServerOptions {
    std::string host = "127.0.0.1";
    int port = 28657;
    std::string api_key;   // require "Authorization: Bearer KEY" or "x-api-key: KEY" (empty = open)
    std::string alias;     // model id reported by /v1/models (default: the model's name)
};

int server_main(Engine& e, const ServerOptions& o);
