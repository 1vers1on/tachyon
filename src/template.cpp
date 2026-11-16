#include "template.h"
#include "vendor/jinja/lexer.h"
#include "vendor/jinja/parser.h"
#include "vendor/jinja/runtime.h"
#include <map>
#include <mutex>
#include <cmath>

namespace {
jinja::value value(const Json& json, int depth=0) {
    using namespace jinja;
    if(depth>JINJA_RECURSIVE_LIMIT)throw std::runtime_error("chat template input nesting limit exceeded");
    if(json.is_null())return mk_val<value_none>();
    if(json.is_bool())return mk_val<value_bool>(json.b);
    if(json.is_str())return mk_val<value_string>(json.str);
    if(json.is_num()){
        if(std::isfinite(json.num)&&json.num>=INT64_MIN&&json.num<9223372036854775808.0&&std::trunc(json.num)==json.num)return mk_val<value_int>((int64_t)json.num);
        return mk_val<value_float>(json.num);
    }
    if(json.is_arr()){auto out=mk_val<value_array>();for(auto& item:json.arr)out->push_back(value(item,depth+1));return out;}
    auto out=mk_val<value_object>();for(auto& item:json.obj)out->insert(item.first,value(item.second,depth+1));return out;
}
}
std::string render_checkpoint_template(const std::string& source,const Json& variables) {
    static std::mutex mutex;
    static std::map<std::string,std::shared_ptr<jinja::program>> cache;
    std::shared_ptr<jinja::program> program;
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto& found=cache[source];
        if(!found){jinja::lexer lexer;found=std::make_shared<jinja::program>(jinja::parse_from_tokens(lexer.tokenize(source)));}
        program=found;
    }
    jinja::context context(source);
    for(auto& item:variables.obj)context.set_val(item.first,value(item.second));
    jinja::runtime runtime(context);
    return jinja::runtime::gather_string_parts(runtime.execute(*program))->val_str.str();
}
