#include "template.h"
#include "common.h"
int g_log_level=0;
int main(int argc,char** argv){if(argc!=3)return 2;try{auto result=render_checkpoint_template(read_file(argv[1]),Json::parse(read_file(argv[2])));fwrite(result.data(),1,result.size(),stdout);return 0;}catch(const std::exception& ex){fprintf(stderr,"%s\n",ex.what());return 1;}}
