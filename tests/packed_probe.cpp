#include "quant.h"
#include <fstream>
int g_log_level=0;
int main(int argc,char** argv){if(argc!=4)return 2;auto qt=qtype_from_name(argv[1]);int k=atoi(argv[2]);std::string bytes=read_file(argv[3]);if(bytes.size()!=qtype_data_bytes(qt,1,k))return 2;std::vector<float> out(k);dequantize_row(qt,(const uint8_t*)bytes.data(),nullptr,k,out.data());return fwrite(out.data(),4,k,stdout)==(size_t)k?0:2;}
