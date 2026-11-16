// Full-vocabulary logits for externally supplied token IDs; no sampling or prompt-template differences.
#include "backend.h"
#include <memory>
int g_log_level=0;
int main(int argc,char** argv){
 if(argc<7||argc>8)return 2;ModelFile mf;mf.open(argv[1]);std::string raw=read_file(argv[2]);if(raw.size()%4)return 2;
 std::vector<int> ids(raw.size()/4);memcpy(ids.data(),raw.data(),raw.size());int chunk=atoi(argv[4]);
 auto backend=std::unique_ptr<Backend>(std::string(argv[6])=="cuda"?create_cuda_backend():create_cpu_backend());
 BackendOptions opt;opt.ctx=atoi(argv[5]);opt.max_batch=std::max(chunk,32);opt.n_slots=1;opt.threads=8;opt.use_graphs=false;opt.state_rollback=getenv("PROBE_ROLLBACK")!=nullptr;opt.kv_int8=argc==8&&std::string(argv[7])=="q8";backend->init(mf,opt);
 FILE* f=fopen(argv[3],"wb");if(!f)return 2;std::vector<float> logits(mf.config.vocab);
 for(int p=0;p<(int)ids.size();p+=chunk){std::vector<BatchToken> batch;for(int i=p;i<std::min(p+chunk,(int)ids.size());i++)batch.push_back({ids[i],0,i,true});backend->forward(batch);for(int i=0;i<(int)batch.size();i++){backend->get_logits(i,logits.data());if(fwrite(logits.data(),4,logits.size(),f)!=logits.size())return 2;}}
 return fclose(f)?2:0;
}
