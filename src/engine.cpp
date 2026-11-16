#include "engine.h"
#include <algorithm>

Engine::~Engine() { stop(); }

void Engine::load(const std::string& path, const EngineOptions& o) {
    opt = o;
    mf.open(path);
    tok.load(mf);
    model_name = mf.config.name;
    gen_defaults = mf.header()["generation"];
    std::string dev = o.device;
    if (dev == "auto") {
        dev = "cpu";
        Backend* b = create_cuda_backend();
        if (b) { be.reset(b); dev = "cuda"; }
    } else if (dev == "cuda") {
        Backend* b = create_cuda_backend();
        if (!b) die("CUDA backend unavailable: %s", cuda_backend_error().c_str());
        be.reset(b);
    }
    if (dev == "cpu") be.reset(create_cpu_backend());
    if (opt.backend.ctx > mf.config.max_ctx) opt.backend.ctx = mf.config.max_ctx;
    if (opt.prefill_chunk <= 0) {
        bool moe = false;
        for (auto& l : mf.config.layers) moe |= l.n_experts > 0;
        // experts streamed over PCIe once per prefill forward: bigger chunks amortize the transfer
        opt.prefill_chunk = moe && dev == "cuda" && mf.file_size > (10ull << 30) ? 2048 : 512;
    }
    opt.backend.max_batch = std::max(opt.backend.max_batch, opt.prefill_chunk);
    double t0 = now_sec();
    if (!opt.draft_model.empty()) {
        // leave room for the draft model's weights and KV cache
        dmf.open(opt.draft_model);
        const ModelConfig& dc = dmf.config;
        size_t per_tok = (size_t)dc.n_layers * dc.n_kv_heads * dc.head_dim * 2 * (opt.backend.kv_int8 ? 1 : 2);
        opt.backend.reserve_bytes = dmf.file_size + (256ull << 20);
        opt.backend.extra_kv_per_tok = per_tok;
    }
    // stateful models can speculate when the backend keeps per-row state snapshots (CUDA)
    opt.backend.state_rollback = mf.config.stateful && dev == "cuda" && (opt.spec_draft != 0 || !opt.draft_model.empty());
    // and keep a per-slot state backup at the end of each conversation prefix (multi-turn / agent loops)
    opt.backend.state_backup = mf.config.stateful && dev == "cuda";
    be->init(mf, opt.backend);
    ckpt_ = mf.config.stateful && be->supports_state_backup() && !getenv("TACH_NO_CKPT");
    opt.backend.state_rollback = false;
    opt.backend.state_backup = false;
    opt.backend.min_ctx = 0;
    opt.backend.reserve_bytes = 0;
    opt.backend.extra_kv_per_tok = 0;
    opt.backend.ctx = be->opt.ctx;          // the backend may have shrunk the context to fit memory
    opt.backend.n_slots = be->opt.n_slots;
    if (!opt.draft_model.empty()) {
        if (dmf.config.vocab != mf.config.vocab) die("draft model vocab (%d) != target vocab (%d)", dmf.config.vocab, mf.config.vocab);
        dbe.reset(dev == "cuda" ? create_cuda_backend() : create_cpu_backend());
        BackendOptions dbo = opt.backend;
        dbo.vocab_limit = opt.draft_vocab;  // drafts only need likely tokens; verification stays exact
        dbo.state_rollback = dmf.config.stateful && dev == "cuda";   // per-slot backup to undo rejected drafts
        dbe->init(dmf, dbo);
        if (dbe->opt.ctx < opt.backend.ctx) opt.backend.ctx = dbe->opt.ctx;
        logi("draft model %s loaded for speculative decoding (k=%d)", dmf.config.name.c_str(), opt.draft_k);
    }
    logi("model %s loaded on %s in %.2fs (%d layers, hidden %d, vocab %d, %s)", model_name.c_str(), be->name(),
         now_sec() - t0, mf.config.n_layers, mf.config.hidden, mf.config.vocab, qtype_name(mf.config.wtype));
    if (opt.spec_draft < 0) opt.spec_draft = std::string(be->name()) == "cuda" || dbe ? 4 : 0;
    if (mf.config.stateful && !be->supports_state_rollback()) {
        // recurrent state (short conv / delta net) cannot be rolled back after rejected drafts on this backend
        if (opt.spec_draft > 0 || dbe) logi("model has recurrent state: speculative decoding disabled");
        opt.spec_draft = 0;
        dbe.reset();
    }
    if (dbe && dmf.config.stateful && !dbe->supports_state_rollback()) {
        logi("draft model has recurrent state but its backend cannot roll it back: draft model disabled");
        dbe.reset();
    }
    slots_.resize(opt.backend.n_slots);
    // sliding-window layers keep a ring of window + max_batch positions: a cached prefix stays usable only while
    // the tokens cached after it have not wrapped around into the window before it
    for (auto& l : mf.config.layers)
        if (l.is_attn() && l.window > 0) {
            const int cap = std::min(opt.backend.ctx, l.window + be->opt.max_batch);
            if (cap < opt.backend.ctx) ring_slack_ = std::min(ring_slack_, cap - l.window);
        }
    logits_buf_.resize(be->vocab());
    for (auto& l : mf.config.layers) if (l.n_experts) spec_min_accept_ = 0.2f;
}

Engine::Stats Engine::stats() {
    Stats st;
    st.slots = (int)slots_.size();
    st.active = n_active_.load();
    std::lock_guard<std::mutex> lk(mu_);
    st.queued = (int)queue_.size();
    return st;
}

void Engine::start() {
    running_ = true;
    th_ = std::thread([this] { loop(); });
}

void Engine::stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!running_) return;
        running_ = false;
    }
    cv_.notify_all();
    if (th_.joinable()) th_.join();
}

void Engine::submit(std::shared_ptr<GenRequest> r) {
    r->t_submit = now_sec();
    {
        std::lock_guard<std::mutex> lk(mu_);
        queue_.push_back(r);
    }
    cv_.notify_all();
}

void Engine::run_sync(std::shared_ptr<GenRequest> r) {
    r->t_submit = now_sec();
    {
        std::lock_guard<std::mutex> lk(mu_);
        queue_.push_back(r);
    }
    while (!r->done) step();
}

void Engine::loop() {
    while (true) {
        {
            std::unique_lock<std::mutex> lk(mu_);
            if (!running_) break;
        }
        if (!step()) {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait_for(lk, std::chrono::milliseconds(50), [&] { return !queue_.empty() || !running_; });
        }
    }
}

void Engine::admit() {
    std::lock_guard<std::mutex> lk(mu_);
    while (!queue_.empty()) {
        // pick the free slot with the longest common prefix (prefix cache), else least recently used
        int best = -1, best_lcp = -1;
        auto& r = queue_.front();
        for (int i = 0; i < (int)slots_.size(); i++) {
            if (slots_[i].req) continue;
            const auto& c = slots_[i].cache;
            int lcp = 0;
            while (lcp < (int)c.size() && lcp < (int)r->prompt.size() && c[lcp] == r->prompt[lcp]) lcp++;
            if (lcp > best_lcp || (lcp == best_lcp && slots_[i].last_used < slots_[best].last_used)) { best = i; best_lcp = lcp; }
        }
        if (best < 0) break;
        Slot& s = slots_[best];
        s.req = r;
        n_active_++;
        queue_.pop_front();
        if ((int)r->prompt.size() + 1 > opt.backend.ctx) {
            r->error = "prompt (" + std::to_string(r->prompt.size()) + " tokens) exceeds context size " + std::to_string(opt.backend.ctx);
            finish(s, "error");
            continue;
        }
        if (r->prompt.empty()) r->prompt.push_back(tok.bos() >= 0 ? tok.bos() : 0);
        // keep at least one prompt token to evaluate so we get logits
        int keep = std::min(best_lcp, (int)r->prompt.size() - 1);
        // ring buffers (sliding window): the tokens cached after the prefix may have overwritten the window before it
        if ((int)s.cache.size() - keep > ring_slack_) keep = 0;
        // recurrent state describes the whole cached sequence: reuse all of it, resume from the prompt checkpoint
        // (state saved after the conversation prefix of the previous request), or start over
        if (mf.config.stateful && keep != (int)s.cache.size()) {
            if (ckpt_ && s.ckpt_len > 0 && keep >= s.ckpt_len) { be->state_restore(best); keep = s.ckpt_len; }
            else keep = 0;
        }
        if (keep < s.ckpt_len) s.ckpt_len = 0;
        s.cache.resize(keep);
        if ((int)s.dcache.size() > keep) s.dcache.resize(keep);
        {
            int dl = 0;
            while (dl < (int)s.dcache.size() && s.dcache[dl] == r->prompt[dl]) dl++;
            s.dcache.resize(dl);
        }
        s.dk = opt.draft_k;
        s.n_prompt_done = keep;
        s.hist = r->prompt;
        s.generating = false;
        s.last = -1;
        s.sampler.init(r->sp);
        s.dec = std::make_unique<StreamDecoder>(&tok);
        s.pending_text.clear();
        s.spec_k = opt.spec_draft;
        s.acc_ema = 0.7f; s.spec_pause = 0;
        s.spec_miss = 0;
        r->n_prompt = (int)r->prompt.size();
        r->n_cached = keep;
        logd("admit slot %d: prompt %d tokens, %d cached", best, (int)r->prompt.size(), keep);
    }
}

void Engine::finish(Slot& s, const std::string& reason) {
    auto r = s.req;
    // flush any held-back text
    if (reason != "error") {
        std::string rest = s.pending_text + (s.dec ? s.dec->flush() : "");
        if (reason != "stop_string" && !rest.empty()) {
            r->text += rest;
            if (r->on_text) r->on_text(rest);
        }
    }
    r->finish_reason = reason == "stop_string" ? "stop" : reason;
    r->t_done = now_sec();
    if (r->t_first == 0) r->t_first = r->t_done;
    s.req.reset();
    n_active_--;
    s.generating = false;
    s.last_used = now_sec();
    {
        std::lock_guard<std::mutex> lk(r->mu);
        r->done = true;
    }
    r->cv.notify_all();
}

// Append a sampled token. Handles EOS, stop strings, max tokens, streaming.
bool Engine::emit_token(Slot& s, int t) {
    auto& r = *s.req;
    if (r.t_first == 0) r.t_first = now_sec();
    s.sampler.accept(t);
    s.hist.push_back(t);
    r.n_gen++;
    if (!r.ignore_eos && tok.is_eos(t)) { finish(s, "stop"); return false; }
    std::string piece = s.dec->push(t);
    s.pending_text += piece;
    // stop strings
    size_t hold = 0;
    for (auto& st : r.stop) {
        if (st.empty()) continue;
        size_t p = s.pending_text.find(st);
        if (p != std::string::npos) {
            std::string out = s.pending_text.substr(0, p);
            r.stop_hit = st;
            r.text += out;
            if (r.on_text && !out.empty()) r.on_text(out);
            s.pending_text.clear();
            finish(s, "stop_string");
            return false;
        }
        // hold back a suffix that could be the start of a stop string
        for (size_t k = std::min(st.size() - 1, s.pending_text.size()); k > 0; k--)
            if (s.pending_text.compare(s.pending_text.size() - k, k, st, 0, k) == 0) { hold = std::max(hold, k); break; }
    }
    if (s.pending_text.size() > hold) {
        std::string out = s.pending_text.substr(0, s.pending_text.size() - hold);
        s.pending_text.erase(0, out.size());
        r.text += out;
        if (r.on_text && !r.on_text(out)) { finish(s, "cancelled"); return false; }
    }
    if (r.cancel) { finish(s, "cancelled"); return false; }
    if (r.n_gen >= r.max_tokens) { finish(s, "length"); return false; }
    if ((int)s.hist.size() + 1 >= opt.backend.ctx) { finish(s, "length"); return false; }
    return true;
}

// Prompt-lookup ("n-gram") drafting: find the latest earlier occurrence of the trailing n-gram and
// propose the tokens that followed it.
std::vector<int> Engine::draft_ngram(const Slot& s, int k, int min_n) const {
    std::vector<int> d;
    const auto& h = s.hist;
    int L = (int)h.size();
    for (int n = 3; n >= min_n && d.empty(); n--) {
        if (L < n + 1) continue;
        const int* tail = h.data() + L - n;
        for (int j = L - n - 1; j >= 0 && j >= L - 8192; j--) {
            if (h[j] == tail[0] && std::equal(tail, tail + n, h.data() + j)) {
                for (int i = j + n; i < L && (int)d.size() < k; i++) d.push_back(h[i]);
                break;
            }
        }
        if (n == 1) break;
    }
    return d;
}

// Draft-model speculation: bring the draft model's KV for each slot up to date with the target context
// (cache + last token), then greedily generate k draft tokens, batched across slots.
void Engine::draft_with_model(const std::vector<std::pair<int, int>>& need, std::vector<std::vector<int>>& out) {
    out.assign(need.size(), {});
    std::vector<int> cur(need.size(), -1);
    // catch-up (per slot): feed every target-context token the draft has not seen
    for (size_t i = 0; i < need.size(); i++) {
        Slot& s = slots_[need[i].first];
        std::vector<int> ctx = s.cache;
        ctx.push_back(s.last);
        int lcp = 0;
        while (lcp < (int)s.dcache.size() && lcp < (int)ctx.size() && s.dcache[lcp] == ctx[lcp]) lcp++;
        if (dmf.config.stateful && lcp < (int)s.dcache.size()) {
            // the draft's recurrent state includes rejected tokens: go back to the backup taken after the last catch-up
            if (s.dsaved > 0 && lcp >= s.dsaved) { dbe->state_restore(need[i].first); lcp = s.dsaved; }
            else lcp = 0;   // position 0 resets the state
        }
        if (dmf.config.stateful && lcp >= (int)ctx.size()) lcp = 0;   // never feed a token twice into the state
        lcp = std::min(lcp, (int)ctx.size() - 1);
        for (int p0 = lcp; p0 < (int)ctx.size(); p0 += opt.prefill_chunk) {
            int n = std::min(opt.prefill_chunk, (int)ctx.size() - p0);
            std::vector<BatchToken> b(n);
            for (int j = 0; j < n; j++) b[j] = {ctx[p0 + j], need[i].first, p0 + j, p0 + j == (int)ctx.size() - 1};
            dbe->forward(b);
        }
        dbe->argmax_all(&cur[i]);
        s.dcache = ctx;
        if (dmf.config.stateful) { dbe->state_save(need[i].first); s.dsaved = (int)ctx.size(); }
        out[i].push_back(cur[i]);
    }
    // autoregressive drafting, all slots batched
    int kmax = 0;
    for (auto& nk : need) kmax = std::max(kmax, nk.second);
    for (int step = 1; step < kmax; step++) {
        std::vector<BatchToken> b;
        std::vector<int> idx;
        for (size_t i = 0; i < need.size(); i++) {
            if ((int)out[i].size() >= need[i].second) continue;
            Slot& s = slots_[need[i].first];
            b.push_back({cur[i], need[i].first, (int)s.dcache.size(), true});
            idx.push_back((int)i);
        }
        if (b.empty()) break;
        dbe->forward(b);
        std::vector<int> am(b.size());
        dbe->argmax_all(am.data());
        for (size_t j = 0; j < idx.size(); j++) {
            int i = idx[j];
            slots_[need[i].first].dcache.push_back(cur[i]);
            cur[i] = am[j];
            out[i].push_back(am[j]);
        }
    }
}

bool Engine::step() {
    admit();
    // ---- prefill (one slot, one chunk) ----
    for (int si = 0; si < (int)slots_.size(); si++) {
        Slot& s = slots_[si];
        if (!s.req || s.generating) continue;
        auto& r = *s.req;
        if (r.cancel) { finish(s, "cancelled"); continue; }
        int n_total = (int)r.prompt.size();
        int n = std::min(opt.prefill_chunk, n_total - s.n_prompt_done);
        const bool ckpt_here = ckpt_ && r.checkpoint > s.n_prompt_done && r.checkpoint < n_total;
        if (ckpt_here) n = std::min(n, r.checkpoint - s.n_prompt_done);   // end a chunk exactly at the checkpoint
        std::vector<BatchToken> b(n);
        for (int i = 0; i < n; i++) {
            int p = s.n_prompt_done + i;
            b[i] = {r.prompt[p], si, p, p == n_total - 1};
        }
        be->forward(b);
        for (int i = 0; i < n; i++) s.cache.push_back(r.prompt[s.n_prompt_done + i]);
        s.n_prompt_done += n;
        if (ckpt_here && s.n_prompt_done == r.checkpoint) { be->state_save(si); s.ckpt_len = r.checkpoint; }
        if (s.n_prompt_done == n_total) {
            int t;
            if (s.sampler.params.greedy() && !s.sampler.params.has_penalties()) be->argmax_all(&t);
            else {
                be->get_logits(0, logits_buf_.data());
                s.sampler.apply_penalties(logits_buf_.data(), be->vocab());
                t = s.sampler.sample(logits_buf_.data(), be->vocab());
            }
            r.t_prefill_done = now_sec();  // after the sync above, so it measures real GPU time
            s.generating = true;
            s.last = t;
            emit_token(s, t);
        }
        return true;
    }
    // ---- decode (all generating slots, with n-gram speculative drafts) ----
    std::vector<BatchToken> b;
    struct Rows { int slot; int first; std::vector<int> draft; };
    std::vector<Rows> rows;
    int n_active = 0;
    for (auto& s : slots_) if (s.req && s.generating) n_active++;
    if (!n_active) return false;
    std::vector<std::pair<int, int>> need_model;  // (slot, k) for draft-model speculation
    std::vector<std::vector<int>> ngram_drafts(slots_.size());
    for (int si = 0; si < (int)slots_.size(); si++) {
        Slot& s = slots_[si];
        if (!s.req || !s.generating) continue;
        int pos = (int)s.cache.size();
        // keep the whole decode batch within one 8-token GEMV pass (weights are streamed once per pass)
        int cap = std::max(0, 8 / n_active - 1);
        cap = std::min(cap, opt.backend.ctx - pos - 2);
        cap = std::min(cap, s.req->max_tokens - s.req->n_gen - 1);
        int k = std::min(s.spec_k, cap);
        if (s.spec_pause > 0) { s.spec_pause--; k = 0; }
        if (k > 0) ngram_drafts[si] = draft_ngram(s, k, dbe ? 2 : 1);
        // the draft model only pays off when few streams are active (batching already fills the GPU otherwise)
        if (ngram_drafts[si].empty() && dbe && n_active <= 2 && std::min(s.dk, cap) > 0) need_model.push_back({si, std::min(s.dk, cap)});
    }
    std::vector<std::vector<int>> model_drafts;
    if (!need_model.empty()) draft_with_model(need_model, model_drafts);
    for (size_t i = 0; i < need_model.size(); i++) ngram_drafts[need_model[i].first] = model_drafts[i];
    std::vector<bool> from_model(slots_.size(), false);
    for (auto& nk : need_model) from_model[nk.first] = true;
    for (int si = 0; si < (int)slots_.size(); si++) {
        Slot& s = slots_[si];
        if (!s.req || !s.generating) continue;
        int pos = (int)s.cache.size();
        Rows rr{si, (int)b.size(), std::move(ngram_drafts[si])};
        b.push_back({s.last, si, pos, true});
        for (size_t i = 0; i < rr.draft.size(); i++) b.push_back({rr.draft[i], si, pos + 1 + (int)i, true});
        rows.push_back(std::move(rr));
    }
    be->forward(b);
    bool all_greedy = true;
    for (auto& rr : rows) {
        auto& sp = slots_[rr.slot].sampler.params;
        if (!sp.greedy() || sp.has_penalties()) all_greedy = false;
    }
    std::vector<int> am;
    if (all_greedy) { am.resize(b.size()); be->argmax_all(am.data()); }
    // non-greedy without penalties: let the backend pre-filter candidates (avoids copying full logits rows)
    std::vector<std::vector<std::pair<float, int>>> cands;
    bool have_cands = false;
    if (!all_greedy) {
        bool any_pen = false;
        float maxT = 0;
        for (auto& rr : rows) {
            auto& sp = slots_[rr.slot].sampler.params;
            any_pen |= sp.has_penalties();
            maxT = std::max(maxT, sp.greedy() ? 0.f : sp.temperature);
        }
        if (!any_pen) have_cands = be->candidates((int)b.size(), 20.f * std::max(maxT, 1e-3f), 1024, cands);
    }
    for (auto& rr : rows) {
        Slot& s = slots_[rr.slot];
        auto req = s.req;
        int nd = (int)rr.draft.size();
        req->n_draft += nd;
        // the last token is now in the KV cache
        s.cache.push_back(s.last);
        int accepted = 0;
        for (int i = 0; i <= nd; i++) {
            int t;
            int row = rr.first + i;
            if (all_greedy) t = am[row];
            else if (have_cands) {
                auto& c = cands[row];
                if (s.sampler.params.greedy()) {
                    t = std::max_element(c.begin(), c.end())->second;
                } else {
                    t = i < nd ? s.sampler.sample_cands_with_draft(c, rr.draft[i]) : s.sampler.sample_cands(c);
                }
            } else {
                be->get_logits(row, logits_buf_.data());
                s.sampler.apply_penalties(logits_buf_.data(), be->vocab());
                t = i < nd ? s.sampler.sample_with_draft(logits_buf_.data(), be->vocab(), rr.draft[i])
                           : s.sampler.sample(logits_buf_.data(), be->vocab());
            }
            bool ok = emit_token(s, t);
            if (!ok) break;
            if (i < nd && t == rr.draft[i]) {
                accepted++;
                s.cache.push_back(t);  // draft token's KV (computed in this step) is valid
                continue;
            }
            s.last = t;
            break;
        }
        req->n_draft_accepted += accepted;
        // recurrent state: undo the rows of rejected drafts (state after the last kept row = row first + accepted)
        if (mf.config.stateful && accepted < nd) be->state_restore_row(rr.slot, rr.first + accepted);
        // adapt speculation depth
        if (nd > 0 && from_model[rr.slot]) {
            if (accepted == nd) s.dk = std::min(s.dk + 1, 7);
            else if (accepted * 2 < nd) s.dk = std::max(s.dk - 1, 1);
        } else if (nd > 0) {
            s.acc_ema = 0.8f * s.acc_ema + 0.2f * accepted / nd;
            if (s.acc_ema < spec_min_accept_) { s.spec_pause = 16; s.acc_ema = spec_min_accept_; }
            if (accepted == 0) { if (++s.spec_miss >= 4 && s.spec_k > 1) { s.spec_k--; s.spec_miss = 0; } }
            else { s.spec_miss = 0; if (accepted == nd && s.spec_k < 8 && s.spec_k < opt.spec_draft * 2) s.spec_k++; }
        }
        if (s.req && accepted == nd && nd > 0 && s.last == rr.draft[nd - 1]) {
            // all drafts accepted: last emitted token is the extra sampled one (already set via s.last)
        }
    }
    return true;
}
