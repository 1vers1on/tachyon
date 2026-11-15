# usage: python tools/tool_test.py PORT MODEL... (needs `pip install openai anthropic`; run against `tachyon serve`)
import json, sys, time, openai, anthropic
PORT = int(sys.argv[1]); models = sys.argv[2:]
oc = openai.OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="x", timeout=600)
ac = anthropic.Anthropic(base_url=f"http://127.0.0.1:{PORT}", api_key="x", timeout=600)
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"},
          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
         {"type": "function", "function": {"name": "search_files", "description": "Search files for a regex",
          "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "max_results": {"type": "integer"}}, "required": ["pattern"]}}}]
ATOOLS = [{"name": t["function"]["name"], "description": t["function"]["description"], "input_schema": t["function"]["parameters"]} for t in TOOLS]
Q = "What's the weather in Paris right now? Use the get_weather tool."
for M in models:
    res = []
    t0 = time.time()
    try:
        r = oc.chat.completions.create(model=M, messages=[{"role": "user", "content": Q}], tools=TOOLS, max_tokens=2000, temperature=0)
        m = r.choices[0].message
        ok = bool(m.tool_calls) and m.tool_calls[0].function.name == "get_weather" and "paris" in m.tool_calls[0].function.arguments.lower()
        res.append(("call", ok))
        if not ok: print("   ", M, "no tool call:", repr((m.content or "")[:300]))
        if ok:
            msgs = [{"role": "user", "content": Q}, m.model_dump(exclude_none=True),
                    {"role": "tool", "tool_call_id": m.tool_calls[0].id, "content": json.dumps({"temp_c": 18, "sky": "cloudy"})}]
            r2 = oc.chat.completions.create(model=M, messages=msgs, tools=TOOLS, max_tokens=2000, temperature=0)
            c = r2.choices[0].message.content or ""
            res.append(("answer", "18" in c))
            res.append(("cached=%d/%d" % (r2.usage.prompt_tokens_details.cached_tokens, r2.usage.prompt_tokens), r2.usage.prompt_tokens_details.cached_tokens > r.usage.prompt_tokens // 2))
            if "18" not in c: print("   ", M, "answer:", repr(c[:300]), r2.choices[0].message.tool_calls)
        with ac.messages.stream(model=M, max_tokens=2000, tools=ATOOLS,
                                messages=[{"role": "user", "content": "Find files containing TODO. Use search_files."}]) as s:
            for _ in s: pass
            fm = s.get_final_message()
        tu = [b for b in fm.content if b.type == "tool_use"]
        ok = fm.stop_reason == "tool_use" and tu and tu[0].name == "search_files" and "TODO" in json.dumps(tu[0].input)
        res.append(("anth", ok))
        if not ok: print("   ", M, "anth:", fm.content)
    except Exception as ex:
        res.append(("exception " + str(ex)[:200], False))
    print("%-16s %5.1fs  " % (M, time.time() - t0) + "  ".join(("PASS " if ok else "FAIL ") + n for n, ok in res), flush=True)
