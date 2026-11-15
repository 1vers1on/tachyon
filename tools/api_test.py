# usage: python tools/api_test.py PORT [MODEL]   (needs `pip install openai anthropic`)
"""End-to-end API checks against a running tachyon server, using the official OpenAI and Anthropic SDKs."""
import json, sys, time, socket, urllib.request
import openai, anthropic

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8091
MODEL = sys.argv[2] if len(sys.argv) > 2 else "test"
BASE = f"http://127.0.0.1:{PORT}"
oc = openai.OpenAI(base_url=BASE + "/v1", api_key="x", timeout=300)
ac = anthropic.Anthropic(base_url=BASE, api_key="x", timeout=300)
failures = []

def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -- {info}"))
    if not cond: failures.append(name)

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"},
                          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
ATOOL = {"name": "get_weather", "description": "Get the current weather for a city",
         "input_schema": WEATHER["function"]["parameters"]}
Q = "What's the weather in Paris right now? Use the tool."

# ---------------- OpenAI
r = oc.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Say hello in one short sentence."}], max_tokens=2000)
check("oai plain", r.choices[0].message.content and r.choices[0].finish_reason in ("stop", "length"), r)

r = oc.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": Q}], tools=[WEATHER], max_tokens=1500)
m = r.choices[0].message
ok = bool(m.tool_calls) and m.tool_calls[0].function.name == "get_weather" and "paris" in json.loads(m.tool_calls[0].function.arguments).get("city", "").lower()
check("oai tool call", ok and r.choices[0].finish_reason == "tool_calls", m)
if ok:
    msgs = [{"role": "user", "content": Q}, m.model_dump(exclude_none=True),
            {"role": "tool", "tool_call_id": m.tool_calls[0].id, "content": json.dumps({"temp_c": 18, "sky": "cloudy"})}]
    r2 = oc.chat.completions.create(model=MODEL, messages=msgs, tools=[WEATHER], max_tokens=1500)
    c = r2.choices[0].message.content or ""
    check("oai tool result -> answer", "18" in c and not r2.choices[0].message.tool_calls, r2.choices[0].message)

# streaming tool call
acc = {}
fin = None
content = ""
for ch in oc.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": Q}], tools=[WEATHER], max_tokens=1500, stream=True,
                                     stream_options={"include_usage": True}):
    if not ch.choices: continue
    d = ch.choices[0].delta
    content += d.content or ""
    for tc in d.tool_calls or []:
        e = acc.setdefault(tc.index, {"name": "", "args": "", "id": None})
        if tc.id: e["id"] = tc.id
        if tc.function and tc.function.name: e["name"] += tc.function.name
        if tc.function and tc.function.arguments: e["args"] += tc.function.arguments
    fin = ch.choices[0].finish_reason or fin
check("oai stream tool call", acc and acc[0]["name"] == "get_weather" and json.loads(acc[0]["args"]) and fin == "tool_calls" and "<tool_call>" not in content, (acc, fin, content))

r = oc.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Tell me a joke."}], tools=[WEATHER],
                               tool_choice={"type": "function", "function": {"name": "get_weather"}}, max_tokens=300)
m = r.choices[0].message
check("oai forced tool_choice", m.tool_calls and m.tool_calls[0].function.name == "get_weather" and isinstance(json.loads(m.tool_calls[0].function.arguments), dict), m)

r = oc.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Give a JSON object with keys name and age for a fictional person."}],
                               response_format={"type": "json_object"}, max_tokens=800)
try:
    txt = r.choices[0].message.content.strip()
    if txt.startswith("```"): txt = txt.strip("`").removeprefix("json")
    json.loads(txt); ok = True
except Exception as ex:
    ok = False
check("oai json_object", ok, r.choices[0].message.content)

r = oc.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Count from 1 to 20 separated by spaces."}], stop=["7"], max_tokens=300,
                               extra_body={"chat_template_kwargs": {"enable_thinking": False}})
check("oai stop", "7" not in (r.choices[0].message.content or "") and r.choices[0].finish_reason == "stop", r.choices[0].message.content)

# ---------------- Anthropic
r = ac.messages.create(model=MODEL, max_tokens=1500, messages=[{"role": "user", "content": "Say hi in five words."}])
check("anth plain", r.stop_reason == "end_turn" and any(b.type == "text" for b in r.content) and r.usage.output_tokens > 0, r)

r = ac.messages.create(model=MODEL, max_tokens=1500, tools=[ATOOL], messages=[{"role": "user", "content": Q}])
tu = [b for b in r.content if b.type == "tool_use"]
check("anth tool_use", r.stop_reason == "tool_use" and tu and tu[0].name == "get_weather" and "paris" in str(tu[0].input).lower(), r)
if tu:
    msgs = [{"role": "user", "content": Q}, {"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in r.content]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu[0].id, "content": "18 degrees C, cloudy"}]}]
    r2 = ac.messages.create(model=MODEL, max_tokens=1500, tools=[ATOOL], messages=msgs)
    txt = "".join(b.text for b in r2.content if b.type == "text")
    check("anth tool_result -> answer", r2.stop_reason == "end_turn" and "18" in txt, r2)

with ac.messages.stream(model=MODEL, max_tokens=1500, tools=[ATOOL], messages=[{"role": "user", "content": Q}]) as s:
    for _ in s: pass
    fm = s.get_final_message()
tu = [b for b in fm.content if b.type == "tool_use"]
check("anth stream tool_use", fm.stop_reason == "tool_use" and tu and isinstance(tu[0].input, dict) and tu[0].input.get("city"), fm)

with ac.messages.stream(model=MODEL, max_tokens=2000, thinking={"type": "enabled", "budget_tokens": 1024},
                        messages=[{"role": "user", "content": "What is 17*23? Answer with the number."}]) as s:
    for _ in s: pass
    fm = s.get_final_message()
types = [b.type for b in fm.content]
txt = "".join(b.text for b in fm.content if b.type == "text")
check("anth stream thinking", "391" in txt and fm.stop_reason == "end_turn", (types, txt))
print("   blocks:", types)

r = ac.messages.create(model=MODEL, max_tokens=300, stop_sequences=["7"], thinking={"type": "disabled"},
                       messages=[{"role": "user", "content": "Count from 1 to 20 separated by spaces."}])
check("anth stop_sequence", r.stop_reason == "stop_sequence" and r.stop_sequence == "7", r)

r = ac.messages.create(model=MODEL, max_tokens=500, tools=[ATOOL], tool_choice={"type": "tool", "name": "get_weather"},
                       messages=[{"role": "user", "content": "hello"}])
check("anth forced tool", r.stop_reason == "tool_use" and r.content[-1].type == "tool_use", r)

n = ac.messages.count_tokens(model=MODEL, tools=[ATOOL], system="You are terse.", messages=[{"role": "user", "content": Q}])
check("anth count_tokens", n.input_tokens > 50, n)

ms = list(ac.models.list())
check("anth models.list", len(ms) >= 1 and any(m.id == MODEL for m in ms) or MODEL == "test", ms)
check("oai models.list", len(list(oc.models.list())) >= 1)

if len(sys.argv) > 3 and sys.argv[3] == "disconnect-only": pass
# ---------------- disconnect cancels generation (streaming and non-streaming)
for stream in (True, False):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "Write a 3000 word essay on rivers."}], "max_tokens": 4000,
                       "stream": stream, "ignore_eos": True})
    sk = socket.create_connection(("127.0.0.1", PORT))
    sk.sendall(f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n{body}".encode())
    time.sleep(1.5)
    sk.close()
    time.sleep(1.0)
    h = json.load(urllib.request.urlopen(BASE + "/health"))
    if h.get("manager"):
        ps = json.load(urllib.request.urlopen(BASE + "/api/ps"))["models"]
        port = ps[0]["port"]
        h = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health"))
    check(f"disconnect cancels ({'stream' if stream else 'non-stream'})", h["active"] == 0, h)

print("\n%d failures: %s" % (len(failures), failures))
