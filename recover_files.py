import json
import os

transcript_path = os.path.expanduser('~/.gemini/antigravity-ide/brain/b06d7b61-940c-4af2-b799-8519cc9a0d48/.system_generated/logs/transcript_full.jsonl')

files_to_recover = [
    '/Users/aayushidhurandhar/InferSpore/src/models/llama/llama_params.h',
    '/Users/aayushidhurandhar/InferSpore/src/models/llama/llama.h',
    '/Users/aayushidhurandhar/InferSpore/src/models/llama/llama.cpp',
    '/Users/aayushidhurandhar/InferSpore/tools/export_tokenizer.py',
    '/Users/aayushidhurandhar/InferSpore/tools/validate_logits.py',
    '/Users/aayushidhurandhar/InferSpore/tests/test_correctness.cpp',
    '/Users/aayushidhurandhar/InferSpore/src/main.cpp',
    '/Users/aayushidhurandhar/InferSpore/web/server.py',
    '/Users/aayushidhurandhar/InferSpore/web/static/index.html',
    '/Users/aayushidhurandhar/InferSpore/web/static/style.css',
    '/Users/aayushidhurandhar/InferSpore/web/static/main.js',
]

file_contents = {}

with open(transcript_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        if 'tool_calls' in data:
            for call in data['tool_calls']:
                if call['name'] == 'write_to_file':
                    args = call['args']
                    if 'TargetFile' in args and 'CodeContent' in args:
                        target = args['TargetFile']
                        if target in files_to_recover:
                            file_contents[target] = args['CodeContent']

for path, content in file_contents.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
        
print("Recovered files:", list(file_contents.keys()))
