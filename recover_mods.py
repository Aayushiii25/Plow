import json
import os

transcript_path = os.path.expanduser('~/.gemini/antigravity-ide/brain/b06d7b61-940c-4af2-b799-8519cc9a0d48/.system_generated/logs/transcript_full.jsonl')

with open(transcript_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        if 'tool_calls' in data:
            for call in data['tool_calls']:
                name = call['name']
                if name in ['replace_file_content', 'multi_replace_file_content']:
                    args = call['args']
                    target = args.get('TargetFile')
                    if target and target.startswith('/Users/aayushidhurandhar/InferSpore/'):
                        if target == '/Users/aayushidhurandhar/InferSpore/README.md': continue
                        if name == 'replace_file_content':
                            print(f"Must manually restore replace_file_content for {target}")
                        elif name == 'multi_replace_file_content':
                            print(f"Must manually restore multi_replace_file_content for {target}")
