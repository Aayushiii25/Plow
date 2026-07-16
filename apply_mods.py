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
                        
                        try:
                            with open(target, 'r') as tf:
                                content = tf.read()
                        except:
                            continue
                        
                        if name == 'replace_file_content':
                            tc = args.get('TargetContent')
                            rc = args.get('ReplacementContent')
                            if tc and rc and tc in content:
                                content = content.replace(tc, rc)
                                with open(target, 'w') as tf:
                                    tf.write(content)
                                print(f"Applied replacement to {target}")
                        elif name == 'multi_replace_file_content':
                            chunks = args.get('ReplacementChunks', [])
                            for chunk in chunks:
                                tc = chunk.get('TargetContent')
                                rc = chunk.get('ReplacementContent')
                                if tc and rc and tc in content:
                                    content = content.replace(tc, rc)
                            with open(target, 'w') as tf:
                                tf.write(content)
                            print(f"Applied multi replacement to {target}")
