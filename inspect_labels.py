# inspect_labels.py
# Prints the current labels.pickle content (name -> id) and reversed mapping (id -> name)

import os, pickle, json

if os.path.exists('labels.pickle'):
    try:
        with open('labels.pickle','rb') as f:
            raw = pickle.load(f)
        print("labels.pickle (name -> id):")
        print(json.dumps(raw, indent=2, ensure_ascii=False))
        rev = {v: k for k, v in raw.items()}
        print("\nReversed (id -> name):")
        print(json.dumps(rev, indent=2, ensure_ascii=False))
    except Exception as e:
        print("Error reading labels.pickle:", e)
else:
    print("labels.pickle not found")