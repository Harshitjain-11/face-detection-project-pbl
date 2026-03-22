# debug_state.py
# Shows folders under static/uploads and counts image files per folder (helps identify misplaced images)

import os

UPLOAD_FOLDER = 'static/uploads'

print("Checking:", UPLOAD_FOLDER)
if not os.path.exists(UPLOAD_FOLDER):
    print("Folder does not exist.")
else:
    entries = sorted(os.listdir(UPLOAD_FOLDER))
    if not entries:
        print("No subfolders in static/uploads")
    for p in entries:
        pp = os.path.join(UPLOAD_FOLDER, p)
        if os.path.isdir(pp):
            # count image files directly inside this folder (not in originals subfolder)
            direct = sum(1 for f in os.listdir(pp) if os.path.isfile(os.path.join(pp, f)) and f.lower().endswith(('.jpg','.jpeg','.png')))
            # count originals
            originals = 0
            orig_path = os.path.join(pp, 'originals')
            if os.path.isdir(orig_path):
                originals = sum(1 for f in os.listdir(orig_path) if os.path.isfile(os.path.join(orig_path, f)) and f.lower().endswith(('.jpg','.jpeg','.png')))
            print(f"- {p}: direct_images={direct}, originals={originals}")