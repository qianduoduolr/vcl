import sys
sys.path.insert(0, '../..')
import json
from collections import *

youtube_json_file = '/home/lr/dataset/YouTube-VOS/train/meta.json'
with open(youtube_json_file, 'r') as f:
    meta_data = json.load(f)

youtube_json_file2 = '/home/lr/dataset/YouTube-VOS/train/generated_meta.json'
with open(youtube_json_file2, 'r') as f:
    g_data = json.load(f)

frame_wise_anno = defaultdict(dict)
categorys = {}

cnum_max = 0
for k,v in meta_data['videos'].items():
    vi = g_data[k]
    vos = vi['obj_sizes']
    frame_wise_anno[k] = defaultdict(list)
    for frame_id, info in vos.items():
        for idx, (obj, size) in enumerate(info.items()):
            category = v['objects'][obj]['category']
            if not category in categorys.keys():
                categorys[category] = len(categorys.keys())+1
                c_num = categorys[category]
            else:
                c_num = categorys[category]
            if c_num >= cnum_max: cnum_max = c_num
            frame_wise_anno[k][frame_id].append((c_num, idx+1))

with open('/home/lr/dataset/YouTube-VOS/train/generated_frame_wise_meta.json','w') as f:
    json.dump(frame_wise_anno,f)