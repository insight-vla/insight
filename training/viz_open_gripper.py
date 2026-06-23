#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import json
import cv2
import io
from PIL import Image, ImageDraw, ImageFont
import subprocess

dataset = Path.home() / '.cache/huggingface/lerobot/maggiewang/lego_primitives_v5_01_30_trimmed'
with open(dataset / 'meta/episodes.jsonl') as f:
    episodes = [json.loads(l) for l in f]

open_gripper_eps = [ep for ep in episodes if 'open gripper' in ep['tasks'][0].lower()]
out_dir = Path('data/libero/dataset_viz_lego_primitives_v5_01_30_trimmed/open_gripper_debug')
out_dir.mkdir(parents=True, exist_ok=True)

try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
except:
    font = None

for i, ep in enumerate(open_gripper_eps[:10]):
    ep_idx = ep['episode_index']
    chunk = ep_idx // 1000
    df = pd.read_parquet(dataset / f'data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet')

    frames = []
    states = np.stack(df['state'].values)
    for j, (_, row) in enumerate(df.iterrows()):
        img_data = row['image']
        if isinstance(img_data, dict) and 'bytes' in img_data:
            img = Image.open(io.BytesIO(img_data['bytes']))
            draw = ImageDraw.Draw(img)
            gripper_state = states[j, 7]
            text = f'Gripper: {gripper_state:.3f} | {j+1}/{len(df)}'
            if font:
                draw.text((11, 11), text, fill=(0, 0, 0), font=font)
                draw.text((10, 10), text, fill=(255, 255, 255), font=font)
            else:
                draw.text((11, 11), text, fill=(0, 0, 0))
                draw.text((10, 10), text, fill=(255, 255, 255))
            frames.append(np.array(img))

    if frames:
        temp_path = out_dir / f'ep{ep_idx}_len{len(df)}_temp.mp4'
        final_path = out_dir / f'ep{ep_idx}_len{len(df)}.mp4'
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        subprocess.run(['ffmpeg', '-y', '-i', str(temp_path), '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', str(final_path)], capture_output=True)
        temp_path.unlink()
        print(f'Ep {ep_idx}: {len(df)} frames')

print('Done!')
