import os
import json

files_in_arch = [f for f in os.listdir("/Users/moatasimfarooque/Downloads/autoresearch-mlx/figs/arch") if f.endswith('.png')]

with open("/Users/moatasimfarooque/Downloads/autoresearch-mlx/figs/arch/.generated_manifest.json") as f:
    manifest_files = json.load(f)

# The user wants to:
# 1. Regenerate `manifest_files`
# 2. Check "models in the registry" for missing visualizations.

# Let's see what is in models/__init__.py
models_file = "/Users/moatasimfarooque/Downloads/autoresearch-mlx/models/__init__.py"
with open(models_file) as f:
    content = f.read()

import re
all_match = re.search(r'__all__\s*=\s*\[(.*?)\]', content, re.DOTALL)
if all_match:
    models_list_str = all_match.group(1)
    models = re.findall(r'"([^"]+)"', models_list_str)
else:
    models = []

print("Models in __init__:", models)

# Let's map model names to expected PNG names
# usually ModelName.png, but if it ends in 1d/2d, maybe stripped or maybe not.
expected_imgs = []
for m in models:
	if "Conv" in m or "Block" in m or "Mixer" in m or "Layer" in m:
		continue
	expected_imgs.append(f"{m}.png")
	if m.endswith('1d'):
		expected_imgs.append(f"{m[:-2]}.png")

print("Missing models that are definitely absent from arch:")
for m in models:
	if "Conv" in m or "Block" in m or "Mixer" in m or "Layer" in m:
		continue
	if f"{m}.png" not in files_in_arch and f"{m[:-2]}.png" not in files_in_arch and f"{m.replace('1d', '2D')}.png" not in files_in_arch and f"{m.replace('1d', '2d')}.png" not in files_in_arch:
		print(" -", m)
