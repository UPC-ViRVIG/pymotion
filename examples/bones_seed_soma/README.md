# BONES-SEED SOMA Example

This example shows how to use PyMotion to load [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) BVH motions and pass them through the SOMA body model.

The example dependencies are intentionally kept outside the main PyMotion package. Install them only in the environment where you want to run this example:

```bash
pip install -r examples/bones_seed_soma/requirements.txt
```

Install [SOMA](https://github.com/NVlabs/SOMA-X) following the upstream SOMA instructions, and make sure this import works in the same environment:

```python
from soma import SOMALayer
```

You also need:

- The [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) dataset
- [SOMA](https://github.com/NVlabs/SOMA-X) assets downloaded locally


Run the example:

```bash
python examples/bones_seed_soma/process_bones_seed_soma.py \
    --bones-seed-root /path/to/BONES-SEED \
    --soma-assets /path/to/soma/assets
```

By default the script chooses a random motion and processes both the uniform and proportional SOMA BVH variants. Use `--motion-index` for a deterministic metadata row.

To sanity-check the selected BVH skeletons in Blender, add `--viz`:

```bash
python examples/bones_seed_soma/process_bones_seed_soma.py \
    --bones-seed-root /path/to/BONES-SEED \
    --soma-assets /path/to/soma/assets \
    --motion-index 0 \
    --viz
```

This opens or connects to Blender through `BlenderConnection`, clears the scene, adds a checkerboard floor, and renders the selected uniform and proportional BVH skeletons with different colors. The example uses a visual-only Blender BVH import scale of `0.01` and hides BVH end-site helper bones plus joints matching `twist`; this does not modify the BVH file or the tensors sent to SOMA.

If Blender is not found automatically, pass the executable path:

```bash
python examples/bones_seed_soma/process_bones_seed_soma.py \
    --bones-seed-root /path/to/BONES-SEED \
    --soma-assets /path/to/soma/assets \
    --viz \
    --blender-executable "/path/to/blender.exe"
```

If your BVH files store translations in centimeters, keep the default `--translation-scale 0.01` to pass meters into SOMA.

This script excludes joints whose names contain `Root` before passing rotations to SOMA. This example follows that behavior by default; override the filter with `--exclude-joint-substring`, or pass `--keep-all-joints` if your local BVH layout differs.


Note: From a PyMotion development checkout, install the local package in editable mode so the example uses your source tree:

```bash
pip install -e .
```
