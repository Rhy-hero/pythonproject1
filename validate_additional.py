"""Additional input-dependence checks, beyond the untouched course checker.

The synthetic image is never in the supplied reference set. Reference data is
used only by this validation script to compare the normal command-line output.
"""
from pathlib import Path
import json
import platform
import sys
import importlib.metadata
import numpy as np
from PIL import Image
from runtime import ROOT, load_images, load_weights
from image_encoder import ImageEncoder

out = ROOT / 'results'
out.mkdir(exist_ok=True)
# Deterministic RGB gradients provide an unseen but valid image file.
y, x = np.mgrid[:257, :319]
pixels = np.stack([(3*x + y) % 256, (x + 2*y) % 256, (7*x + 5*y) % 256], -1).astype('uint8')
path = out / 'synthetic_input.png'
Image.fromarray(pixels).save(path)
encoder = ImageEncoder(load_weights())
image = load_images([path])
single = encoder(image)
changed = image.copy()
changed[:, :, 32:160, 64:192] *= -1
batch = encoder(np.concatenate([changed, image, changed], 0))
np.testing.assert_allclose(batch[1], single[0], atol=2e-5, rtol=2e-4)
np.testing.assert_allclose(batch[0], batch[2], atol=2e-5, rtol=2e-4)
np.testing.assert_allclose(np.linalg.norm(batch, axis=-1), 1, atol=2e-4, rtol=2e-4)
assert batch.dtype == np.float32 and np.isfinite(batch).all()
change = float(np.max(np.abs(batch[0] - single[0])))
assert change > 1e-5, 'An input change must not return a cached row.'
with np.load(out / 'output_tokens.npz', allow_pickle=False) as saved:
    with np.load(ROOT / 'assets/reference_outputs.npz', allow_pickle=False) as reference:
        np.testing.assert_array_equal(saved['ids'], reference['ids'])
        np.testing.assert_allclose(saved['output_tokens'], reference['output_tokens'], atol=2e-5, rtol=2e-4)
        cli_error = np.abs(saved['output_tokens'] - reference['output_tokens'])
        cli_stats = {'max_abs': float(cli_error.max()), 'mean_abs': float(cli_error.mean()),
                     'shape': list(saved['output_tokens'].shape),
                     'max_norm_error': float(np.max(np.abs(np.linalg.norm(saved['output_tokens'], axis=1)-1)))}
summary = {
    'python': platform.python_version(), 'numpy': np.__version__,
    'pillow': importlib.metadata.version('Pillow'), 'imageio': importlib.metadata.version('imageio'),
    'platform': platform.platform(),
    'checks': {'unseen_image': 'PASS', 'changed_input': 'PASS', 'reordered_duplicate_batch': 'PASS',
               'finite_float32_unit_length': 'PASS', 'cli_output_and_id_order': 'PASS'},
    'changed_input_max_abs_difference': change,
    'single_vs_batch_max_abs': float(np.max(np.abs(single[0]-batch[1]))),
    'duplicate_max_abs': float(np.max(np.abs(batch[0]-batch[2]))),
    'cli_vs_reference': cli_stats,
    'input_assumption': 'nonempty finite float32 arrays [B,3,224,224]; fixed supplied checkpoint',
}
(out / 'additional_checks.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
print(json.dumps(summary, indent=2, ensure_ascii=False))
