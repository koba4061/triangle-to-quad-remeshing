from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_ROOT.parent

DEFAULT_TEST_MESH = PROJECT_ROOT / "out_8eeec1fa.glb"
RUNS_DIR = PROJECT_ROOT / "runs"
RUN_SHARED_DIR = RUNS_DIR / "_shared"
LEGACY_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

MERGE_PERCENT = 0.01
DEGENERATE_HEIGHT = 1e-6
# VRoid / glTF は Y-up。Cross field の「たて」方向に使う
UP_AXIS = "Y"  # "X" | "Y" | "Z"
# 手動回転補正（通常は0。座標変換は coords.yup_to_blender_zup を使う）
FIELD_ROTATE_AXIS = "X"
FIELD_ROTATE_DEG = 0.0
BLENDER_ZUP_EXPORT = True
