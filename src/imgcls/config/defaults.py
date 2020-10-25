from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# MobileNets
# ---------------------------------------------------------------------------- #
_C.MODEL.MNET = CN()

# Output features
_C.MODEL.MNET.OUT_FEATURES = ['linear']
# Width mult
_C.MODEL.MNET.WIDTH_MULT = 1.0

# ---------------------------------------------------------------------------- #
# ClsNets
# ---------------------------------------------------------------------------- #
_C.MODEL.CLSNET = CN()
_C.MODEL.CLSNET.ENABLE = False
# classes number
_C.MODEL.CLSNET.NUM_CLASSES = 16
# In features
_C.MODEL.CLSNET.IN_FEATURES = ['linear']
# Input Size
_C.MODEL.CLSNET.INPUT_SIZE = 224

# data directory
_C.DATA_DIR_PATH = './data'

# weights for criterion
_C.MODEL.POS_WEIGHT = [1 for i in range(16)]

# eval only mode
_C.EVAL_ONLY = False

# early stopping mode
_C.EARLY_STOPPING = CN()
_C.EARLY_STOPPING.ENABLE = False
_C.EARLY_STOPPING.MONITOR = ''
_C.EARLY_STOPPING.PATIENCE = 0
_C.EARLY_STOPPING.MODE = 'max'
