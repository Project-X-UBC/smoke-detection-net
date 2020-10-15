from detectron2.config import CfgNode


__all__ = ['get_cfg']


def get_cfg() -> CfgNode:
    """Get a copy of the default config

    Returns:
        CfgNode -- a detectron2 CfgNode instance
    """

    from .defaults import _C
    return _C.clone()
    

