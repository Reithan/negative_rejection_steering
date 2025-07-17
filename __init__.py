from .NRS.nodes_NRS import NRS, NRSEpsilon, NRSFDG

NODE_CLASS_MAPPINGS = {
    "NRS": NRS,
    "NRSEpsilon": NRSEpsilon,
    "NRSFDG": NRSFDG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NRS": "Negative Rejection Steering",
    "NRSEpsilon": "NRS EpsilonPred",
    "NRSFDG": "NRS + Frequency Denoise Guidance",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']