from .NRS.nodes_NRS import *
from .NRS.nodes_NRS import NRSEpsilon # Import NRSEpsilon

NODE_CLASS_MAPPINGS = {
    "NRS": NRS,
    "NRSEpsilon": NRSEpsilon, # Add NRSEpsilon mapping
}
NODE_DISPLAY_NAME_MAPPINGS = { # Corrected typo MAPPINS to MAPPINGS
    "NRS": "Negative Rejection Steering",
    "NRSEpsilon": "NRS EpsilonPred", # Add NRSEpsilon display name mapping
}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']