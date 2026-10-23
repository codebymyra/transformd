"""
From ATESA.
Factory script for obtaining the desired interfaces from the various interface scripts.
"""

from transformd import mdengine

def mdengine_factory(mdengine_toolkit):
    """
    Factory function for MDEngines.

    Parameters
    ----------
    mdengine_toolkit : str
        Name of the MDEngine to invoke

    Returns
    -------
    mdengine : MDEngine
        Instance of an MDEngine adapter

    """

    mdengine_toolkits = {'amber': mdengine.AdaptAmber()
                         }

    if mdengine_toolkit not in mdengine_toolkits.keys():
        raise ValueError('unsupported MDEngine name: ' + mdengine_toolkit)

    return mdengine_toolkits[mdengine_toolkit]