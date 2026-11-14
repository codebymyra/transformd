"""
From ATESA.
Factory script for obtaining the desired interfaces from the various interface scripts.
"""

import mdengine
from models import sequence_featurizer 

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


def sequence_featurizer_factory(featurizer_toolkit):
    '''
    Factory function for Sequence Featurizers. 

    Parameters
    ---
    featurizer_toolkit: str
        Name of sequence featurizer to invoke
    
    Returns
    ---
    featurizer: Featurizer
        Instance of a sequence Featurizer adapter 
    '''

    featurizer_toolkits = {
        'onehot': sequence_featurizer.OneHot(),
        'physicochemical': sequence_featurizer.Physicochemical(),
        'esm2': sequence_featurizer.ESM2(),
    }

    if featurizer_toolkit not in featurizer_toolkits.keys():
        raise ValueError('unsupported sequence featurizer: ', featurizer_toolkit)

    return featurizer_toolkits[featurizer_toolkit]