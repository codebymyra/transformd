"""Repository for TransforMD, a pipeline for integrating molecular dynamics simulations and machine learning."""

from . import main
from . import configure
from . import database
from . import factory
from . import features
from . import featurizer 
from . import mdengine
from . import transformd

# Add imports here
#from ._version import __version__
from .features import *
from .utilities import *
from ._version import __version__
#from transformd import main, featurizer, mdengine, factory, database, configure


# Handle versioneer
#from ._version import get_versions

#versions = get_versions()
#__version__ = versions['version']
#__git_revision__ = versions['full-revisionid']
#del get_versions, versions

