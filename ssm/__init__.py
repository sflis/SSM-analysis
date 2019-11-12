from . import version
import daiquiri as _daiquiri
_daiquiri.setup()
__version__ = version.get_version(pep440=False)
