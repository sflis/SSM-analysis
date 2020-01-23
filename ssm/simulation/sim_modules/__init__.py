__all__ = []

import pkgutil as _pkgutil
import inspect as _inspect
from ssm.core.pchain import ProcessingModule

for _loader, _name, _is_pkg in _pkgutil.walk_packages(__path__):
    _module = _loader.find_module(_name).load_module(_name)

    for _key, _value in _inspect.getmembers(_module):
        if _name.startswith("__"):
            continue
        if _inspect.isclass(_value) and issubclass(_value, ProcessingModule):
            globals()[_key] = _value
            __all__.append(_key)
