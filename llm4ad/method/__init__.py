from llm4ad.method import (
    funsearch,
    hillclimb,
    randsample,
    eoh,
    meoh,
    mles,
    moead,
    nsga2,
    partevo,
    llamea,
    reevo
)

__all__ = ['funsearch', 'hillclimb', 'randsample', 'eoh', 'meoh', 'mles', 'moead', 'nsga2', 'partevo', 'llamea', 'reevo']
# try to import reevo
# try:
#     from . import reevo
# except ImportError:
#     pass

import os
import importlib
import inspect

__all__ = []

# Get the directory of the current package
package_dir = os.path.dirname(__file__)

# Iterate over all subdirectories in the package directory
for subdir_name in os.listdir(package_dir):
    subdir_path = os.path.join(package_dir, subdir_name)
    
    # Check if it is a directory and not a special directory like __pycache__
    if os.path.isdir(subdir_path) and subdir_name != '__pycache__':
        try:
            # Import the submodule (e.g., llm4ad.method.eoh)
            submodule = importlib.import_module(f'.{subdir_name}', package=__name__)
            
            # Iterate over the attributes of the submodule
            for attr_name in dir(submodule):
                attr = getattr(submodule, attr_name)
                
                # Check if the attribute is a class defined in that submodule
                if inspect.isclass(attr) and attr.__module__.startswith(submodule.__name__):
                    # Add the class to the package's globals
                    globals()[attr_name] = attr
                    # Add the class name to __all__ to be exported
                    if attr_name not in __all__:
                        __all__.append(attr_name)
        except (ImportError, ModuleNotFoundError) as e:
            # Optionally print a warning if a submodule cannot be imported
            pass

def import_all_method_classes_from_subfolders(root_directory: str):
    """
    This function is kept for compatibility but the dynamic import
    is now handled by the package's __init__.py.
    """
    # This function can be left empty or with a pass statement.
    # The dynamic importing is now handled when the package is imported.
    pass
