import os
import importlib
import inspect

__all__ = []

# Get the directory of the current package
package_dir = os.path.dirname(__file__)

# Iterate over all files in the package directory
for filename in os.listdir(package_dir):
    # Check if it is a Python file and not the __init__.py file
    if filename.endswith('.py') and filename != '__init__.py':
        module_name = filename[:-3]
        try:
            # Import the module (e.g., llm4ad.tools.profiler.profile)
            module = importlib.import_module(f'.{module_name}', package=__name__)
            
            # Iterate over the attributes of the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                # Check if the attribute is a class defined in that module
                if inspect.isclass(attr) and attr.__module__ == module.__name__:
                    # Add the class to the package's globals
                    globals()[attr_name] = attr
                    # Add the class name to __all__ to be exported
                    if attr_name not in __all__:
                        __all__.append(attr_name)
        except (ImportError, ModuleNotFoundError) as e:
            # Optionally print a warning if a module cannot be imported
            pass

def import_all_profiler_classes_from_subfolders(root_directory):
    """
    This function is kept for compatibility but the dynamic import
    is now handled by the package's __init__.py.
    """
    pass
