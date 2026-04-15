import os
import importlib
import inspect

__all__ = []

# Get the directory of the current package
package_dir = os.path.dirname(__file__)
root_package = __name__

# Recursively walk through the directory to find evaluation.py files
for dirpath, _, filenames in os.walk(package_dir):
    if 'evaluation.py' in filenames:
        # Calculate relative path from package_dir
        rel_path = os.path.relpath(dirpath, package_dir)
        
        if rel_path == '.':
            submodule_suffix = 'evaluation'
        else:
            # Convert path separators to dots
            submodule_suffix = rel_path.replace(os.path.sep, '.') + '.evaluation'
            
        try:
            # Import the module (e.g., llm4ad.task.optimization.cvrp_construct.evaluation)
            module = importlib.import_module(f'.{submodule_suffix}', package=root_package)
            
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
            # Optionally print a warning if a submodule cannot be imported
            pass

def import_all_evaluation_classes(root_directory):
    """
    This function is kept for compatibility but the dynamic import
    is now handled by the package's __init__.py.
    """
    pass
