

def module_to_dict(Module):
    """
    Convert a python module to a dict

    Parameters:
    -----------

    Module: Module
        Python module to convert to dict

    Returns:
    --------
    context dictionary : dict
        Dictionary containing the variables of the module
    """

    var_names = [x for x in dir(Module) if not x.startswith('__')]
    var_val = [getattr(Module, x) for x in var_names]
    return dict(zip(var_names, var_val))