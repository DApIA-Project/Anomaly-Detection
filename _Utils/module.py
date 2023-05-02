

def module_to_dict(MODULE):
    var_names = [x for x in dir(MODULE) if not x.startswith('__')]
    var_val = [getattr(MODULE, x) for x in var_names]
    return dict(zip(var_names, var_val))