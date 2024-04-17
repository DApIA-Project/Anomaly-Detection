
from D_DataLoader.Airports import *

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
    res = dict(zip(var_names, var_val))

    if ("USED_FEATURES" in res):

        if ("toulouse" in res["USED_FEATURES"]):
            res["USED_FEATURES"].remove("toulouse")
            for airport in range(len(TOULOUSE)):
                res["USED_FEATURES"].append("toulouse_"+str(airport))

            if ("FEATURES_IN" in res):
                res["FEATURES_IN"] = len(res["USED_FEATURES"])

                if ("FEATURE_MAP" in res):
                    res["FEATURE_MAP"] = dict([[res["USED_FEATURES"][i], i] for i in range(res["FEATURES_IN"])])
            
    if ("ADD_AIRPORT_CONTEXT" in res and res["ADD_AIRPORT_CONTEXT"]):
        res["AIRPORT_CONTEXT_IN"] = len(TOULOUSE)
        if ("ADD_TAKE_OFF_CONTEXT" in res and res["ADD_TAKE_OFF_CONTEXT"]):
            res["AIRPORT_CONTEXT_IN"] *= 2

    return res  