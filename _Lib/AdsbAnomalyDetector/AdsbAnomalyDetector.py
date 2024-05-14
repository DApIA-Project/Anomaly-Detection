import os
import numpy as np
HERE = os.path.abspath(os.path.dirname(__file__))
from ._Utils_module import module_to_dict
from .E_Trainer_TrajectorySeparator_Trainer import Trainer as TrajectorySeparator
from .E_Trainer_AircraftClassification_Trainer import Trainer as AircraftClassification
from . import _Utils_FeatureGetter as FG
from ._Utils_ADSB_Streamer import cast_msg

def getCTX(CTX, default_CTX=None):
    CTX = module_to_dict(CTX)
    if (default_CTX != None):
        default_CTX = module_to_dict(default_CTX)
        for param in default_CTX:
            if (param not in CTX):
                CTX[param] = default_CTX[param]
    return CTX



from .B_Model_AircraftClassification_CNN2 import Model as CNN2
from . import C_Constants_AircraftClassification_CNN as CNN2_CTX
from . import C_Constants_AircraftClassification_DefaultCTX as CNN2_DefaultCTX
CTX_AC = getCTX(CNN2_CTX, CNN2_DefaultCTX)
aircraftClassification = AircraftClassification(CTX_AC, CNN2)
aircraftClassification.load(HERE)


from .B_Model_TrajectorySeparator_Model import Model as ALG
from . import C_Constants_TrajectorySeparator_Model as ALG_CTX
from . import C_Constants_TrajectorySeparator_DefaultCTX as ALG_DefaultCTX
from .D_DataLoader_AircraftClassification_Utils import getLabel
CTX_TS = getCTX(ALG_CTX, ALG_DefaultCTX)
trajectorySeparator = TrajectorySeparator(CTX_TS, ALG)
trajectorySeparator.load(HERE)





def predict(messages: "list[dict[str, str]]"):
    for i in range(len(messages)):
        messages[i] = {col:cast_msg(col,  messages[i].get(col, np.nan)) for col in messages[i]}


    FG.init(CTX_TS)
    sub_icaos = trajectorySeparator.predict(messages)
    for i in range(len(messages)):
        messages[i]["icao24"] = sub_icaos[i]


    FG.init(CTX_AC)
    label_proba = aircraftClassification.predict(messages)
    icaos = [messages[i]["icao24"] for i in range(len(messages))]
    icaos = [icaos[i].split("_")[0] if ("_" in icaos[i]) else icaos[i] for i in range(len(icaos))]


    preds_labels = [CTX_AC["USED_LABELS"][np.argmax(label_proba[i])] for i in range(len(label_proba))]
    labels = [getLabel(CTX_AC, icaos[i]) for i in range(len(icaos))]

    for i in range(len(messages)):
        messages[i]["spoofing"] = preds_labels[i] != labels[i]

    return messages