import os
import subprocess


ALL_PY = []
for root, dirs, files in os.walk(f"../"):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            path = path.replace("../", "")
            path = path.replace(".py", "")
            path = path.replace("/", ".")
            ALL_PY.append(path)




def list_imports(py_lines):

    imports = []
    locs = []
    for i in range(len(py_lines)):
        line = py_lines[i]
        if line.startswith("from"):
            l = line.split("import")
            file = l[0].strip().split(" ")[-1]
            if (file in ALL_PY):
                imports.append(file)
            # if is directory
            elif (os.path.isdir(f"../{file.replace('.', '/')}/")):
                imports.append(file)
            else:
                imports.append(l[0].split(" ")[1] + "." + l[1].strip())
            locs.append(i)

        elif line.startswith("import"):
            imports.append(line.split(" ")[1])
            locs.append(i)
    return imports, locs

def read_lines(file):
    if (os.path.isdir(file)):
        return []
    flux = open(file, "r")
    content = flux.read()
    flux.close()
    return content.split("\n")

def compute_trg_name(file:str):
    if (file.startswith("../")):
        file = file[2:]
    if (file.endswith(".py")):
        file = file[:-3]

    file = file.strip("/")
    file = file.replace("/", "_")

    return file


files_map = {}
def add_file_to_lib(file, level = 0, folder = False):
    global files_map

    # get file register
    if (file in files_map):
        print("\t"*level + f"{file} already copied")
        return

    # the target file name
    trg_name = compute_trg_name(file)

    if not(folder):
        os.system(f"cp {file} ./AdsbAnomalyDetector/{trg_name}.py")
    else:
        os.system(f"cp -r {file} ./AdsbAnomalyDetector/{trg_name}")
    files_map[file] = trg_name
    print("\t"*level + f"> cp {file} ./AdsbAnomalyDetector/{trg_name}")


    lines = read_lines(file)
    imports, locs = list_imports(lines)

    # print all imports
    print("\t"*(level) + f"{file.split('/')[-1]} imports :")
    for i, imp in zip(locs, imports):

        valid = imp in ALL_PY
        folder = False
        path = f"../{imp.replace('.', '/')}/"
        if (os.path.isdir(path)):
            valid = True
            folder = True

        if valid:
            print("\t"*(level+1) + f"{imp}")

            filename = "../"+imp.replace(".", "/")+ ".py"
            if (folder):
                filename = path
                add_file_to_lib(filename, level = level + 1, folder = True)
            else: add_file_to_lib(filename, level = level + 1)

            imp_new_name = files_map[filename]

            if lines[i].startswith("from"):

                lines[i] = f"from .{imp_new_name} import {lines[i].split('import')[1].strip()}"
            if (lines[i].startswith("import")):
                if (" as " in lines[i]):
                    lines[i] = f"from . import {imp_new_name} as {lines[i].split(' as ')[1].strip()}"
                else:
                    lines[i] = f"from .  import {imp_new_name}"

    # write the file
    flux = open(f"./AdsbAnomalyDetector/{trg_name}.py", "w")
    flux.write("\n".join(lines))
    flux.close()


def file_content_remplace(_file, find, remplace):
    file = open(_file, "r")
    content = file.read()
    file.close()

    content = content.replace(find, remplace)

    file = open(f"{_file}", "w")
    file.write(content)
    file.close()





# |====================================================================================================================
# | LIB BUILDING
# |====================================================================================================================




# clean lib before build
to_reomve = []
for root, dirs, files in os.walk(f"./AdsbAnomalyDetector/"):
    for file in files:
        if file != "AdsbAnomalyDetector.py" and file != "__init__.py":
            to_reomve.append(os.path.join(root, file))
    for dir in dirs:
        if dir != "__pycache__":
            to_reomve.append(os.path.join(root, dir))

for file in to_reomve:
    os.system(f"rm -r {file}")




# list required imports
files = [
    "../G_Main/AircraftClassification/exp_CNN2.py",
    "../G_Main/TrajectorySeparator/exp_GEO.py",
    "../G_Main/ReplaySolver/exp_HASH.py",
    "../G_Main/FloodingSolver/exp_LSTM.py"
]
MODELS = [
    "CNN2",
    "ALG",
    "HASH",
    "LSTM"
]

imports = set()
for path in files:
    file = open(path, "r")
    content = file.read()
    file.close()
    lines = content.split("\n")
    imp, _ = list_imports(lines)
    imports = imports.union(set(imp))
imports = list(imports)


to_reomve = []
for i in range(len(imports)):
    # remove all lib imports (files that are not in ALL_PY)
    if imports[i] not in ALL_PY:
        to_reomve.append(i)

    # remove runner imports (launching training so useless)
    if imports[i].startswith("F_Runner"):
        to_reomve.append(i)

for i in to_reomve[::-1]:
    imports.pop(i)

imports.append("_Utils.module")










# copy all imports
for import_ in imports:
    f =  f"../{import_.replace('.', '/')}.py"
    to = f"./AdsbAnomalyDetector/{import_.split('.')[-1]}.py"

    if ("B_Model" in f):
        to = f"./AdsbAnomalyDetector/model.py"
    if ("C_Constant" in f and not("Default" in f)):
        to = f"./AdsbAnomalyDetector/CTX.py"

    add_file_to_lib(f)


# |====================================================================================================================
# | AIRCRAFT CLASSIFICATION
# |====================================================================================================================
# # copy weights
os.system(f"mkdir ./AdsbAnomalyDetector/AircraftClassification")
os.system(f"cp ../_Artifacts/AircraftClassification/{MODELS[0]}/w ./AdsbAnomalyDetector/AircraftClassification/w")
os.system(f"cp ../_Artifacts/AircraftClassification/{MODELS[0]}/xs ./AdsbAnomalyDetector/AircraftClassification/xs")
os.system(f"cp ../_Artifacts/AircraftClassification/{MODELS[0]}/xts ./AdsbAnomalyDetector/AircraftClassification/xts")
os.system(f"cp ../_Artifacts/AircraftClassification/{MODELS[0]}/xas ./AdsbAnomalyDetector/AircraftClassification/xas")
os.system(f"cp ../_Artifacts/AircraftClassification/{MODELS[0]}/pad ./AdsbAnomalyDetector/AircraftClassification/pad")
# # copy geo map
os.system("cp ../A_Dataset/AircraftClassification/map.png ./AdsbAnomalyDetector/map.png")
os.system("cp ../A_Dataset/AircraftClassification/labels.csv ./AdsbAnomalyDetector/labels.csv")

file_content_remplace("./AdsbAnomalyDetector/D_DataLoader_AircraftClassification_Utils.py",
                      "from ._Utils_os_wrapper import os",
                      "from ._Utils_os_wrapper import os\nHERE = os.path.abspath(os.path.dirname(__file__))")

file_content_remplace("./AdsbAnomalyDetector/D_DataLoader_AircraftClassification_Utils.py",
                      "\"A_Dataset/AircraftClassification/map.png\"",
                      "HERE+\"/map.png\"")

file_content_remplace("./AdsbAnomalyDetector/D_DataLoader_AircraftClassification_Utils.py",
                      "\"./A_Dataset/AircraftClassification/labels.csv\"",
                      "HERE+\"/labels.csv\"")

file_content_remplace("./AdsbAnomalyDetector/E_Trainer_TrajectorySeparator_Trainer.py",
                      "DEBUG_PER_TIMESTEPS = True",
                      "DEBUG_PER_TIMESTEPS = False")
file_content_remplace("./AdsbAnomalyDetector/E_Trainer_TrajectorySeparator_Trainer.py",
                      "DEBUG = True",
                      "DEBUG = False")


# |====================================================================================================================
# | FLODDING SOLVER
# |====================================================================================================================
# FloodingSolver
# # copy weights
os.system(f"mkdir ./AdsbAnomalyDetector/FloodingSolver")
os.system(f"cp ../_Artifacts/FloodingSolver/{MODELS[3]}/w ./AdsbAnomalyDetector/FloodingSolver/w")
os.system(f"cp ../_Artifacts/FloodingSolver/{MODELS[3]}/xs ./AdsbAnomalyDetector/FloodingSolver/xs")
os.system(f"cp ../_Artifacts/FloodingSolver/{MODELS[3]}/ys ./AdsbAnomalyDetector/FloodingSolver/ys")
os.system(f"cp ../_Artifacts/FloodingSolver/{MODELS[3]}/pad ./AdsbAnomalyDetector/FloodingSolver/pad")

# |====================================================================================================================
# | REPLAY SOLVER
# |====================================================================================================================

# ReplaySolver
# # copy weights
os.system(f"mkdir ./AdsbAnomalyDetector/ReplaySolver")
os.system(f"cp ../_Artifacts/ReplaySolver/{MODELS[2]}/w ./AdsbAnomalyDetector/ReplaySolver/w")


# list every files starting with C_Constants
# find lines starting with EPOCHS
# remplace by EPOCHS = 0

files = os.listdir("./AdsbAnomalyDetector")
files = [f for f in files if f.startswith("C_Constants")]
for file in files:
    lines = read_lines(f"./AdsbAnomalyDetector/{file}")
    for i in range(len(lines)):
        if lines[i].startswith("EPOCHS"):
            lines[i] = "EPOCHS = 0"
    flux = open(f"./AdsbAnomalyDetector/{file}", "w")
    flux.write("\n".join(lines))
    flux.close()




if (os.path.exists("./dist")):
    os.system("rm -r ./dist/*")


# run setup.py
os.system("python ./setup.py sdist")