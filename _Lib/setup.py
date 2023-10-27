import os
import subprocess

used_model = "CNN2"

ALL_PY = []
for root, dirs, files in os.walk(f"../"):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            path = path.replace("../", "")
            path = path.replace(".py", "")
            path = path.replace("/", ".")
            ALL_PY.append(path)



already_copied = []


def list_imports(py_lines):

    imports = []
    locs = []
    for i in range(len(py_lines)):
        line = py_lines[i]
        if line.startswith("from"):
            l = line.split("import")
            file = l[0].split(" ")[1]
            if (file in ALL_PY):
                imports.append(file)
            else:
                imports.append(l[0].split(" ")[1] + "." + l[1].strip())
            locs.append(i)

        elif line.startswith("import"):
            imports.append(line.split(" ")[1])
            locs.append(i)
    return imports, locs


def copy_past_py(file, dest, level = 0):
    global already_copied

    # get file register
    already_copied.append(file)
    os.system(f"cp {file} {dest}")
    print("\t"*level + f"copy {file} to {dest} : ")

    flux = open(f"{dest}", "r")
    content = flux.read()
    flux.close()

    lines = content.split("\n")
    imports, locs = list_imports(lines)
    
    
    # print all imports
    print("\t"*(level) + f"{file.split('/')[-1]} imports :")
    for import_ in imports:
        print("\t"*(level+1) + f"{import_}")


    import_final_name = []
    do_not_copy = []
    for import_ in imports:

        if import_ in ALL_PY:

            file = import_.split(".")[-1]
            # is this file is already copied 
            do_not_copy.append((f"../{import_.replace('.', '/')}.py" in already_copied))
            
            if not(do_not_copy[-1]):
                n = 1
                # check if file name already exist
                while os.path.exists(f"./AdsbAnomalyDetector/{file}.py"):
                    file = import_.split(".")[-1] + f"_{n}"
                    n += 1
            import_final_name.append(f"{file}")
        else:
            import_final_name.append("None")
            do_not_copy.append(True)


    print("\t"*(level) + f"flattening file tree : ")
    for i in range(len(imports)):
        import_, file, loc = imports[i], import_final_name[i], locs[i]

        if import_ in ALL_PY:
            # rename import in dest
            print("\t"*(level+1) + f"{import_} renamed to {file}")

            if (lines[loc].startswith("from")):
                # check that in "from [file] ...", file exist
                f = lines[loc].split(" ")[1]
                if (f in ALL_PY): 
                    lines[loc] = lines[loc].replace(import_, f".{file}")
                else:
                    lines[loc] = "from . import " + file
            else:
                lines[loc] = lines[loc].replace(import_, f"{file}")
                lines[loc] = "from . " + lines[loc]

    content = "\n".join(lines)
    file = open(f"{dest}", "w")
    file.write(content)
    file.close()

    print("\t"*(level) + f"copying files : ")
    for import_, file, n_copy in zip(imports, import_final_name, do_not_copy):
        if import_ in ALL_PY and not(n_copy):
            copy_past_py(f"../{import_.replace('.', '/')}.py", f"./AdsbAnomalyDetector/{file}.py", level = level + 1)
        
def file_content_remplace(_file, find, remplace):
    file = open(_file, "r")
    content = file.read()
    file.close()

    content = content.replace(find, remplace)

    file = open(f"{_file}", "w")
    file.write(content)
    file.close()

# file = f"../B_Model/AircraftClassification/{used_model}.py"
# dest = f"./AdsbAnomalyDetector/model.py"
to_reomve = []
for root, dirs, files in os.walk(f"./AdsbAnomalyDetector/"):
    for file in files:
        if file != "AdsbAnomalyDetector.py" and file != "__init__.py":
            to_reomve.append(os.path.join(root, file))

for file in to_reomve:
    os.system(f"rm {file}")




# take main, and list all it's imports
file = open(f"../G_Main/AircraftClassification/exp_{used_model}.py", "r")
content = file.read()
file.close()
lines = content.split("\n")
imports, _ = list_imports(lines)

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

    copy_past_py(f, to)



# copy weights
os.system(f"cp ../_Artefact/{used_model}.w ./AdsbAnomalyDetector/w")
os.system(f"cp ../_Artefact/{used_model}.xs ./AdsbAnomalyDetector/xs")
os.system(f"cp ../_Artefact/{used_model}.xts ./AdsbAnomalyDetector/xts")
# copy geo map
os.system(f"cp ../A_Dataset/AircraftClassification/map.png ./AdsbAnomalyDetector/map.png")
os.system(f"cp ../A_Dataset/AircraftClassification/labels.csv ./AdsbAnomalyDetector/labels.csv")


# os.system(f"cp ../_Utils/module.py ./AdsbAnomalyDetector/module.py")


file_content_remplace("./AdsbAnomalyDetector/Utils.py", 
                      "import os", 
                      "import os\nHERE = os.path.abspath(os.path.dirname(__file__))")

file_content_remplace("./AdsbAnomalyDetector/Utils.py", 
                      "\"A_Dataset/AircraftClassification/map.png\"", 
                      "HERE+\"/map.png\"")


file_content_remplace("./AdsbAnomalyDetector/mlflow.py",
                      "USE_MLFLOW = True", 
                      "USE_MLFLOW = False")

file_content_remplace("./AdsbAnomalyDetector/mlviz.py",
                      "USE_MLVIZ=True", 
                      "USE_MLVIZ=False")






from setuptools import setup, find_packages

VERSION = '0.1.0' 
DESCRIPTION = 'Low altitude aircraft anomaly detector'
LONG_DESCRIPTION = """
    This package contains a trained model to detect anomaly in low altitude aircraft.
    The model focus on spoofing attack, by recognizing the aircraft the type based on its ADS-B signal.
    Hence, if the aircraft pretend to be a small aircraft, but the model recognize it as a plane, it is an anomaly. 
"""

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="AdsbAnomalyDetector", 
        version=VERSION,
        author="Pirolley Melvyn",
        author_email="",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=["tensorflow", "numpy", "pandas", "scikit-learn", "matplotlib", "pickle-mixin"], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        package_data={'': ['*.py', "w", "xs", "xts", "map.png", "labels.csv"]},
        keywords=['python', 'deep learning', 'tensorflow', 'aircraft', 'classification', 'ADS-B'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)

# scp dist folder to ..
if (os.path.exists("../_dist")):
    os.system("rm -r ../_dist/*")
else:
    os.system("mkdir ../_dist")

os.system("cp -r ./dist/* ../_dist/")