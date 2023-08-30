import os
import subprocess

used_model = "CNN_img"
type = "Map" if ("img" in used_model) else "Raw"

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




def copy_past_py(file, dest, level = 0):
    global already_copied

    # get file register
    already_copied.append(file)
    os.system(f"cp {file} {dest}")
    

    file = open(f"{dest}", "r")
    content = file.read()
    file.close()

    lines = content.split("\n")

    imports = []
    locs = []
    for i in range(len(lines)):
        line = lines[i]
        if line.startswith("from"):
            imports.append(line.split(" ")[1])
            locs.append(i)
        elif line.startswith("import"):
            imports.append(line.split(" ")[1])
            locs.append(i)


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
                while os.path.exists(f"./AircraftClassifier/{file}.py"):
                    file = import_.split(".")[-1] + f"_{n}"
                    n += 1
            import_final_name.append(f"{file}")
        else:
            import_final_name.append("None")
            do_not_copy.append(True)



    for i in range(len(imports)):
        import_, file, loc = imports[i], import_final_name[i], locs[i]

        if import_ in ALL_PY:
            # rename import in dest
            print("\t"*level + f"edit import {import_}")

            if (lines[loc].startswith("from")):
                lines[loc] = lines[loc].replace(import_, f".{file}")
            else:
                lines[loc] = lines[loc].replace(import_, f"{file}")
                lines[loc] = "from . " + lines[loc]

    content = "\n".join(lines)
    file = open(f"{dest}", "w")
    file.write(content)
    file.close()

    for import_, file, n_copy in zip(imports, import_final_name, do_not_copy):
        if import_ in ALL_PY and not(n_copy):
            print("\t"*level + f"copy {import_}")
            copy_past_py(f"../{import_.replace('.', '/')}.py", f"./AircraftClassifier/{file}.py", level = level + 1)
        
def file_content_remplace(_file, find, remplace):
    file = open(_file, "r")
    content = file.read()
    file.close()

    content = content.replace(find, remplace)

    file = open(f"{_file}", "w")
    file.write(content)
    file.close()

# file = f"../B_Model/AircraftClassification/{used_model}.py"
# dest = f"./AircraftClassifier/model.py"
to_reomve = []
for root, dirs, files in os.walk(f"./AircraftClassifier/"):
    for file in files:
        if file != "AircraftClassification.py" and file != "__init__.py":
            to_reomve.append(os.path.join(root, file))

for file in to_reomve:
    os.system(f"rm {file}")




# copy model
copy_past_py(
    f"../B_Model/AircraftClassification/{used_model}.py", 
    f"./AircraftClassifier/model.py")


# copy CTX
os.system(f"cp ../C_Constants/AircraftClassification/{used_model}.py ./AircraftClassifier/CTX.py")
os.system(f"cp ../C_Constants/AircraftClassification/DefaultCTX.py ./AircraftClassifier/DefaultCTX.py")

# copy dataloader
copy_past_py(
    f"../D_DataLoader/AircraftClassification/{type}DataLoader.py",
    f"./AircraftClassifier/dataloader.py")

# copy trainer
copy_past_py(
    f"../E_Trainer/AircraftClassification/{type}Trainer.py",
    f"./AircraftClassifier/trainer.py")


# copy weights
os.system(f"cp ../_Artefact/{used_model}.w ./AircraftClassifier/w")
os.system(f"cp ../_Artefact/{used_model}.xs ./AircraftClassifier/xs")
os.system(f"cp ../_Artefact/{used_model}.ys ./AircraftClassifier/ys")


os.system(f"cp ../_Utils/module.py ./AircraftClassifier/module.py")

# copy geo map
os.system(f"cp ../A_Dataset/AircraftClassification/map.png ./AircraftClassifier/map.png")

file_content_remplace("./AircraftClassifier/dataloader.py", 
                      "import os", 
                      "import os\nHERE = os.path.abspath(os.path.dirname(__file__))")

file_content_remplace("./AircraftClassifier/dataloader.py", 
                      "\"A_Dataset/AircraftClassification/map.png\"", 
                      "HERE+\"/map.png\"")

# finally trainer.py remplace dataloader import
file_content_remplace("./AircraftClassifier/trainer.py", 
                      f"from .{type}DataLoader import DataLoader", 
                      "from .dataloader import DataLoader")





from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Classification of aircraft type from ADSB data'
LONG_DESCRIPTION = ''

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="AircraftClassifier", 
        version=VERSION,
        author="Pirolley Melvyn",
        author_email="",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=["tensorflow", "numpy", "pandas", "scikit-learn", "matplotlib", "pickle-mixin"], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        package_data={'': ['*.py', "w", "xs", "ys", "map.png"]},
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