from _Utils.os_wrapper import os


os.system("mv ../A_Dataset/AircraftClassification/Train/* ./C_dataset/ToulouseV2/Train/")
os.system("mv ../A_Dataset/AircraftClassification/Eval/* ./C_dataset/ToulouseV2/Eval/")

os.system("mv ./C_dataset/ToulouseV2/Train/* ./B_csv/ToulouseV2/")
os.system("mv ./C_dataset/ToulouseV2/Eval/* ./B_csv/ToulouseV2/")
os.system("mv ./ToulouseV2/ToulouseV2/* ./B_csv/ToulouseV2/")

          
