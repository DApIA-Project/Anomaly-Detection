# stocastic search, return the indice of the largest time smaller or equal to the time

from _Utils.RunLogger import RunLogger


logger = RunLogger("./_Artifacts/logs.pkl")

l = logger.where("PROBLEM", eq="FloodingSolver", inplace=False)

for i in range(len(l)):
    logger.remove_run(l.get("i", i))
    
print(logger)