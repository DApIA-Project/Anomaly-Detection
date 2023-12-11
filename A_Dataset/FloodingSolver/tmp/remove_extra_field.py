

file = open("./base3.csv", "r")

lines = file.readlines()

for i, line in enumerate(lines):
    l = line.split(",")[:17]
    l = ",".join(l)
    lines[i] = l + "\n"


# write to file
file = open("./base3.csv", "w")
file.writelines(lines)

