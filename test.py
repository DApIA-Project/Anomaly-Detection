import pandas as pd

# set a as index
df = pd.DataFrame({"a":[1,2,3, 8, 9, 10], "b":[4,5,6, 7, 8, 9]})
df.set_index("a", inplace=True)


df.append(pd.DataFrame({"a":[4,5,6], "b":[14,15,16]}).set_index("a"))