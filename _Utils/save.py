import tensorflow as tf

import pickle as pkl

def one_line_json(multiline_json):
    one_line_json = multiline_json.replace("\n", "")
    one_line_json = one_line_json.replace("\t", "")
    return one_line_json

def formatJson(one_line_json):
    """
    {"a":1, "b":[1,2,3], "c": {"d": 4, "e": 5}}

    becomes

    {
        "a": 1,
        "b": [1, 2, 3],
        "c": {
            "d": 4,
            "e": 5
        }
    }
    """

    multiline_json = ""
    indent = 0
    for c in one_line_json:
        if c == "{":
            indent += 1
            multiline_json += "{\n" + "    "*indent
        elif c == "}":
            indent -= 1
            multiline_json += "\n" + "    "*indent + "}"
        elif c == ",":
            multiline_json += ",\n" + "    "*indent
        else:
            multiline_json += c

    return multiline_json


def write(path, array, level=0):
    """
    Write the array in the given path
    """

    if (isinstance(array, (tf.Variable, tf.Tensor))):
        array = array.numpy()

    file = open(path, "wb")
    pkl.dump(array, file)
    file.close()




def load(path):
    """
    Load the array from the given path
    """
    file = open(path, "rb")
    array = pkl.load(file)
    file.close()
    return array



        


