import numpy as np
import tensorflow as tf

import struct
import sys

import pickle as pkl


def float2bin(f):
    ''' Convert float to 64-bit binary string.
        then to int64 and then to string with base
        abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_
    '''
    [d] = struct.unpack(">Q", struct.pack(">d", f))
    return f'{d:064b}'

def bin2float(b):
    ''' Convert binary string to float.
        then to int64 and then to string with base
        abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_
    '''
    d = int(b, 2)
    return struct.unpack(">d", struct.pack(">Q", d))[0]

def int2bin6(i):
    ''' Convert int to 6-bit binary string.
        then to int64 and then to string with base
        abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_
    '''
    return f'{i:06b}'
def bin2int6(b):
    ''' Convert binary string to int.
        then to int64 and then to string with base
        abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_
    '''
    return int(b, 2)

alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
def _reval(c):
    """
    Reverse eval
    """
    i = ord(c)
    if i >= 97 and i <= 122:
        return i-97
    elif i >= 65 and i <= 90:
        return i-39
    elif i >= 48 and i <= 57:
        return i+4
    elif i == 45:
        return 62
    elif i == 95:
        return 63
    
def _eval(c):
    return alphabet[c]

def is_base64(c):
    i = ord(c)
    if i >= 97 and i <= 122:
        return True
    elif i >= 65 and i <= 90:
        return True
    elif i >= 48 and i <= 57:
        return True
    elif i == 45:
        return True
    elif i == 95:
        return True
    else:
        return False


def base642bin(b):
    ''' Convert binary string to int64.
        then to int64 and then to string with base
        abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_
    '''
    value = ""
    for i in range(len(b)):
        value += int2bin6(_reval(b[i]))
    return value

def bin2base64(b):
    ''' Convert binary string to int64.
        then to int64 and then to string with base
        abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_
    '''
    value = ""
    # do euclidean division
    d, r = divmod(len(b), 6)
    if r != 0:
        value = _eval(bin2int6(b[:r]))

    for i in range(d):
        value += _eval(bin2int6(b[i*6+r:(i+1)*6+r]))
    return value


def float2base64(f):
    s = bin2base64(float2bin(f))

    # remove all the trailing a
    return s.rstrip("a")

def base642float(b:str):
    # add all the trailing a
    b = b + "a" * (11 - len(b))
    return bin2float(base642bin(b))

def parse(content, i = 0, level=0):

    array = []
    if (content[i] == "["):
        # print('\t'*level+"parse array at", i)
        i += 1
        last_comma = False
        while (content[i] != "]"):

            if (content[i] == "["):
                element, i = parse(content, i, level+1)
                array.append(element)
                i += 1
                last_comma = False
            
            elif (content[i] == ","):
                if (last_comma):
                    array.append(0)

                last_comma = True
                i += 1
                continue

            else:
                last_comma = False
                # we read the next 11 characters
                # and convert it to float
                end = i
                while (is_base64(content[end])):
                    end += 1
                array.append(base642float(content[i:end]))
                i = end
            # i += 1

        # print('\t'*level+"parse end at", i)

    return array, i







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



        


