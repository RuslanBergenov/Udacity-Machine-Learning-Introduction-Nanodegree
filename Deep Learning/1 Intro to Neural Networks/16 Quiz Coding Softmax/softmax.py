import numpy as np
import math

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
#https://stackabuse.com/the-python-math-library/
def softmax(L):
    e = math.e

    total = 0
    
    for num in L:
        myExp = e ** num
        total = total + myExp
    
    result_list = []
    for num in L:
        result = e ** num / total
        result_list.append(result)
    return result_list
    


myList = [2,1,0]

print(softmax(myList))

# equals 1
print(sum(softmax(myList)))



udacityList=[5,6,7]

print(softmax(udacityList))