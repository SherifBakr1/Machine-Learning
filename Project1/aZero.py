#Student Name: Sherif Bakr
#Student ID : 873624609

import pandas as pd
import numpy as np
import scipy
from scipy.stats import pearsonr, spearmanr
import A0_Utils as A0

## Question 1 - Basics

def add(a, b):
    if (isinstance(a, int) and isinstance(b, int)):
        return a+b
    
    if (isinstance(a, int) and isinstance(b, float)) or (isinstance(a, float) and isinstance(b, int)) or (isinstance(a, float) and isinstance(b, float)) :
        return a+b
    
    if (isinstance(a, str) and isinstance(b, int)) or (isinstance(a, str) and isinstance(b, float)):
        return a+str(b)
    
    if (isinstance(a, int) and isinstance(b, str)) or (isinstance(a, float) and isinstance(b, str)):
        return str(a)+b
    
    if (isinstance(a, str)) and (isinstance(b, str)):
        return a+b

    if (isinstance(a, list)) and (isinstance(b, list)):
        return a+b
    
    print("Error!")
    return None

def calcMyGrade(AssignmentScores, MidtermScores, PracticumScores, ICAScores, Weights):

    assignments = sum(AssignmentScores) / len(AssignmentScores) / 100.0
    midterms = sum(MidtermScores) / len(MidtermScores) / 100.0
    practicum = sum(PracticumScores) / len(PracticumScores) / 100.0
    activities = sum(ICAScores) / len(ICAScores) / 100.0

    assignments_weight, midterms_weight, practicums_weight, activities_weight = [w / 100.0 for w in Weights]

    weighted_average = ((assignments * assignments_weight +midterms * midterms_weight + practicum * practicums_weight + activities * activities_weight) * 100)

    return round(weighted_average,1)


## Question 2 - Classes
class node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.leftchild = None
        self.rightchild = None

    def getChildren(self):
        return [self.leftchild, self.rightchild]

    def getKey(self):
        return self.key

    def getValue(self):
        return self.value

    def assignLeftChild(self, child):
        self.leftchild = child

    def assignRightChild(self, child):
        self.rightchild = child

    def inOrderTraversal(self):
        result = []
        if self.leftchild:
            result += self.leftchild.inOrderTraversal()
        result.append(self.value)
        if self.rightchild:
            result += self.rightchild.inOrderTraversal()
        return result


class queue:
    # See Question 2b
    def __init__(self):
        self.items=[]
    
    def push(self, value):
        self.items.append(value)

    def pop(self):
        if not self.is_empty():
            return self.items.pop(0)
        
    def checkSize(self):
        return len(self.items)
    
    def is_empty(self):
        return len(self.items)==0
    


## Question 3 - Libraries

def generateMatrix(numRows, numcolumns, minVal, maxVal):
    m = np.random.uniform(minVal, maxVal, size=(numRows, numcolumns))
    return m

def multiplyMat(m1, m2):
    try:
        ans = np.dot(m1, m2)
        return ans
    except ValueError:
        print("Incompatible Matrices")
        return None


def statsTuple(a, b):
    try:
        sum_a= np.sum(a)
        mean_a= np.mean(a)
        min_a= np.min(a)
        max_a=np.max(a)
        sum_b=np.sum(b)
        mean_b=np.mean(b)
        min_b=np.min(b)
        max_b=np.max(b)

        r, x = pearsonr(a, b)
        rho, y = spearmanr(a, b)

        result = (sum_a, mean_a, min_a, max_a, sum_b, mean_b, min_b, max_b, r, rho)
        twoDecimalPlaceResult=tuple()
        for i in result:
            if isinstance(i, float):
                twoDecimalPlaceResult+=(round(i,2),)
            else:
                twoDecimalPlaceResult+=(i,)

        return twoDecimalPlaceResult
    except Exception as e:
        return None
    

def pandas_func(path):
    df = pd.read_csv(path, sep='\t')

    ListOfMeans=[]
    ListOfColumnNames=[]

    for columnName, dtype in df.dtypes.items():
        if pd.api.types.is_numeric_dtype(dtype):
            #print("true, this is just a test")
            mean= round(df[columnName].mean(),2)
            ListOfMeans.append(mean)
        else:
            #print("False, this is just a test")
            ListOfColumnNames.append(columnName)

    return ListOfMeans, ListOfColumnNames
    