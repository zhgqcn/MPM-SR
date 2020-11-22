import os
 
 
def readname(filename):
    nameList = os.listdir(filename)
    return nameList

def readNumFromName(nameList):
    numList = []
    for name in nameList:
        pos = name.find('-')
        numList.append(name[:pos])
    return numList

 
if __name__ == "__main__":
    filePath = 'D:\\A_GraduationProject\\kaggle\\DataSet\\train_LR\\'
    nameList = readname(filePath)
    num_List = readNumFromName(nameList)
    

    filePath_remove = 'D:\\A_GraduationProject\\kaggle\\DataSet\\train_HR_2\\'
    name_remove = readname(filePath_remove)
    num_remove = readNumFromName(name_remove)
    
    ext = name_remove[0].find('-')
    last_name = name_remove[0][ext:]
    
    for num in num_remove:
        if num not in num_List:
            path = filePath_remove + num + last_name
            print(path)
            os.remove(path)
