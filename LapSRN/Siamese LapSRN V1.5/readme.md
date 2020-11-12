# 在原来网络基础上，利用了新的数据集进行训练
**新的数据集是经过挑选的，即剔除不好的图，只选择好的**
- 挑选28张，切成448张作为训练集
- 挑选2张，切成32张作为训练集

## 训练集如下：
![Snipaste_2020-11-12_18-11-30](https://tva3.sinaimg.cn/large/005tpOh1ly1gkmjyvnpuwj31bu0h248m.jpg)

## 测试集如下：
![Snipaste_2020-11-12_18-12-25](https://tvax3.sinaimg.cn/large/005tpOh1ly1gkmjz2aoanj308j04ndg4.jpg)

## 为了方便数据挑选，即使每个文件夹下相同，写了代码`select.py`
```python
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
```
