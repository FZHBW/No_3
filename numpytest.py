import numpy as np

array1=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]]).reshape(12,1)
array2=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]]).reshape(12,1)
array3=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]]).reshape(12,1)
array4=np.append(array1,array2,axis=1)
array4=np.append(array4,array3,axis=1).reshape(3,4,3)
print(array4)
print('Hello World')