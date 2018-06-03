import random
import numpy as np
import cv2
from tqdm import tqdm
import numba

@numba.jit
def norms(vector):
    return np.dot(vector,vector)
def getNormalizedFx(Fx,y,x):
    vector = Fx[0][y][x]
    return vector/norms(vector)

def getPatchPosition(image, imY, imX, patch,height,width):
    points=[]
    len_mx1 = -int(patch[0]/2)
    len_px1 = int(patch[0]/2)+1
    len_mx2 = -int(patch[1]/2)
    len_px2 = int(patch[1]/2)+1
    for x in range(len_mx1,len_px1):
        tempX = imX+x
        for y in range(len_mx2,len_px2):
            if 0 <= tempX < width and 0<= imY+y < height:
                points.append([imY+y,tempX])
            else:
                points.append([-1,-1])
    return points

#指定された位置の半径(randomwalkarea)分の有効な座標を返す
def getSearchList(image, imY, imX, randomWalkArea,height,width):
    points=[]
    len_mx1 = -int(randomWalkArea[0]/2)
    len_px1 = int(randomWalkArea[0]/2)+1
    len_mx2 = -int(randomWalkArea[1]/2)
    len_px2 = int(randomWalkArea[1]/2)+1
    for x in range(len_mx1,len_px1):
        tempX = imX+x
        for y in range(len_mx2,len_px2):
            if 0 <= tempX < width and 0<= imY+y < height:
                points.append([imY+y,tempX])
    map(int,points)
    return points


#指定された位置の半径(randomwalkarea)分の有効な座標を返す
def getSearchPosition(image, imY, imX, randomWalkArea,height,width):
    len_mx1 = -int(randomWalkArea[0]/2)
    len_px1 = int(randomWalkArea[0]/2)+1
    len_mx2 = -int(randomWalkArea[1]/2)
    len_px2 = int(randomWalkArea[1]/2)+1
    
#     serchX =  random.choice(list(range(max(imX+len_mx1, 0),min(imX+len_px1,height))))
#     serchY =  random.choice(list(range(max(imY+len_mx2, 0),min(imY+len_px2,width ))))
    serchX =  np.random.randint(max(imX+len_mx1, 0),min(imX+len_px1,height))
    serchY =  np.random.randint(max(imY+len_mx2, 0),min(imY+len_px2,width ))
    return [int(serchY),int(serchX)]

def randomSearch(A, ANormlizeMap, AdashNormlizedMap, imY, imX, BNormlizedMap, BdashNormlizedMap, randomWalkArea, patch, height, width):
    minimum = getDistance(A, ANormlizeMap, AdashNormlizedMap, imY, imX, BNormlizedMap, BdashNormlizedMap, patch, height, width)

    saveY=imY
    saveX=imX
    maxHeight= randomWalkArea[0]
    maxWidth = randomWalkArea[1]

    while maxHeight > 1:
#         searchList = getSearchList(BNormlizedMap, imY, imX, [maxHeight, maxWidth], height, width)
        points=getSearchPosition(BNormlizedMap, imY, imX, [maxHeight, maxWidth], height, width)
#         searchList = getDiffSearchList(BNormlizedMap, imY, imX, [maxHeight, maxWidth], [maxHeight//2, maxWidth//2], height, width)
#         if searchList == []:
#             return [saveY,saveX]

#         for a in range(5):
#         points = random.choice(searchList)

        tempY = points[0]
        tempX = points[1]

        dis = getDistance(A, ANormlizeMap, AdashNormlizedMap, tempY, tempX, BNormlizedMap, BdashNormlizedMap, patch, height, width)

        if minimum > dis:
            minimum = dis
            saveY = tempY
            saveX = tempX
        maxHeight //=2
        maxWidth  //=2
    return [saveY,saveX]

@numba.jit
def patchMatchA(imgA, imgAdash, imgB, imgBdash, randomWalkArea, patch, Phi,i):
    height=len(imgA[0])
    width =len(imgA[0][0])
    ANormlizeMap, AdashNormlizedMap, BNormlizedMap, BdashNormlizedMap = getNormlizedMap(imgA, imgAdash, imgB, imgBdash, patch, Phi)
    
#     ret = np.zeros_like(Phi)
    

    for y in tqdm(range(height)):
        for x in range(width):
            positionList=[Phi[y][x]]
            minimum = 100000

            if y>0:
                positionList.append(Phi[y-1][x])
            if x>0:
                positionList.append(Phi[y][x-1])
            saveX=x
            saveY=y
            
            tempY=0
            tempX=0
            A = getPatchPosition(ANormlizeMap,y, x, patch,height,width)

            #一番値が近いエリアを探索
            for i,pos in enumerate(positionList):
                tempY = int(pos[0])
                tempX = int(pos[1])

                dis = getDistance(A, ANormlizeMap,AdashNormlizedMap, tempY, tempX, BNormlizedMap,BdashNormlizedMap, patch, height,width)
                if minimum > dis:
                    minimum = dis
                    if i==0:
                        saveY = tempY
                        saveX = tempX
                    elif i==1:
                        if 0 <= tempY-1 < height:
                            saveY = tempY-1
                            saveX = tempX
                        else:
                            saveY = tempY
                            saveX = tempX                            
                    elif i==2:
                        if 0 <= tempX-1 < width:
                            saveY = tempY
                            saveX = tempX-1
                        else:
                            saveY = tempY
                            saveX = tempX
            #一番値が近いエリアのみを保存
            minimum = 100000

#             searchList = getSearchList(ANormlizeMap,saveY,saveX,randomWalkArea,height,width)
#             for searchArea in searchList:
#                 tempY=searchArea[0]
#                 tempX=searchArea[1]
#                 dis = getDistance(A, ANormlizeMap,AdashNormlizedMap, tempY, tempX, BNormlizedMap,BdashNormlizedMap, patch, height, width)

#                 if minimum > dis:
#                     minimum = dis
#                     saveY = tempY
#                     saveX = tempX
#                 phi[y][x]=[saveY,saveX]

            Phi[y][x]=randomSearch(A, ANormlizeMap, AdashNormlizedMap, saveY,saveX, BNormlizedMap, BdashNormlizedMap, randomWalkArea, patch, height, width)

#             Phi[y][x]=randomSearch(A, ANormlizeMap, AdashNormlizedMap, Phi[saveY][saveX][0],Phi[saveY][saveX][1], BNormlizedMap, BdashNormlizedMap, randomWalkArea, patch, height, width)
    return Phi





@numba.jit
def getNormlizedMap(imgA, imgAdash, imgB, imgBdash, patch, Phi):
    lengthY = len(imgA[0])
    lengthX = len(imgA[0][0])
    lengthZ = len(imgA[0][0][0])
    ANormlizeMap=np.zeros((lengthY,lengthX,lengthZ))
    AdashNormlizedMap=np.zeros((lengthY,lengthX,lengthZ))
    BNormlizedMap=np.zeros((lengthY,lengthX,lengthZ))
    BdashNormlizedMap=np.zeros((lengthY,lengthX,lengthZ))

    for y in range(lengthY):
        for x in range(lengthX):
            ANormlizeMap[y][x]     = getNormalizedFx(imgA,y,x)
            AdashNormlizedMap[y][x]= getNormalizedFx(imgAdash,y,x)
            BNormlizedMap[y][x]    = getNormalizedFx(imgB, y,x)
            BdashNormlizedMap[y][x]= getNormalizedFx(imgBdash,y,x)
#     ANormlizeMap = imgA[0]/np.linalg.norm(imgA[0],ord=2,axis=(2),keepdims=True)
#     AdashNormlizedMap = imgAdash[0]/np.linalg.norm(imgAdash[0],ord=2,axis=(2),keepdims=True)
#     BNormlizedMap =imgB[0]/np.linalg.norm(imgB[0],ord=2,axis=(2),keepdims=True)
#     BdashNormlizedMap = imgBdash[0]/np.linalg.norm(imgBdash[0],ord=2,axis=(2),keepdims=True)
    
    return ANormlizeMap, AdashNormlizedMap, BNormlizedMap, BdashNormlizedMap


#patchmatchで用いるエネルギー関数を出力する
def getDistance(A,ANormlizeMap,AdashNormlizedMap,By,Bx,BNormlizedMap,BdashNormlizedMap,patch,height,width):
    B = getPatchPosition(BNormlizedMap,By,Bx,patch,height,width)
    sumationA = 0
    sumationB = 0
    length = len(A)
    for i in range(length):
        addressAY = A[i][0]
        addressAX = A[i][1]
        addressBY = B[i][0]
        addressBX = B[i][1]
        tempA=0
        tempB=0

        if addressAX!=-1 and addressBX!=-1:
            tempA = norms(
                                    (ANormlizeMap[addressAY][addressAX])
                                   -(BNormlizedMap[addressBY][addressBX])
                                   )
            tempB = norms(
                                    (AdashNormlizedMap[addressAY][addressAX])
                                   -(BdashNormlizedMap[addressBY][addressBX])
                                   )
        elif addressAX == -1 and addressBY != -1:
            tempA = norms(
                                   -(BNormlizedMap[addressBY][addressBX])
                                   )
            tempB = norms(
                                   -(BdashNormlizedMap[addressBY][addressBX])
                                   )
        elif addressAX != -1 and addressBY == -1:
            tempA = norms(ANormlizeMap[addressAY][addressAX])
            tempB = norms(AdashNormlizedMap[addressAY][addressAX])
        sumationA += tempA*tempA
        sumationB += tempB*tempB
    return sumationA+sumationB





def getPhi_Random(image):
    ret = []
    for y in range(len(image[0])):
        for x in range(len(image[0][y])):
            ret.append([y,x])
    random.shuffle(ret)
    ret = np.asarray(ret)
    randomRet = np.reshape(ret,(len(image[0]),len(image[0][0]),2))
    return randomRet

@numba.jit
def Phi2Image(Phi):
    ret = np.zeros((len(Phi),len(Phi[0]),3))
    for y in range(len(Phi)):
        for x in range(len(Phi[y])):
            ret[y][x] = [Phi[y][x][1],0,Phi[y][x][0]]
    ret = np.clip(ret,0,255)
    ret = np.asarray(ret,dtype='uint8')
    return ret

A=cv2.imread("testA.jpg")
B=cv2.imread("testB.jpg")
phi=getPhi_Random([A])


for i in range(5):
    phi=patchMatchA([A], [B], [B], [B], [200,200], [5,5], phi,0)
    cv2.imwrite(str(i)+".jpg",Phi2Image(phi))
cv2.imshow("",Phi2Image(phi))
cv2.waitKey(0)