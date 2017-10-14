import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import tensorflow.python.ops
import os
import glob
import scipy
from scipy import io
import time
from tensorflow.contrib import learn
import random
from joblib import Parallel, delayed
import numba



def _weights(layer, expected_layer_name, vgg_layers):
    """
    Return the weights and bias from the VGG model for a given layer.
    """
    # print(vgg_layers[0][layer][0][0][0])
    W = vgg_layers[0][layer][0][0][2][0][0]
    b = vgg_layers[0][layer][0][0][2][0][1]
    layer_name = vgg_layers[0][layer][0][0][0]
    # assert layer_name == expected_layer_name
    return W, b

def _relu(conv2d_layer):
    """
    Return the RELU function wrapped over a TensorFlow layer. Expects a
    Conv2d layer input.
    """
    return tf.nn.relu(conv2d_layer)

def _conv2d(prev_layer, layer, layer_name, vgg_layers):
    """
    Return the Conv2D layer using the weights, biases from the VGG
    model at 'layer'.
    """
    W, b = _weights(layer, layer_name, vgg_layers)
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    return tf.nn.conv2d(
        prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

def _conv2d_relu(prev_layer, layer, layer_name, vgg_layers):
    """
    Return the Conv2D + RELU layer using the weights, biases from the VGG
    model at 'layer'.
    """
    return _relu(_conv2d(prev_layer, layer, layer_name, vgg_layers))

def _avgpool(prev_layer):
    """
    Return the AveragePooling layer.
    """
    return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def load_vgg_model(path):
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224
    COLOR_CHANNELS = 3
 
    vgg = scipy.io.loadmat(path)

    vgg_layers = vgg['layers']

    # Constructs the graph model.
    graph = {}
    graph['input']   = tf.Variable(np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)), dtype = 'float32')
    graph['conv1_1']  = _conv2d_relu(graph['input'], 0, 'conv1_1', vgg_layers)
    graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2', vgg_layers)
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1', vgg_layers)
    graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2', vgg_layers)
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1', vgg_layers)
    graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2', vgg_layers)
    graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3', vgg_layers)
    graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4', vgg_layers)
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1', vgg_layers)
    graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2', vgg_layers)
    graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3', vgg_layers)
    graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4', vgg_layers)
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1', vgg_layers)
    graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2', vgg_layers)
    graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3', vgg_layers)
    graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4', vgg_layers)
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    return graph

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



@numba.jit
def norms(vector):
    return np.sqrt(np.dot(vector,vector))

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

    return ANormlizeMap, AdashNormlizedMap, BNormlizedMap, BdashNormlizedMap


@numba.jit
def getDiffSearchList(image, imY, imX, randomWalkAreaA, randomWalkAreaB, height, width):
    len_pyA = int(randomWalkAreaA[0])
    len_myA = -int(randomWalkAreaA[0])
    len_pxA = int(randomWalkAreaA[1])
    len_mxA = -int(randomWalkAreaA[1])

    len_pyB = int(randomWalkAreaB[0]) 
    len_myB = -int(randomWalkAreaB[0])
    len_pxB = int(randomWalkAreaB[1]) 
    len_mxB = -int(randomWalkAreaB[1])

    A=[[0,0]]
    for y in range(len_myA, len_pyA):
        tempY = imY+y
        for x in range(len_mxA, len_pxA):
            if 0 <= imX+x < width and 0 <= tempY < height:
                A.append([tempY,imX+x])
    B=[[0,0]]
    for y in range(len_myB, len_pyB):
        tempY = imY+y
        for x in range(len_mxB, len_pxB):
            if 0 <= imX+x < width and 0 <= tempY < height:
                B.append([tempY,imX+x])
    for item in B:
        if item in A:
            A.remove(item)
    return A

def randomSearch(A, ANormlizeMap, AdashNormlizedMap, imY, imX, BNormlizedMap, BdashNormlizedMap, randomWalkArea, patch, height, width):
    minimum = getDistance(A, ANormlizeMap, AdashNormlizedMap, imY, imX, BNormlizedMap, BdashNormlizedMap, patch, height, width)

    saveY=0
    saveX=0
    maxHeight= randomWalkArea[0]
    maxWidth = randomWalkArea[1]

    while maxHeight > 1:
        searchList = getDiffSearchList(BNormlizedMap, imY, imX, [maxHeight, maxWidth], [maxHeight//2, maxWidth//2], height, width)
        if searchList == []:
            return [saveY,saveX]

        for a in range(5):
            points = random.choice(searchList)

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
    
    ret = np.zeros_like(Phi)
    

    for y in range(height):
        for x in range(width):
            positionList=[Phi[y][x]]
            minimum = 100000

            if y>0:
                positionList.append(Phi[y-1][x])
            if x>0:
                positionList.append(Phi[y][x-1])
            saveX=0
            saveY=0
            
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
                        if 0 <= tempY+1 < height:
                            saveY = tempY+1
                            saveX = tempX
                        else:
                            saveY = tempY
                            saveX = tempX                            
                    elif i==2:
                        if 0 <= tempX+1 < width:
                            saveY = tempY
                            saveX = tempX+1
                        else:
                            saveY = tempY
                            saveX = tempX
            #一番値が近いエリアのみを保存
            minimum = 100000

            searchList = getSearchList(ANormlizeMap,saveY,saveX,randomWalkArea,height,width)
            for searchArea in searchList:
                tempY=searchArea[0]
                tempX=searchArea[1]
                dis = getDistance(A, ANormlizeMap,AdashNormlizedMap, tempY, tempX, BNormlizedMap,BdashNormlizedMap, patch, height, width)

                if minimum > dis:
                    minimum = dis
                    saveY = tempY
                    saveX = tempX
                ret[y][x]=[saveY,saveX]

            #     ret[y][x]=randomSearch(y,x, ANormlizeMap, AdashNormlizedMap, saveY, saveX, BNormlizedMap, BdashNormlizedMap, randomWalkArea, patch, height, width)
    return ret




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
def warp(image,Phi):
    ret = np.zeros_like(image)
    height = len(Phi)
    width = len(Phi[0])
    for y in range(height):
        for x in range(width):
            newY = int(Phi[y][x][0])
            newX = int(Phi[y][x][1])
            ret[0][y][x] = image[0][newY][newX]
    return ret

# @numba.jit
def makeFinImage(img,Phi):
    temp = np.zeros_like(img[0])
    height = len(img[0])
    width = len(img[0][0])
    for y in range(height):
        for x in range(width):
            positions = getPatchPosition(img[0], y, x, [5,5],height,width)
            sumX =0
            sumY=0
            R=0
            G=0
            B=0
            for pos in positions:
                if pos[0] != -1 or pos[1]!=-1:
                    sumY += Phi[pos[0]][pos[1]][0]
                    sumX += Phi[pos[0]][pos[1]][1]
                    R += img[0][int(Phi[pos[0]][pos[1]][0])][int(Phi[pos[0]][pos[1]][1])][0]
                    G += img[0][int(Phi[pos[0]][pos[1]][0])][int(Phi[pos[0]][pos[1]][1])][1]
                    B += img[0][int(Phi[pos[0]][pos[1]][0])][int(Phi[pos[0]][pos[1]][1])][2]
            sumY /= 25.0
            sumX /= 25.0

            temp[y][x] = img[0][int(sumY)][int(sumX)]
            temp[y][x][0] = int(R/25)
            temp[y][x][1] = int(G/25)
            temp[y][x][2] = int(B/25)

    return temp


@numba.jit
def getWeight(layer,a):
    k=300
    t=0.05

    height = len(layer[0])
    width = len(layer[0][0])
    layerChanged = np.zeros((height,width))   
    for y in range(height):
        for x in range(width):
            layerChanged[y][x] = np.dot(layer[0][y][x],layer[0][y][x])
    layer = W2tensor(layerChanged)
    M = np.zeros((height,width))

    for y in range(height):
        for x in range(width):
            f = layer[y][x]

            M[y][x]=a/(1 + np.exp( -k*(f-t)) )
    return M

@numba.jit
def weightBlend(F,W,R):
    OtherW = 1.0-W

    height=len(F[0])
    width =len(F[0][0])
    channel=len(F[0][0][0])
    for y in range(height):
        for x in range(width):
            for z in range(channel):
                F[0][y][x][z] = F[0][y][x][z]*W[y][x] + R[0][y][x][z]*OtherW[y][x]
    return F

def build_model(input_img,IMAGE_WIDTH,IMAGE_HEIGHT,CHANNEL, layer_num):

  VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
  net = {}
  vgg_rawnet     = scipy.io.loadmat(VGG_MODEL)
  # vgg_layers     = vgg_rawnet['layers'][0]
  vgg_layers = vgg_rawnet['layers']
  net['input']   = tf.Variable(np.asarray([input_img]), dtype=np.float32)
    # np.zeros((1,IMAGE_WIDTH,IMAGE_HEIGHT,CHANNEL)
  if layer_num==1:
    net['conv1_1']  = _conv2d_relu(net['input'], 0, 'conv1_1', vgg_layers)
    net['conv1_2']  = _conv2d_relu(net['conv1_1'], 2, 'conv1_2', vgg_layers)
    net['avgpool1'] = _avgpool(net['conv1_2'])
    net['conv2_1']  = _conv2d_relu(net['avgpool1'], 5, 'conv2_1', vgg_layers)

  if layer_num == 2:  
    net['conv2_1']  = _conv2d_relu(net['input'], 5, 'conv2_1', vgg_layers)
    net['conv2_2']  = _conv2d_relu(net['conv2_1'], 7, 'conv2_2', vgg_layers)
    net['avgpool2'] = _avgpool(net['conv2_2'])
    net['conv3_1']  = _conv2d_relu(net['avgpool2'], 10, 'conv3_1', vgg_layers)

  if layer_num == 3:
    net['conv3_1']  = _conv2d_relu(net['input'], 10, 'conv3_1', vgg_layers)
    net['conv3_2']  = _conv2d_relu(net['conv3_1'], 12, 'conv3_2', vgg_layers)
    net['conv3_3']  = _conv2d_relu(net['conv3_2'], 14, 'conv3_3', vgg_layers)
    net['conv3_4']  = _conv2d_relu(net['conv3_3'], 16, 'conv3_4', vgg_layers)
    net['avgpool3'] = _avgpool(net['conv3_4'])
    net['conv4_1']  = _conv2d_relu(net['avgpool3'], 19, 'conv4_1', vgg_layers)

  if layer_num == 4:
    net['conv4_1']  = _conv2d_relu(net['input'], 19, 'conv4_1', vgg_layers)
    net['conv4_2']  = _conv2d_relu(net['conv4_1'], 21, 'conv4_2', vgg_layers)
    net['conv4_3']  = _conv2d_relu(net['conv4_2'], 23, 'conv4_3', vgg_layers)
    net['conv4_4']  = _conv2d_relu(net['conv4_3'], 25, 'conv4_4', vgg_layers)
    net['avgpool4'] = _avgpool(net['conv4_4'])
    net['conv5_1']  = _conv2d_relu(net['avgpool4'], 28, 'conv5_1', vgg_layers)

  if layer_num == 5:
    net['conv5_1']  = _conv2d_relu(net['input'], 28, 'conv5_1', vgg_layers)
    net['conv5_2']  = _conv2d_relu(net['conv5_1'], 30, 'conv5_2', vgg_layers)
    net['conv5_3']  = _conv2d_relu(net['conv5_2'], 32, 'conv5_3', vgg_layers)
    net['conv5_4']  = _conv2d_relu(net['conv5_3'], 34, 'conv5_4', vgg_layers)
    net['avgpool5'] = _avgpool(net['conv5_4'])

  return net




@numba.jit
def createImage(tensor):
    newImage = np.zeros((len(tensor[0]),len(tensor[0][0]),3))
    for  z in range(len(tensor[0][0][0])):
        for y in range(len(tensor[0])):
          for x in range(len(tensor[0][y])):
            newImage[y][x][0] = tensor[0][y][x][z]
            newImage[y][x][1] = tensor[0][y][x][z]
            newImage[y][x][2] = tensor[0][y][x][z]
        newImage = np.asarray(newImage).astype('float64')
        newImage += 127.0#MEAN_VALUES[0]
        newImage = newImage[:, :, ::-1]
        newImage = np.clip(newImage, 0, 255).astype('uint8')
    return newImage


def minimize_with_lbfgs(sess, net, optimizer, init_img, goal_img):
  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  sess.run(net['input'].assign(init_img))
  optimizer.minimize(sess)

def minimize_with_adam(sess, net, optimizer, init_img, goal_img, loss, max_iterations, check_layer, output_layer, print_iterations):
  train_op = optimizer.minimize(loss)
  init_op = tf.global_variables_initializer()
  sess.run(init_op)

  iterations = 0
  while (iterations < max_iterations):
    sess.run(train_op)
    if iterations % print_iterations==0:
      curr_loss = loss.eval()
      print("At iterate {}\tf=  {:.5E}".format(iterations, curr_loss))

    iterations += 1


def getLoss(sess, vggModel, layer, generate_image, goal_image):

    sess.run(vggModel['input'].assign(generate_image))

    sub = tf.subtract(vggModel['y_'],vggModel[layer])
    abso= tf.norm(sub)
    return tf.pow(abso,2)


def get_optimizer(loss, select_optimizer, learning_rate, max_iterations, print_iterations):
  if select_optimizer == 'lbfgs':
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(
      loss, method='L-BFGS-B',
      options={'maxiter': max_iterations
              # ,'disp': print_iterations
                  })
  elif select_optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate)
  return optimizer



def newDeconv(sess, goal_img, select_optimizer, max_iterations, check_layer, output_layer, layer_num, channel):
    noise = generateNoiseImage(int(len(goal_img[0])*2),int(len(goal_img[0][0])*2),channel)
    noise = np.asarray(noise,dtype=np.float32)
    noise -= 127.0
    init_img = np.asarray([noise],dtype=np.float32)



    vggModel = build_model(noise,int(len(goal_img[0])/2),int(len(goal_img[0][0])/2),channel,layer_num)
    vggModel['y_'] = tf.constant(goal_img)

    L_total=getLoss(sess, vggModel, check_layer, init_img, goal_img)
    optimizer=get_optimizer(L_total,select_optimizer, 1e-0, max_iterations,1000)
    if select_optimizer == 'adam':
        minimize_with_adam(sess, vggModel, optimizer, init_img, goal_img, L_total,max_iterations,check_layer, output_layer,1000)
    elif select_optimizer == 'lbfgs':
        minimize_with_lbfgs(sess, vggModel, optimizer, init_img, goal_img)

    return sess.run(vggModel[output_layer])


def generateNoiseImage(height, width, channel):
    randomByteArray = bytearray(os.urandom(height*width*channel)) #画素数文の乱数発生
    return np.asarray(randomByteArray).reshape((height, width, channel))

def getMono(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


@numba.jit
def upsampling(Phi):
    temp = np.zeros((len(Phi),len(Phi[0]),3))

    for y in range(len(Phi)):
        for x in range(len(Phi[y])):
            temp[y][x] = [Phi[y][x][0],Phi[y][x][1],0]

    temp = cv2.resize(temp, None, fx=2, fy=2, interpolation= cv2.INTER_NEAREST)

    ret = np.zeros((len(temp),len(temp[0]),2))
    for y in range(len(temp)):
        for x in range(len(temp[y])):
            ret[y][x] = [temp[y][x][0]*2.0,temp[y][x][1]*2.0]
    return np.asarray(ret)


@numba.jit
def Phi2Image(Phi):
    ret = np.zeros((len(Phi),len(Phi[0]),3))
    for y in range(len(Phi)):
        for x in range(len(Phi[y])):
            ret[y][x] = [Phi[y][x][1],0,Phi[y][x][0]]
    ret = np.clip(ret,0,255)
    ret = np.asarray(ret,dtype='uint8')
    return ret


@numba.jit
def W2tensor(W):
    ave = (W.max()-W.min())
    mini= W.min()
    W -= mini
    if ave!=0:
        W /= ave
    return W

# if __name__ == '__main__':
def run(pathes):
    if len(pathes)!=2:
        return []
    MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

    VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    model = load_vgg_model(VGG_MODEL)

    imgA = cv2.resize(cv2.imread(pathes[0]), (224,224))
    img = np.asarray([imgA]).astype('float64')
    img -= MEAN_VALUES
    # img = img[:, :, ::-1].astype('float64')


    sess.run(model['input'].assign(np.asarray(img)))
    A=[]
    A.append(sess.run(model['conv1_1']))
    A.append(sess.run(model['conv2_1']))
    A.append(sess.run(model['conv3_1']))
    A.append(sess.run(model['conv4_1']))
    A.append(sess.run(model['conv5_1']))


    # imgB = cv2.imread(pathes[1])
    imgB = cv2.resize(cv2.imread(pathes[1]), (224,224))
    img = np.asarray([imgB]).astype('float64')
    img -= MEAN_VALUES
    # img = img[:, :, ::-1].astype('float64')
    # cv2.imshow("",createImage(img))
    # cv2.waitKey(0)


    sess.run(model['input'].assign(np.asarray(img)))
    B=[]
    B.append(sess.run(model['conv1_1']))
    B.append(sess.run(model['conv2_1']))
    B.append(sess.run(model['conv3_1']))
    B.append(sess.run(model['conv4_1']))
    B.append(sess.run(model['conv5_1']))


    a = [0.8, 0.7, 0.6, 0.1]
    channel=[256,128,64,3]
    # check_layers=['relu5_1','relu4_1','relu3_1','relu2_1','relu1_1']
    check_layers=['conv5_1','conv4_1','conv3_1','conv2_1','conv1_1']
    patch=[[5,5],[5,5],[3,3],[3,3],[3,3]]
    randomWalkArea=[[9,9],[9,9],[9,9],[13,13],[13*3,13*3]]

    PhiAB = getPhi_Random(A[5-1])
    PhiBA = getPhi_Random(B[5-1])

    ######
    # i=0
    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # PhiAB = patchMatch(A[5-1-i],  A[5-1-i],B[5-1-i],B[5-1-i],  randomWalkArea[5-1-i], patch[5-1-i], PhiAB)
    # # PhiAB = upsampling(PhiAB)
    # # PhiAB = upsampling(PhiAB)
    # # PhiAB = upsampling(PhiAB)
    # # PhiAB = upsampling(PhiAB)

    # pic = Phi2Image(PhiAB)
    # pic = np.asarray(pic*14, dtype='uint8')
    # cv2.imshow("img",pic)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(pic)
    #######

    FB    = None
    FAdash= None
    for i in range(5):
        print("PhiAB:{0}".format(PhiAB.shape))
        print("PhiBA:{0}".format(PhiBA.shape))
        if i==0:
            # for aaa in range(5):

            start = time.time()
            PhiAB = patchMatchA(A[5-1-i], A[5-1-i], B[5-1-i],B[5-1-i],  randomWalkArea[5-1-i], patch[5-1-i], PhiAB,i)
            elapsed_time = time.time() - start
            print ("2_time:{0}".format(elapsed_time) + "[sec]")
            start = time.time()
            PhiBA = patchMatchA(B[5-1-i], B[5-1-i], A[5-1-i],A[5-1-i],  randomWalkArea[5-1-i], patch[5-1-i], PhiBA,i)
            elapsed_time = time.time() - start
            print ("2_time:{0}".format(elapsed_time) + "[sec]")
        else:
            # PhiAB = getPhi_L_MtoN(A[5-1-i], FB, patch[5-1-i])
            # PhiBA = getPhi_L_MtoN(B[5-1-i], FADash, patch[5-1-i])
            for aaa in range(5):
                start = time.time()
                # PhiAB = patchMatchA(A[5-1-i], A[5-1-i], B[5-1-i],B[5-1-i],  randomWalkArea[5-1-i], patch[5-1-i], PhiAB)
                # for j in range(3):
                PhiAB = patchMatchA(A[5-1-i], FAdash, FB, B[5-1-i], randomWalkArea[5-1-i], patch[5-1-i], PhiAB,i)
                elapsed_time = time.time() - start
                print ("2_time:{0}".format(elapsed_time) + "[sec]")
                start = time.time()
                # PhiBA = patchMatchA(B[5-1-i], B[5-1-i], A[5-1-i],A[5-1-i],  randomWalkArea[5-1-i], patch[5-1-i], PhiBA)
                # for j in range(3):
                PhiBA = patchMatchA(FB, B[5-1-i], A[5-1-i], FAdash, randomWalkArea[5-1-i], patch[5-1-i], PhiBA,i)
                elapsed_time = time.time() - start
                print ("2_time:{0}".format(elapsed_time) + "[sec]")


        if i<4:
            Adash = warp(A[5-1-i],PhiBA)
            Bdash = warp(B[5-1-i],PhiAB)

            RA    = newDeconv(sess, Adash, 'adam', 1001, check_layers[i], check_layers[i+1], 5-1-i,channel[i]) 
            RB    = newDeconv(sess, Bdash, 'adam', 1001, check_layers[i], check_layers[i+1], 5-1-i,channel[i]) 

            WA    = getWeight(A[5-2-i],a[i])
            WB    = getWeight(B[5-2-i],a[i])
            # WA=W2tensor(WA)
            # WB=W2tensor(WB)
            # print(WA)

            print("RB:{0}".format(RB.shape))
            print("A:{0}".format(A[5-2-i].shape))
            print("WA:{0}".format(WA.shape))

            FAdash= weightBlend(A[5-2-i],WA, RB) 
            FB    = weightBlend(B[5-2-i],WB, RA)

            PhiAB = upsampling(PhiAB)
            PhiBA = upsampling(PhiBA)
        # cv2.imwrite("A"+str(i)+".jpg",createImage(A[5-1-i]))
        # cv2.imwrite("B"+str(i)+".jpg",createImage(B[5-1-i]))
        # cv2.imwrite("warpA"+str(i)+".jpg",createImage(Adash))
        # cv2.imwrite("warpB"+str(i)+".jpg",createImage(Bdash))
        # cv2.imwrite("PhiA"+str(i)+".jpg",Phi2Image(PhiAB))
        # cv2.imwrite("PhiB"+str(i)+".jpg",Phi2Image(PhiBA)) 
        # cv2.imwrite("RA"+str(i)+".jpg",createImage(RA))
        # cv2.imwrite("RB"+str(i)+".jpg",createImage(RB)) 
        # cv2.imwrite("WA"+str(i)+".jpg",WA*255)
        # cv2.imwrite("WB"+str(i)+".jpg",WB*255) 
        # cv2.imwrite("FAdash"+str(i)+".jpg",createImage(FAdash))
        # cv2.imwrite("FB"+str(i)+".jpg",createImage(FB)) 

    # tempA = warp(np.asarray([imgA]),PhiBA)  
    # tempB = warp(np.asarray([imgB]),PhiAB)
    A=makeFinImage(np.asarray([imgB]), PhiAB)
    B=makeFinImage(np.asarray([imgA]), PhiBA)
    # print(A.shape)
    # print(B.shape)
    # print(A)
    # print(B)
    # np.divide(temp,25.0)
    # A += MEAN_VALUES
    # B += MEAN_VALUES
    cv2.imwrite("A.jpg",A)
    cv2.imwrite("B.jpg",B) 
    # cv2.imwrite("B________.jpg",np.asarray(tempB[0],dtype='uint8'))

    # cv2.imshow("A",np.asarray(A,dtype='uint8'))
    # cv2.waitKey(0)
    # cv2.imshow("B",np.asarray(B,dtype='uint8'))
    # cv2.waitKey(0)


    sess.close()
    return ["A.jpg","B.jpg"]