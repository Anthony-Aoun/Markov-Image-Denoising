import utils
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

img_names = ['beee2', 'country2', 'zebre2']

# get image
img_name = img_names[2] # SELECT IMAGE
img = cv.cvtColor(cv.imread('./Images/'+img_name+'.bmp'),cv.COLOR_BGR2GRAY)

# define parameters
param1 = [0,3,1,2] # [m1, m2, sig1, sig2]
param2 = [1,1,1,5] # [m1, m2, sig1, sig2]
param3 = [0,1,1,1] # [m1, m2, sig1, sig2]
params = [param1, param2, param3]

def compute(param):
    # 2D to 1D
    signal = utils.peano_transform_img(img)

    # bruitage
    Y = utils.bruit_gauss2(signal,0,255,param[0],param[2],param[1],param[3])
    img_noise = utils.transform_peano_in_img(Y, img.shape[0])

    # kmean
    kmeans = KMeans(n_clusters=2, random_state=0).fit(Y.reshape(-1,1))
    pred = kmeans.labels_

    m0 = []
    m1 = []

    for i in range(len(pred)):
        if (pred[i] == 0):
            m0 += [Y[i]]
        else:
            m1 += [Y[i]]

    m0 = np.array(m0)
    m1 = np.array(m1)

    # parameters
    m1, sig1, m2, sig2 = np.mean(m0),np.std(m0), np.mean(m1), np.std(m1)
    p1,p2 = utils.calc_probaprio2(signal,0,255)
    A = utils.calc_probatrans2(signal,0,255)
    p1_f, p2_f, A_f, m1_f, sig1_f, m2_f, sig2_f = utils.estim_param_EM_mc(10, Y, A, p1, p2, m1, sig1, m2, sig2)

    # segmentation
    Mat_f = utils.gauss2(Y,len(Y),m1_f,sig1_f,m2_f,sig2_f)
    X = utils.MPM_chaines2(Mat_f,len(Y),0,255,A_f,p1_f,p2_f)
    img_segmented = utils.transform_peano_in_img(X, img.shape[0])

    #error
    error = utils.taux_erreur(X,signal)

    return error, img_noise, img_segmented

# MAIN
images = []
errors = []

# stacking images
for ind, param in enumerate(params):
    # computation
    error, img_noise, img_segmented = compute(param)
    while(error > 0.5):
        error, img_noise, img_segmented = compute(param)

    # error
    print("Taux d'erreur pour param",ind+1," : ",error)
    
    images.append([img, img_noise, img_segmented])
    errors.append(error)

# vizualising results
plt.figure(figsize=(15,15))
# param1
img, img_noise, img_segmented = images[0][0], images[0][1], images[0][2]
plt.subplot(3,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.xticks([])
plt.yticks([])
plt.ylabel("param1")
plt.subplot(3,3,2)
plt.imshow(img_noise, cmap='gray')
plt.title("Noised")
plt.xticks([])
plt.yticks([])
plt.subplot(3,3,3)
plt.imshow(img_segmented, cmap='gray')
plt.title("Segmented")
plt.xticks([])
plt.yticks([])
plt.xlabel("Taux erreur : "+str(errors[0]))
# param2
img, img_noise, img_segmented = images[1][0], images[1][1], images[1][2]
plt.subplot(3,3,4)
plt.imshow(img, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.ylabel("param2")
plt.subplot(3,3,5)
plt.imshow(img_noise, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.subplot(3,3,6)
plt.imshow(img_segmented, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.xlabel("Taux erreur : "+str(errors[1]))
# param3
img, img_noise, img_segmented = images[2][0], images[2][1], images[2][2]
plt.subplot(3,3,7)
plt.imshow(img, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.ylabel("param3")
plt.subplot(3,3,8)
plt.imshow(img_noise, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.subplot(3,3,9)
plt.imshow(img_segmented, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.xlabel("Taux erreur : "+str(errors[2]))

plt.suptitle('Image : '+img_name)
plt.savefig('./ResultsMC/'+img_name+'.png')
plt.show()


