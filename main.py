# Import Library
import cv2
from keras.models import load_model
from tensorflow_addons.losses import TripletSemiHardLoss
import numpy as np
from retinaface import RetinaFace
import pandas as pd
import time
from sklearn.metrics.pairwise import cosine_similarity
import imutils

#########################################################################

model = load_model('Embedded.h5')
data_label = pd.read_csv('label.csv')
name = data_label['Name'].values



# Load dữ liệu
np.random.seed(2022)
y_data = np.load('npydata/label_data.npy')
x_data = np.load('npydata/img_data.npy')
test_vec = model.predict(x_data)
np.save('npydata/test_vec.npy', test_vec)
softmax = load_model('Softmax.h5')
def rec_bound(alpha, frame, area):
    w = area[2] - area[0]
    h = area[3] - area[1]

    x1 = 4*area[0]
    y1 = 4*area[1]
    x2 = 4*area[2]
    y2 = 4*area[3]

    cv2.rectangle(frame, (x1,y1), (x2, y2), (255, 255, 255), 1)

    cv2.rectangle(frame, (x1,y1), (x1+alpha, y1), (255, 255, 255), 4)
    cv2.rectangle(frame, (x1,y1), (x1, y1+alpha), (255, 255, 255), 4)

    cv2.rectangle(frame, (x2,y2), (x2-alpha, y2), (255, 255, 255), 4)
    cv2.rectangle(frame, (x2,y2), (x2, y2-alpha), (255, 255, 255), 4)

# Đưa dữ liệu ảnh về dạng chuẩn
def to_tensor(x):
    x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA)
    x = x.reshape((1, 224, 224, 3))
    return x

def CosineSimilarity(embedded_vec,test_vec,y):
    start = time.time()
    result_lst = []
    for i in range(test_vec.shape[0]):

        result = cosine_similarity(embedded_vec[i,:].reshape(1,-1),test_vec[i].reshape(1,-1))
        result_lst.append(result[0])
    idex = np.argmax(np.array(result_lst))
    percent = result_lst[idex]
    id = y[idex]
    end = time.time()
    print('Thời gian Cosine',end-start)
    return id,percent


def CosineSimilarityv2(embedded_vec,test_vec,y):
    start = time.time()
    result_lst = []
    a = np.zeros((test_vec.shape[0],1))

    for i in range(test_vec.shape[0]):
        a[i] = embedded_vec[i].reshape(1,-1) @ test_vec[i].reshape(-1,1) #1000,128 * 128,1000 = 1000*1000
    b = np.linalg.norm(embedded_vec,axis=1) *np.linalg.norm(test_vec,axis=1)#1000,1
    result = a/b.reshape(-1,1)
    #print(a.shape)
    #print(b.shape)
    idex = np.argmax(result)
    percent = result[idex]
    id = y[idex]
    end = time.time()
    #print('Thời gian Cosine',end-start)
    return id, percent

def algin(frame, right_eye,left_eye):
    start = time.time()
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx)) - 180
    rotated = imutils.rotate_bound(frame, -angle)
    end = time.time()
    #print('Thời gian xoay ảnh:',end-start)
    return rotated

def recognition():
    cap = cv2.VideoCapture('video_test/Phuc.mp4')
    while cap.isOpened():
        _,frame = cap.read()
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        count =0
        # Using GPU
        #gpu_frame = cv2.cuda_GpuMat()
        #frame = gpu_frame.upload(frame)
        if count < 5:

            small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)

            small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            obj = RetinaFace.detect_faces(small_frame)
            count += 1

            if type(obj) == dict:
                for key in obj.keys():
                    face = obj[key]

                    area = face['facial_area']
                    landmark = face['landmarks']
                    right_eye = np.array(landmark['right_eye'])
                    left_eye = np.array(landmark['left_eye'])

                    rec_bound(15, frame, area)

                    img = frame[4*area[1]+3:4*area[3]+3, 4*area[0]+3:4*area[2]+3,:]
                    img = algin(img, right_eye, left_eye)

                    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    #print(img.shape)
                    img = to_tensor(img)


                    embedded_vec = []
                    emb = model.predict(img)

                    #for i in range(x_data.shape[0]):
                    #    embedded_vec.append(emb)
                    embedded_vec = np.vstack([emb]*x_data.shape[0])
                    # Tạo bộ vector để test

                    #predict, percent = CosineSimilarityv2(embedded_vec, test_vec, y_data)


        
                    result = softmax.predict(emb.reshape((1, 1, 128)))
        
                    percent = np.max(result)
                    predict = np.argmax(result)
        

                    cv2.putText(frame, 'FRAME": '+ str(video_fps), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, [255, 255, 255], 1)
                    if percent>0.6:
                        cv2.putText(frame, str(round(percent,2)), (4*area[0], 4*area[1] - 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, [255, 255, 255], 1)
                        cv2.putText(frame, str(name[predict]), (4*area[0],4*area[1]-5),cv2.FONT_HERSHEY_COMPLEX,0.5,[255,255,255],1)
                    else:
                        cv2.putText(frame, 'Unknown', (4*area[0],4*area[1]-5),cv2.FONT_HERSHEY_COMPLEX,0.5,[255,255,255],1)

                    print(name[predict])
            #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            count = 0
            time.sleep(0.5)
        cv2.imshow('vid', frame)

        #time.sleep(0.5)
        if cv2.waitKey(50) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



recognition()