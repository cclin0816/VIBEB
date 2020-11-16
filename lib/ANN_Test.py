#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from keras.models import load_model
from numpy import sum
#read in data using pandas
test_name = 'TrackNet_coordinates_denoise_datatransform.csv'
test_df = pd.read_csv(test_name)
test_df = test_df.drop(columns=['frame_id','series','visible'])
test_X = test_df


model = load_model('hitpoint_model.h5')

y_pred=model.predict(test_X)
y_pred_type = []
y_pred_score = []
prid = []
TH = 0
for i in range(len(y_pred)):
    if np.argmax(y_pred[i]) == 0 and y_pred[i][0] > TH:
        y_pred_type.append('dead')
        y_pred_score.append(y_pred[i][0])
        prid.append(1)
    elif np.argmax(y_pred[i]) == 1 and y_pred[i][1] > TH:
        y_pred_type.append('fly')
        y_pred_score.append(y_pred[i][1])
        prid.append(0)
    elif np.argmax(y_pred[i]) == 2 and y_pred[i][2] > TH:
        y_pred_type.append('hit')
        y_pred_score.append(y_pred[i][2])
        prid.append(1)
    elif np.argmax(y_pred[i]) == 3 and y_pred[i][3] > TH:
        y_pred_type.append('start')
        y_pred_score.append(y_pred[i][3])
        prid.append(1)
    else:
        y_pred_type.append('fly')
        y_pred_score.append('<TH')
        prid.append(0)

y_pred1 = pd.DataFrame(y_pred, columns=['dead1','fly1','hit1','start1'])

test_df2 = pd.read_csv(test_name)
frame_id = test_df2['frame_id']
series = test_df2['series']


prid2 = [0] *len(y_pred_type)

for i in range(5,len(y_pred_type)):
    prid2[0:6] = ['fly','fly','fly','fly','fly','fly']
    #check within 6 frames
    if 1 < sum(prid[i-5:i+1]) < 7 and prid[i]==1 and series[i]==1:
        n_zero = np.where(np.array(prid[i-5:i+1]))
        score = np.array(y_pred_score[i-5:i+1])
        maximum = np.where(score==max(score[n_zero]))
        label = y_pred_type[i-5:i+1]
        for k in range(len(prid[i-5:i+1])):
            if k == sum(maximum):
                label[k] = label[k]
            else:
                label[k] = 'fly'
        prid2[i-5:i+1] = label
    elif prid[i] == 1 and series[i]==1:
        prid2[i] = y_pred_type[i]
    elif prid[i] == 0 and series[i]==1:
        prid2[i] = 'fly'
    elif series[i]==0:
        prid2[i] = 'fly'

dist = test_df2['Lvec11']
Gnd = 5
prid3 = [0] * len(prid2)

for i in range(len(prid2)):
    if i in range(0,31) : prid3[i] = prid2[i]
    elif i in range(len(prid2)-33,len(prid2)) : prid3[i] = prid2[i]
    elif prid2[i]=='start' or prid2[i]=='hit' or prid2[i]=='dead':
        frame = frame_id[i]
        L_bf = []
        L_af = []
        for j in range(0,31):
            if frame_id[i-j] == frame -j and dist[i-j] < 2 and series[i-j] == 1:
                L_bf.insert(0,dist[i-j])
            elif frame_id[i+j] == frame +j and dist[i+j] <2 and series[i+j] == 1:
                L_af.insert(0,dist[i+j])
        if len(L_bf) > Gnd: 
            prid3[i] = 'start'
        elif len(L_af) > Gnd:
            prid3[i] = 'dead'
        else: 
            prid3[i] = prid2[i]
    else:
        prid3[i] = prid2[i]

test_df2['pred_type'] = y_pred_type
test_df2['score'] = y_pred_score
test_df2['check'] = prid
test_df2['shot_events'] = prid3
test_df2=pd.concat([test_df2,y_pred1],axis=1)
test_df2.to_csv(test_name[:-4]+'_result_analysis.csv',index=False)



