#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np

test_name = 'TrackNet_coordinates_denoise.csv'
df = pd.read_csv(test_name)

import numpy as np

# coordinates shift
df['x_1'] = df['x'].shift(1) #X下移一列
df['y_1'] = df['y'].shift(1) #Y下移一列
df['x_2'] = df['x'].shift(2) #X下移二列
df['y_2'] = df['y'].shift(2) #Y下移二列
df['x_3'] = df['x'].shift(3) #X下移三列
df['y_3'] = df['y'].shift(3) #Y下移三列
df['x_4'] = df['x'].shift(4) 
df['y_4'] = df['y'].shift(4) 
df['x1'] = df['x'].shift(-1) #X上移一列
df['y1'] = df['y'].shift(-1) #Y上移一列
df['x2'] = df['x'].shift(-2) #X上移二列
df['y2'] = df['y'].shift(-2) #Y上移二列
df['x3'] = df['x'].shift(-3) #X上移三列
df['y3'] = df['y'].shift(-3) #Y上移三列
df['x4'] = df['x'].shift(-4) 
df['y4'] = df['y'].shift(-4) 

# features
#vector
df['dx_4_3'] = df['x_3'] - df['x_4']
df['dy_4_3'] = df['y_3'] - df['y_4']
df['dx_3_2'] = df['x_2'] - df['x_3']
df['dy_3_2'] = df['y_2'] - df['y_3']
df['dx_2_1'] = df['x_1'] - df['x_2']
df['dy_2_1'] = df['y_1'] - df['y_2']
df['dx_1_0'] = df['x'] - df['x_1']
df['dy_1_0'] = df['y'] - df['y_1']
df['dx01'] = df['x1'] - df['x']
df['dy01'] = df['y1'] - df['y']
df['dx12'] = df['x2'] - df['x1']
df['dy12'] = df['y2'] - df['y1']
df['dx23'] = df['x3'] - df['x2']
df['dy23'] = df['y3'] - df['y2']
df['dx34'] = df['x4'] - df['x3']
df['dy34'] = df['y4'] - df['y3']

df['dx_2_0'] = df['x'] - df['x_2'] #angle2
df['dy_2_0'] = df['y'] - df['y_2']
df['dx_3_0'] = df['x'] - df['x_3'] #angle3
df['dy_3_0'] = df['y'] - df['y_3']
df['dx_4_0'] = df['x'] - df['x_4'] #angle4
df['dy_4_0'] = df['y'] - df['y_4']
df['dx02'] = df['x2'] - df['x'] #angle2
df['dy02'] = df['y2'] - df['y']
df['dx03'] = df['x3'] - df['x'] #angle3
df['dy03'] = df['y3'] - df['y']
df['dx04'] = df['x4'] - df['x'] #angle4
df['dy04'] = df['y4'] - df['y']

#mulitiplication
df['mulX1'] = df['dx_1_0'] * df['dx01']
df['mulY1'] = df['dy_1_0'] * df['dy01']
df['mulX2'] = df['dx_2_0'] * df['dx02']
df['mulY2'] = df['dy_2_0'] * df['dy02']
df['mulX3'] = df['dx_3_0'] * df['dx03']
df['mulY3'] = df['dy_3_0'] * df['dy03']
df['mulX4'] = df['dx_4_0'] * df['dx04']
df['mulY4'] = df['dy_4_0'] * df['dy04']

#dif
df['difX1'] = df['dx_1_0'] - df['dx01']
df['difY1'] = df['dy_1_0'] - df['dy01']
df['difX2'] = df['dx_2_0'] - df['dx02']
df['difY2'] = df['dy_2_0'] - df['dy02']
df['difX3'] = df['dx_3_0'] - df['dx03']
df['difY3'] = df['dy_3_0'] - df['dy03']
df['difX4'] = df['dx_4_0'] - df['dx04']
df['difY4'] = df['dy_4_0'] - df['dy04']


#cos_angle=x.dot(y)/(Lx*Ly)
#angle1_cos
df['x_dot1'] = (-df['dx_1_0'])*df['dx01']
df['y_dot1'] = (-df['dy_1_0'])*df['dy01']
df['Lvec11'] = (((df['dx_1_0'])**2)+((df['dy_1_0'])**2))**0.5
df['Lvec21'] = (((df['dx01'])**2)+((df['dy01'])**2))**0.5
df['cos_angle1'] = (df['x_dot1']+df['y_dot1'])/(df['Lvec11']*df['Lvec21'])
#angle2_cos
df['x_dot2'] = (-df['dx_2_0'])*df['dx02']
df['y_dot2'] = (-df['dy_2_0'])*df['dy02']
df['Lvec12'] = (((df['dx_2_0'])**2)+((df['dy_2_0'])**2))**0.5
df['Lvec22'] = (((df['dx02'])**2)+((df['dy02'])**2))**0.5
df['cos_angle2'] = (df['x_dot2']+df['y_dot2'])/(df['Lvec12']*df['Lvec22'])
#angle3_cos
df['x_dot3'] = (-df['dx_3_0'])*df['dx03']
df['y_dot3'] = (-df['dy_3_0'])*df['dy03']
df['Lvec13'] = (((df['dx_3_0'])**2)+((df['dy_3_0'])**2))**0.5
df['Lvec23'] = (((df['dx03'])**2)+((df['dy03'])**2))**0.5
df['cos_angle3'] = (df['x_dot3']+df['y_dot3'])/(df['Lvec13']*df['Lvec23'])
#angle4_cos
df['x_dot4'] = (-df['dx_4_0'])*df['dx04']
df['y_dot4'] = (-df['dy_4_0'])*df['dy04']
df['Lvec14'] = (((df['dx_4_0'])**2)+((df['dy_4_0'])**2))**0.5
df['Lvec24'] = (((df['dx04'])**2)+((df['dy04'])**2))**0.5
df['cos_angle4'] = (df['x_dot4']+df['y_dot4'])/(df['Lvec14']*df['Lvec24'])

#distance
df['sum_bf'] = df['Lvec11'] + df['Lvec12'] + df['Lvec13'] + df['Lvec14']
df['sum_af'] = df['Lvec21'] + df['Lvec22'] + df['Lvec23'] + df['Lvec24']

#angle
df['angle1'] = (np.arccos(df['cos_angle1']))*360/2/np.pi
df['angle2'] = (np.arccos(df['cos_angle2']))*360/2/np.pi
df['angle3'] = (np.arccos(df['cos_angle3']))*360/2/np.pi
df['angle4'] = (np.arccos(df['cos_angle4']))*360/2/np.pi
df['angle_sum'] = df['angle1'] + df['angle2'] +df['angle3'] +df['angle4']

df = df.fillna(0)

df.to_csv(test_name[:-4]+'_datatransform.csv',index=False)





