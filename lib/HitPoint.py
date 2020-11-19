import numpy as np
from numpy.core.fromnumeric import shape
from tensorflow import keras
import pandas as pd
import math


def extract_feature(traj):
    feature = np.copy(traj)
    df = pd.DataFrame({'x': feature[:, 0], 'y': feature[:, 1]})
    df['x_1'] = df['x'].shift(1)
    df['y_1'] = df['y'].shift(1)
    df['x_2'] = df['x'].shift(2)
    df['y_2'] = df['y'].shift(2)
    df['x_3'] = df['x'].shift(3)
    df['y_3'] = df['y'].shift(3)
    df['x_4'] = df['x'].shift(4)
    df['y_4'] = df['y'].shift(4)
    df['x1'] = df['x'].shift(-1)
    df['y1'] = df['y'].shift(-1)
    df['x2'] = df['x'].shift(-2)
    df['y2'] = df['y'].shift(-2)
    df['x3'] = df['x'].shift(-3)
    df['y3'] = df['y'].shift(-3)
    df['x4'] = df['x'].shift(-4)
    df['y4'] = df['y'].shift(-4)
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
    df['dx_2_0'] = df['x'] - df['x_2']  # angle2
    df['dy_2_0'] = df['y'] - df['y_2']
    df['dx_3_0'] = df['x'] - df['x_3']  # angle3
    df['dy_3_0'] = df['y'] - df['y_3']
    df['dx_4_0'] = df['x'] - df['x_4']  # angle4
    df['dy_4_0'] = df['y'] - df['y_4']
    df['dx02'] = df['x2'] - df['x']  # angle2
    df['dy02'] = df['y2'] - df['y']
    df['dx03'] = df['x3'] - df['x']  # angle3
    df['dy03'] = df['y3'] - df['y']
    df['dx04'] = df['x4'] - df['x']  # angle4
    df['dy04'] = df['y4'] - df['y']
    df['mulX1'] = df['dx_1_0'] * df['dx01']
    df['mulY1'] = df['dy_1_0'] * df['dy01']
    df['mulX2'] = df['dx_2_0'] * df['dx02']
    df['mulY2'] = df['dy_2_0'] * df['dy02']
    df['mulX3'] = df['dx_3_0'] * df['dx03']
    df['mulY3'] = df['dy_3_0'] * df['dy03']
    df['mulX4'] = df['dx_4_0'] * df['dx04']
    df['mulY4'] = df['dy_4_0'] * df['dy04']
    df['difX1'] = df['dx_1_0'] - df['dx01']
    df['difY1'] = df['dy_1_0'] - df['dy01']
    df['difX2'] = df['dx_2_0'] - df['dx02']
    df['difY2'] = df['dy_2_0'] - df['dy02']
    df['difX3'] = df['dx_3_0'] - df['dx03']
    df['difY3'] = df['dy_3_0'] - df['dy03']
    df['difX4'] = df['dx_4_0'] - df['dx04']
    df['difY4'] = df['dy_4_0'] - df['dy04']
    df['x_dot1'] = (-df['dx_1_0'])*df['dx01']
    df['y_dot1'] = (-df['dy_1_0'])*df['dy01']
    df['Lvec11'] = (((df['dx_1_0'])**2)+((df['dy_1_0'])**2))**0.5
    df['Lvec21'] = (((df['dx01'])**2)+((df['dy01'])**2))**0.5
    df['cos_angle1'] = (df['x_dot1']+df['y_dot1'])/(df['Lvec11']*df['Lvec21'])
    # angle2_cos
    df['x_dot2'] = (-df['dx_2_0'])*df['dx02']
    df['y_dot2'] = (-df['dy_2_0'])*df['dy02']
    df['Lvec12'] = (((df['dx_2_0'])**2)+((df['dy_2_0'])**2))**0.5
    df['Lvec22'] = (((df['dx02'])**2)+((df['dy02'])**2))**0.5
    df['cos_angle2'] = (df['x_dot2']+df['y_dot2'])/(df['Lvec12']*df['Lvec22'])
    # angle3_cos
    df['x_dot3'] = (-df['dx_3_0'])*df['dx03']
    df['y_dot3'] = (-df['dy_3_0'])*df['dy03']
    df['Lvec13'] = (((df['dx_3_0'])**2)+((df['dy_3_0'])**2))**0.5
    df['Lvec23'] = (((df['dx03'])**2)+((df['dy03'])**2))**0.5
    df['cos_angle3'] = (df['x_dot3']+df['y_dot3'])/(df['Lvec13']*df['Lvec23'])
    # angle4_cos
    df['x_dot4'] = (-df['dx_4_0'])*df['dx04']
    df['y_dot4'] = (-df['dy_4_0'])*df['dy04']
    df['Lvec14'] = (((df['dx_4_0'])**2)+((df['dy_4_0'])**2))**0.5
    df['Lvec24'] = (((df['dx04'])**2)+((df['dy04'])**2))**0.5
    df['cos_angle4'] = (df['x_dot4']+df['y_dot4'])/(df['Lvec14']*df['Lvec24'])

    # distance
    df['sum_bf'] = df['Lvec11'] + df['Lvec12'] + df['Lvec13'] + df['Lvec14']
    df['sum_af'] = df['Lvec21'] + df['Lvec22'] + df['Lvec23'] + df['Lvec24']
    df = df.fillna(0)
    # angle
    df['angle1'] = (np.arccos(df['cos_angle1']))*360/2/np.pi
    df['angle2'] = (np.arccos(df['cos_angle2']))*360/2/np.pi
    df['angle3'] = (np.arccos(df['cos_angle3']))*360/2/np.pi
    df['angle4'] = (np.arccos(df['cos_angle4']))*360/2/np.pi
    df['angle_sum'] = df['angle1'] + df['angle2'] + df['angle3'] + df['angle4']

    df = df.fillna(0)
    feature = df.to_numpy()
    return feature


def classify(feature):
    model = keras.models.load_model("model/HitPoint.h5")
    y_pred = model.predict(feature)
    # pred = []
    hp = []
    threshold = 0.9
    itr = 0
    for y in y_pred:
        max = np.argmax(y)
        if y[max] >= threshold and max != 1:
            # pred.append(max)
            hp.append(itr)
        itr += 1
        # else:
        # pred.append(1)
    # ball_type = np.reshape(pred, (-1, 1))
    # return ball_type
    return hp


def get_dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def polyfit(front, back, size, deg):
    fit = np.zeros(size)
    x = np.array([0, 1, 2, 3 + size, 4 + size, 5 + size])
    y = np.zeros(6)
    y[:3] = front
    y[3:] = back
    z = np.polyfit(x, y, deg)
    p = np.poly1d(z)
    for i in range(size):
        fit[i] = p(3 + i)
    return fit


def smoothing(traj):
    stable = []
    num_frame = traj.shape[0]
    delta = np.zeros(num_frame)
    for i in range(num_frame - 1):
        delta[i] = get_dist(traj[i, 1:], traj[i + 1, 1:])

    for i in range(num_frame):
        if traj[i, 0] == 0:
            stable.append(0)
        else:
            if i == 0 or traj[i - 1, 0] == 0:
                stable.append(1)
            elif delta[i - 1] < 80:
                stable.append(stable[i - 1] + 1)
            else:
                stable.append(1)

    valid = False
    valid_count = 0
    for i in range(num_frame):
        k = num_frame - i - 1
        if not valid and stable[k] < 4:
            stable[k] = 0
        elif valid:
            stable[k] = 1
            valid_count -= 1
            if valid_count == 0:
                valid = False
        else:
            valid_count = stable[k]
            stable[k] = 1
            valid_count -= 1
            valid = True
    st_ball = -1
    for i in range(num_frame - 3):
        if np.sum(stable[i:i + 4]) == 4 and np.sum(delta[i:i + 3]) > 15:
            st_ball = i
            break

    lt_ball = -1
    for i in range(num_frame):
        if stable[num_frame - i - 1] == 1:
            lt_ball = num_frame - i - 1
            break
    for i in range(lt_ball, num_frame):
        traj[i, 1:] = traj[lt_ball, 1:]

    st_scan = -1
    for i in range(num_frame):
        if stable[i] == 1:
            st_scan = i
            break
    for i in range(st_scan):
        traj[i, 1:] = traj[st_scan, 1:]

    i = st_scan + 2
    back = -1
    front = -1
    while i < num_frame:
        if stable[i] == 1 and stable[i - 1] == 0:
            back = i
            size = back - front
            if(front >= st_ball):
                deg = 3
            else:
                deg = 1
            traj[front:back, 1] = polyfit(
                traj[front - 3: front, 1], traj[back:back + 3, 1], size, deg)
            traj[front:back, 2] = polyfit(
                traj[front - 3: front, 2], traj[back:back + 3, 2], size, deg)
        elif stable[i] == 0 and stable[i - 1] == 1:
            front = i
        i += 1

    return traj


def HitPoint(traj):
    traj = smoothing(traj)
    feature = extract_feature(traj[:, 1:])
    hp = classify(feature)
    return hp, traj[:, 1:]
