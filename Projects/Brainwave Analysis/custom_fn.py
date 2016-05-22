import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()


def loadData():
    # Load Data
    X1, y1, m1 = loadDataP1()  # YHY
    X2, y2, m2 = loadDataP2()  # YHS
    X3, y3, m3 = loadDataP3()  # CU
    X4, y4, m4 = loadDataP4()  # US

    # Integrate Data
    X = pd.concat([X1, X2, X3, X4], axis=0)
    y = np.concatenate((y1, y2, y3, y4), axis=0)
    m_vec = np.concatenate((m1, m2, m3, m4), axis=0)

    # Reset indices
    X = X.reset_index(drop=True)

    return X, y, m_vec


def loadDataP1():
    X01 = pd.read_csv('yhy_150801_1300_1350.csv')
    X02 = pd.read_csv('yhy_150801_1400_1450.csv')
    X03 = pd.read_csv('yhy_150801_1500_1550.csv')
    X04 = pd.read_csv('yhy_150801_1700_1750.csv')
    X06 = pd.read_csv('yhy_150802_1400_1450.csv')
    X07 = pd.read_csv('yhy_150802_1500_1550.csv')
    X08 = pd.read_csv('yhy_150802_1600_1640.csv')
    X10 = pd.read_csv('yhy_150816_0400_0450.csv')
    X11 = pd.read_csv('yhy_150816_1220_1320.csv')

    # Label YHY's Data
    y01 = np.zeros(X01.shape[0])
    y02 = np.zeros(X02.shape[0])
    y03 = np.zeros(X03.shape[0])
    y04 = np.zeros(X04.shape[0])
    y04[600 - 1:768 - 1] = 1  # sleepy
    y04[769 - 1:1534 - 1] = 2  # Deep Sleep
    y06 = np.zeros(X06.shape[0])
    y07 = np.zeros(X07.shape[0])
    y08 = np.zeros(X08.shape[0])
    y10 = np.zeros(X10.shape[0])
    y10[1 - 1:1272 - 1] = 1  # sleepy
    y10[1273 - 1:2360 - 1] = 2  # Deep sleep
    y11 = np.zeros(X11.shape[0])
    y11[1 - 1:1700 - 1] = 1  # sleepy

    # Drop Invalid examples
    X01, y01 = dropInvalid(X01, y01)
    X02, y02 = dropInvalid(X02, y02)
    X03, y03 = dropInvalid(X03, y03)
    X04, y04 = dropInvalid(X04, y04)
    X06, y06 = dropInvalid(X06, y06)
    X07, y07 = dropInvalid(X07, y07)
    X08, y08 = dropInvalid(X08, y08)
    X10, y10 = dropInvalid(X10, y10)
    X11, y11 = dropInvalid(X11, y11)

    # Save the sizes of  YHY's examples
    m_vec = []
    m_vec.append(X01.shape[0])
    m_vec.append(X02.shape[0])
    m_vec.append(X03.shape[0])
    m_vec.append(X04.shape[0])
    m_vec.append(X06.shape[0])
    m_vec.append(X07.shape[0])
    m_vec.append(X08.shape[0])
    m_vec.append(X10.shape[0])
    m_vec.append(X11.shape[0])

    # Integrate YHY's Data
    X = pd.concat([X01, X02, X03, X04, X06, X07, X08, X10, X11], axis=0)
    y = np.concatenate((y01, y02, y03, y04, y06, y07, y08, y10, y11), axis=0)
    return X, y, m_vec


def loadDataP2():
    X01 = pd.read_csv('yhs_150817_1815_1838.csv')
    X02 = pd.read_csv('yhs_150817_1950_2040.csv')
    X03 = pd.read_csv('yhs_150817_2254_2333.csv')
    X04 = pd.read_csv('yhs_150818_1430_1540.csv')
    X05 = pd.read_csv('yhs_150818_2400_2510.csv')
    # ambiguous data: X06 = pd.read_csv('yhs_150819_1922_2008.csv')

    # Label YHS's Data
    y01 = np.zeros(X01.shape[0])
    y02 = np.zeros(X02.shape[0])
    y03 = np.zeros(X03.shape[0])
    y04 = np.ones(X04.shape[0]) * 2  # Deep sleep
    y05 = np.ones(X05.shape[0])

    # Drop Invalid examples
    X01, y01 = dropInvalid(X01, y01)
    X02, y02 = dropInvalid(X02, y02)
    X03, y03 = dropInvalid(X03, y03)
    X04, y04 = dropInvalid(X04, y04)
    X05, y05 = dropInvalid(X05, y05)

    # Save the sizes of  YHS's examples
    m_vec = []
    m_vec.append(X01.shape[0])
    m_vec.append(X02.shape[0])
    m_vec.append(X03.shape[0])
    m_vec.append(X04.shape[0])
    m_vec.append(X05.shape[0])

    # Integrate YHY's Data
    X = pd.concat([X01, X02, X03, X04, X05], axis=0)
    y = np.concatenate((y01, y02, y03, y04, y05), axis=0)
    return X, y, m_vec


def loadDataP3():
    X = pd.read_csv('cu_150823_1450_1530.csv')

    # Label CU's Data
    y = np.zeros(X.shape[0])
    y[600 - 1:1400 - 1] = 1

    # Drop Invalid examples
    X, y = dropInvalid(X, y)

    # Save the sizes of  CU's examples
    m_vec = [X.shape[0]]

    return X, y, m_vec


def loadDataP4():
    X = pd.read_csv('us_150822_1700_1800.csv')

    # Label US's Data
    y = np.zeros(X.shape[0])
    y[:2430 - 1] = 1
    y[2431 - 1:3174 - 1] = 2  # Deep sleep
    y[3175 - 1:3270 - 1] = 1
    # Drop Invalid examples
    X, y = dropInvalid(X, y)

    # Save the sizes of  CU's examples
    m_vec = [X.shape[0]]

    return X, y, m_vec


def addFeatures(X, y, m_vec):
    # 01. Set minimum needed for creating time series features
    min_t = 60
    X_add = pd.DataFrame([])
    y_add = np.array([])
    i = 1 - 1
    for j in range(len(m_vec)):
        # 02. Retrieve individual X's
        X_temp = X.iloc[i:i + m_vec[j]]
        X_temp = X_temp.reset_index(drop=True)
        y_temp = y[i:i + m_vec[j]]

        # 03. Create new Features for every example
        f1 = countblinks(X_temp['blinkStrength'], f_name='N(blinks)_1m', time=60)
        f2 = countAboveValue(X_temp['blinkStrength'], f_name='N(50<blinkStrength)_1m', value=50, time=60)
        f3 = countBelowValue(X_temp['eegRawValue'], f_name='N(eeg<-500)_1m', value=-500, time=60)
        f4 = countAboveValue(X_temp['alphaLow'], f_name='N(alphaLow>1.677e+007)_1m', value=1.677e+007, time=60)
        f5 = countAboveValue(X_temp['betaLow'], f_name='N(betaLow>1.674e+007)_1m', value=1.674e+007, time=60)
        f6 = countBWValues(X_temp['theta'], f_name='N(1e+005<theta<5e+005)_1m', lower=1e+005, upper=5e+005, time=60)
        f7 = countBWValues(X_temp['alphaHigh'], f_name='N(1e+005<alphaHigh<2e+005)_1m', lower=1e+005, upper=2e+005,
                           time=60)

        # 04. Concatenate X and additional Features
        X_temp = pd.concat([X_temp, f1, f2, f3, f4, f5, f6, f7], axis=1)

        # 05. Drop terms w/o time series features
        X_temp = X_temp.iloc[min_t - 1:]
        y_temp = y_temp[min_t - 1:]

        # 06. Roll back into output matrix
        X_add = pd.concat([X_add, X_temp], axis=0)
        y_add = np.concatenate((y_add, y_temp), axis=0)

        # 07. Increment Index for loop
        i = i + m_vec[j]

    X_add = X_add.reset_index(drop=True)

    return X_add, y_add


def countblinks(X, f_name, time):
    X = X.values  # Convert DataFrame to ndarray for speed
    m = X.shape[0]
    # 01. Create new empty array that has same structure but whose elements are all zero.
    temp = np.zeros(m)
    # 02. For elements in X[i] and X[i-1], substitute 1 into corresponding element in temp if they are identical.
    for i in range(1, m):
        if X[i] != X[i - 1]:
            temp[i] = 1
    # 03. Create new empty array 'count' whose elements are all zero.
    count = np.zeros(m)
    # 04. From index time-1 to end of X, sum the elements in temp_df, and substitute into corresponding count_df.
    for j in range(time - 1, m):
        count[j] = np.sum(temp[j - time + 1:j + 1])
    # 05. Return count_df
    count_df = pd.DataFrame(count)
    # Rename column
    count_df.columns = pd.Index([f_name], dtype='object')
    return count_df


def countBWValues(X, f_name, lower, upper, time):
    X = X.values
    m = X.shape[0]
    count = np.zeros(m)
    # 01. sum were condition 1 and condition 2 is met
    for i in range(time - 1, m):
        cond1 = lower < X[i - time + 1: i + 1]
        cond2 = upper > X[i - time + 1: i + 1]
        count[i] = np.sum(np.all([cond1, cond2], axis=0))
    count_df = pd.DataFrame(count)
    count_df.columns = pd.Index([f_name], dtype='object')
    return count_df


def countBelowValue(X, f_name, value, time):
    X = X.values
    m = X.shape[0]
    count = np.zeros(m)
    # 01. sum were condition 1 and condition 2 is met
    for i in range(time - 1, m):
        cond = value > X[i - time + 1: i + 1]
        count[i] = np.sum(cond)
    count_df = pd.DataFrame(count)
    count_df.columns = pd.Index([f_name], dtype='object')
    return count_df


def countAboveValue(X, f_name, value, time):
    X = X.values
    m = X.shape[0]
    count = np.zeros(m)
    # 01. sum were condition 1 and condition 2 is met
    for i in range(time - 1, m):
        cond = value < X[i - time + 1: i + 1]
        count[i] = np.sum(cond)
    count_df = pd.DataFrame(count)
    count_df.columns = pd.Index([f_name], dtype='object')
    return count_df


def LWR(X, f_name, time, gamma):
    # Locally weighted regression for attention and meditation, using analytical methods
    X = X.reset_index(drop=False)
    t = X.pop('index')
    t = t.values
    X = X.values
    m = X.shape[0]
    theta = np.zeros((2, m))
    for i in range(time - 1, m):
        # 01. Construct Local X(i)'s
        t_local = t[i - time + 1:i + 1]
        X_local = X[i - time + 1:i + 1]
        # 02. Calculate Weights for each X(i)'s
        landmark = t[i - time / 2]
        w_vec = np.exp(-1 * (t_local - landmark) ** 2 / (2 * gamma))
        # 03. Calculate theta using normal equation
        m_local = t_local.shape[0]
        temp1 = np.ones((m_local, 1))
        temp2 = t_local.reshape((m_local, 1))
        t_loc_mat = np.concatenate((temp1, temp2), axis=1)
        X_loc_mat = X_local.reshape((m_local, 1))
        LHS = np.dot(np.dot(t_loc_mat.T, np.diag(w_vec)), t_loc_mat)
        RHS = np.dot(np.dot(t_loc_mat.T, np.diag(w_vec)), X_loc_mat)
        theta[(0, i)] = np.linalg.solve(LHS, RHS)[0][0]
        theta[(1, i)] = np.linalg.solve(LHS, RHS)[1][0]
    # Transpose, to fit into the DataFrame
    theta = theta.T
    # 04. Extract slope only
    slope_df = pd.DataFrame(theta[:, 1])
    slope_df.columns = pd.Index([f_name], dtype='object')
    return slope_df


def loadData_and_addFeatures():
    X, y, m_vec = loadData()
    X_add, y_add = addFeatures(X, y, m_vec)

    return X_add, y_add


def loadNewData_and_addFeatures(filename):

    # Load Data

    X_new = pd.read_csv(filename)

    # 01. Set conditions
    cond1 = X_new['poorSignal'] == 0
    cond2 = X_new['blinkStrength'] != 0
    cond = cond1 & cond2

    # 02. Leave valid examples only
    X_valid = X_new[cond]

    m = X_valid.shape[0]
    if m < 60:
        return pd.DataFrame(['Invalid values'])
    else:
        # Extract recent 60 data
        X_valid = X_valid[m-60:m]

    # Reset indices
    X_valid = X_valid.reset_index(drop=True)

    # Add Features
    # 01. Set minimum needed for creating time series features
    min_t = 60
    X_add = pd.DataFrame([])

    # 03. Create new Features for every example
    f1 = countblinks(X_valid['poorSignal'], f_name='N(blinks)_1m', time=60)
    f2 = countAboveValue(X_valid['blinkStrength'], f_name='N(50<blinkStrength)_1m', value=50, time=60)
    f3 = countBelowValue(X_valid['eegRawValue'], f_name='N(eeg<-500)_1m', value=-500, time=60)
    f4 = countAboveValue(X_valid['alphaLow'], f_name='N(alphaLow>1.677e+007)_1m', value=1.677e+007, time=60)
    f5 = countAboveValue(X_valid['betaLow'], f_name='N(betaLow>1.674e+007)_1m', value=1.674e+007, time=60)
    f6 = countBWValues(X_valid['theta'], f_name='N(1e+005<theta<5e+005)_1m', lower=1e+005, upper=5e+005, time=60)
    f7 = countBWValues(X_valid['alphaHigh'], f_name='N(1e+005<alphaHigh<2e+005)_1m', lower=1e+005, upper=2e+005,
                       time=60)

    # 04. Concatenate X and additional Features
    X_add = pd.concat([X_valid, f1, f2, f3, f4, f5, f6, f7], axis=1)

    return X_add


def dropInvalid(X, y):
    # Drop invalid data

    # 01. Set conditions
    cond1 = X['poorSignal'] == 0
    cond2 = X['blinkStrength'] != 0
    cond = cond1 & cond2

    # 02. Leave valid examples only
    X_valid = X[cond]
    y_valid = y[cond.values]

    # 03. Drop unnecessary columns
    X_valid.pop('timestampMs')
    X_valid.pop('poorSignal')
    X_valid.pop('eegRawValueVolts')
    X_valid.pop('location')
    X_valid.pop('tagEvent')

    return X_valid, y_valid


def F1_score(pred_pos, y, class_idx):
    n_pred_pos = float(sum(pred_pos == class_idx))
    n_True_pos = float(sum(np.all([pred_pos == class_idx, y == class_idx], axis=0) == True))
    n_actual_pos = float(sum(y == class_idx))
    Precision = 0
    Recall = 0
    F1 = 0
    if n_pred_pos != 0:
        Precision = n_True_pos / n_pred_pos
    if n_actual_pos != 0:
        Recall = n_True_pos / n_actual_pos
    if Precision + Recall != 0:
        F1 = 2 * Precision * Recall / (Precision + Recall)
    return F1, Precision, Recall


def shuffleData(X, y):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X_shuffled = X[idx]
    y_shuffled = y[idx]
    return X_shuffled, y_shuffled


def plotFeature(X, y, f_name, lim_lower=-1.0, lim_upper=-1.0):
    normal = plt.scatter(X.index, X[f_name], marker='.', color='blue', label='normal')
    sleepy = plt.scatter(X[y == 1].index, X[y == 1][f_name], marker='.', color='red', label='sleepy')
    deep = plt.scatter(X[y == 2].index, X[y == 2][f_name], marker='.', color='green', label='deep sleep')
    plt.title(f_name)
    plt.legend(handles=[normal, sleepy, deep])
    if lim_upper != -1.0:
        plt.ylim(lim_lower, lim_upper)
    plt.show()

    return None


def saveAllPlots(X, y):
    for i in range(X.values.shape[1]):
        save_dir = 'E:\Github\Brainwave Analysis\Images\matplotlib'
        f_name = X.columns[i]
        normal = plt.scatter(X.index, X[f_name], marker='.', color='blue', label='normal')
        sleepy = plt.scatter(X[y == 1].index, X[y == 1][f_name], marker='.', color='red', label='sleepy')
        deep = plt.scatter(X[y == 2].index, X[y == 2][f_name], marker='.', color='green', label='deep sleep')
        plt.title(f_name)
        plt.legend(handles=[normal, sleepy, deep])
        plt.savefig(save_dir + '\\' + 'figure' + str(i), dpi=400)
        plt.clf()

    return None


def plotNotableObservation(X, y):
    plotFeature(X, y, 'blinkStrength')
    plotFeature(X, y, 'eegRawValue', -1000, 100)
    plotFeature(X, y, 'theta', 0, 5e+005)
    plotFeature(X, y, 'alphaLow', 1.675e+007, 1.679e+007)
    plotFeature(X, y, 'betaLow', 1.675e+007, 1.679e+007)
    plotFeature(X, y, 'alphaHigh', 0, 5e+005)

    return None
