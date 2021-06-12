from IPython import get_ipython
get_ipython().magic('reset -sf')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




data = pd.read_csv("density_3.csv")

data['correct'] = data['answer'] == data['train_key1.keys']

refRadius = 2
testRadius = 2
correctN = []



###number of gaussian blobs for test patch in 3x3 conditions

r2_t2 = [64, 81, 102, 128, 162, 203, 256]
r2_t28 = [125, 158, 198, 251, 316, 399, 502]
r2_t4 = [256, 324, 408, 512, 648, 812, 1024]

r28_t2 = [33, 41, 51, 66, 82, 103, 130]
r28_t28 = [64, 81, 102, 128, 162, 203, 256]
r28_t4 = [132, 164, 204, 264, 328, 412, 520]

r4_t2 = [16, 20, 25, 32, 40, 51, 64]
r4_t28 = [32, 40, 50, 63, 79, 100, 126]
r4_t4 = [64, 81, 102, 128, 162, 203, 256]



for num in r4_t4:
    data['rR_tR'] = np.where((data['refRadius'] == refRadius) & (data['testRadius'] == testRadius) & (data['testDotNum'] == num), 1, 0)
    data['rR_tR_correct'] = np.where((data['rR_tR'] == 1) & (data['correct'] == True) & (data['denser'] == 'test'), 1, 0)
    try:
        correct = data['rR_tR_correct'].value_counts()[1]
    except KeyError:    
        correct = 0
    correctN.append(correct)
    
    
print correctN
#correctN = [n/5.0 for n in correctN]
#print correctN



"""

plt.xlim(-0.2, 6.2)
plt.ylim(-0.1, 1.1)



dr2_t2 = [0, 0, 0, 0.4, 1, 1, 1]
dmodel_r2_t2 = [0, 0, 0, 0.5, 1, 1, 1]


dr2_t28 = [0, 0, 0, 1, 1, 1, 1]
dmodel_r2_t28 = [0, 0, 0, 1, 1, 1, 1]


dr2_t4 = [0, 0, 0, 1, 1, 1, 1]
dmodel_r2_t4 = [0, 0, 1, 1, 1, 1, 1]





plt.plot(dr2_t2, 'o', color='black',  markersize=12)
plt.plot(dmodel_r2_t2, 'o', color='red')
plt.plot(dr2_t2, '--', color='black')
plt.plot(dmodel_r2_t2, '--', color='red')


#plt.plot(dr2_t28, 'o', color='black',  markersize=12)
#plt.plot(dmodel_r2_t28, 'o', color='red')
#plt.plot(dr2_t28, '--', color='black')
#plt.plot(dmodel_r2_t28, '--', color='red')


#plt.plot(dr2_t4, 'o', color='black',  markersize=12)
#plt.plot(dmodel_r2_t4, 'o', color='red')
#plt.plot(dr2_t4, '--', color='black')
#plt.plot(dmodel_r2_t4, '--', color='red')






ax = plt.gca()
ax.set_xticklabels(('0', '0.5', '0.63', '0.79', '1.0', '1.26', '1.59', '2.0'))


ax.set_title("Ref:Test radius($2^o:2.8^o$)")
plt.xlabel('Test/Ref density')
plt.ylabel('P("Test>Ref")')

plt.legend(('observer', 'model'), loc='upper left')

"""

