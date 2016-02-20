#Tests use of histogram in adaptation

import numpy as np
FitnessValues_res=np.array([0,4,29,30,28,16,25,24]);

tmp1 = np.histogram(FitnessValues_res,  (0, 25, 26, 32))[0]
tmp = tmp1[2]
tmpmax = tmp1[1]

print tmp
print tmpmax
