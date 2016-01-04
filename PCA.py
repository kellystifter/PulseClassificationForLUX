import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

#perform principle component analysis using only the listed features
Xpca = designMatrix.loc[:,['pulse_area_phe','s2filter_max_s2_area','pulse_std_phe_per_sample',
                           'pulse_length_samples','prompt_fraction_tlx','amis1_fraction',
                           's2filter_max_s1_area','top_bottom_asymmetry','pulse_height_phe_per_sample',
                           'skinny_pulse_area_phe','rms_width_samples']]
pca = PCA(n_components = 2)
X_pca = pca.fit_transform(Xpca)
X_pca = pd.DataFrame(X_pca)

#plot the result
plt.scatter(x=X_pca.iloc[:,0],y=X_pca.iloc[:,1], c = y, s=15, lw=0)

plt.xlabel('PCA feature 1')
plt.ylabel('PCA feature 2')
plt.title('Transformed samples')
plt.axis([-2340, -2200, 140, 160])

plt.show()
