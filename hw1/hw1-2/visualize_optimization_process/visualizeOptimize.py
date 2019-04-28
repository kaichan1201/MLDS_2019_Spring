import numpy as np
from keras.models import load_model
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt

LAYERS = 3
CHOOSE_LAYER = 0
EVENT_NUM = 8
EPOCH_NUM = 5
MODEL_NUM = EVENT_NUM*EPOCH_NUM

with open('loss_history.txt',mode='r') as file:
    loss_history = file.readlines()
loss_history = np.array([round(float(x.strip()),2)*1000 for x in loss_history])

ready_weights = []
choose_layer_weights = []

for e in range(EVENT_NUM):
    for ep in range(EPOCH_NUM):
        print("EVENT_NUM: %02d, EPOCH_NUM: %02d" % (e,(ep+1)*3))
        model = load_model("weights/%d_weights-record-%02d.hdf5" % (e,(ep+1)*3))
        weights = model.get_weights()
        temp = []
        for w in weights:
            temp += list(w.flatten())
        ready_weights.append(temp)
        choose_layer_weights.append(list(np.append(weights[CHOOSE_LAYER*2],weights[CHOOSE_LAYER*2+1])))

pca = PCA(n_components=2)
pca_weights = pca.fit_transform(ready_weights)
choose_pca_weights = pca.fit_transform(choose_layer_weights)

fig, ax = plt.subplots()
cmap = plt.cm.get_cmap('Spectral')
colors = []
for i in range(EVENT_NUM):
    colors += [i]*EPOCH_NUM

plt.scatter(pca_weights[:,0],pca_weights[:,1],c=colors)
for i,l in enumerate(loss_history):
    plt.text(pca_weights[i,0],pca_weights[i,1],l)
plt.savefig('visualize.png')

plt.figure()
plt.scatter(choose_pca_weights[:,0],choose_pca_weights[:,1],c=colors)
for i,l in enumerate(loss_history):
    plt.text(choose_pca_weights[i,0],choose_pca_weights[i,1],l)
plt.savefig('layer_visualize.png')
del model