import os,cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import itertools

from keras import backend as K
K.set_image_dim_ordering('th')
def get_featuremaps(model, layer_idx, X_batch):
	get_activations = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer_idx].output,])
	activations = get_activations([X_batch,0])
	return activations

def filtre_goster(model,layer_num,test_image):
    layer_num=0
    filter_num=0
    
    activations = get_featuremaps(model, int(layer_num),test_image)
    
    print (np.shape(activations))
    feature_maps = activations[0][0]      
    print (np.shape(feature_maps))
    
    if K.image_dim_ordering()=='th':
    	feature_maps=np.rollaxis((np.rollaxis(feature_maps,2,0)),2,0)
    print (feature_maps.shape)
    
    fig=plt.figure(figsize=(16,16))
    plt.imshow(feature_maps[:,:,filter_num],cmap='gray')
    plt.savefig("Özellik Haritası-katman-{}".format(layer_num) + "-filter no-{}".format(filter_num)+'.jpg')
    
    num_of_featuremaps=feature_maps.shape[2]
    fig=plt.figure(figsize=(16,16))	
    plt.title("Özellik Haritası-katman-{}".format(layer_num))
    subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
    for i in range(int(num_of_featuremaps)):
    	ax = fig.add_subplot(subplot_num, subplot_num, i+1)
    	#ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
    	ax.imshow(feature_maps[:,:,i],cmap='gray')
    	plt.xticks([])
    	plt.yticks([])
    	plt.tight_layout()
    plt.show()
    fig.savefig("Özellik Haritası-katman-{}".format(layer_num) + '.jpg')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Tahmin matrisi',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Doğru tahmin')
    plt.xlabel('Ağın tahmini')
    plt.show()
    
def tahmin_matrisi(y_test,y_pred):
    target_names = ['class 0(kedi)', 'class 1(kopek)', 'class 2(at)','class 3(insan)']
    cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))
    plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Tahmin matrisi')
    