
import tensorflow as tf 
classifierLoad = tf.keras.models.load_model('model.h5')

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('C:/Users/user/Desktop/Code with GUI/Diptera.jpg', target_size = (200,200))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifierLoad.predict(test_image)



if result[0][0] == 1:
    print("The result of classification is:-"+"    Auchenorrhyncha")
elif result[0][1] == 1:
    print("The result of classification is:-"+"    Heteroptera")
elif result[0][2] == 1:
    print("The result of classification is:-"+"    Hymenoptera")
elif result[0][3] == 1:
    print("The result of classification is:-"+"    Lepidoptera")
elif result[0][4] == 1:
    print("The result of classification is:-"+"    Megalptera")
elif result[0][5] == 1:
    print("The result of classification is:-"+"    Neuroptera")
elif result[0][6] == 1:
    print("The result of classification is:-"+"    Odonata")
elif result[0][7] == 1:
    print("The result of classification is:-"+"    Orthoptera")
elif result[0][8] == 1:
    print("The result of classification is:-"+"   Diptera")
