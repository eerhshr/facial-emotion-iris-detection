import os
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot
from PIL import Image

data_org = pd.read_csv("fer2013/fer2013.csv")
data_org.head()

data_org.emotion.unique()

data_org.groupby(['Usage']).count()

for i, data in data_org.groupby('Usage'):
    data.to_csv("{}.csv".format(i))

test_private = pd.read_csv("PrivateTest.csv")
test_private

test_public = pd.read_csv("PublicTest.csv")
test_public

train = pd.read_csv("Training.csv")
train

"""#Creating Images for available data emotions

## Converting pixel information to image
"""

def pixel_to_img(path, data_df):
  tosave_path = path
  data = data_df
  for i in range(len(data)):
    x = np.array(data.pixels[i].split(" ")).reshape(48,48).astype('float')
    #pyplot.figure()
    #pyplot.imshow(x)
    fname = path.strip('/')
    filename = fname+str(i)+".jpg"
    pyplot.imsave(path+filename, x)

"""#### Conversion for Train Data"""

# Commented out IPython magic to ensure Python compatibility.
# %cd train/

!ls

train_angry = pd.read_csv('0.csv')
pixel_to_img('angry/',train_angry)

train_disgust = pd.read_csv('1.csv')
pixel_to_img('disgust/',train_disgust)

train_fear = pd.read_csv('2.csv')
pixel_to_img('fear/',train_fear)

train_happy = pd.read_csv('3.csv')
pixel_to_img('happy/',train_happy)

train_sad = pd.read_csv('4.csv')
pixel_to_img('sad/',train_sad)

train_suprise = pd.read_csv('5.csv')
pixel_to_img('suprise/',train_suprise)

train_neutral = pd.read_csv('6.csv')
pixel_to_img('neutral/',train_neutral)

"""#### Conversion for Test Data

"""

test_angry = pd.read_csv('0.csv')
pixel_to_img('angry/',test_angry)

test_disgust = pd.read_csv('1.csv')
pixel_to_img('disgust/',test_disgust)

test_fear = pd.read_csv('2.csv')
pixel_to_img('fear/',test_fear)

test_happy = pd.read_csv('3.csv')
pixel_to_img('happy/',test_happy)

test_sad = pd.read_csv('4.csv')
pixel_to_img('sad/',test_sad)

test_suprise = pd.read_csv('5.csv')
pixel_to_img('suprise/',test_suprise)

test_neutral = pd.read_csv('6.csv')
pixel_to_img('neutral/',test_neutral)

"""#### Conversion for Validation data"""

v_angry = pd.read_csv('0.csv')
pixel_to_img('angry/',v_angry)

v_disgust = pd.read_csv('1.csv')
pixel_to_img('disgust/',v_disgust)

v_fear = pd.read_csv('2.csv')
pixel_to_img('fear/',v_fear)

v_happy = pd.read_csv('3.csv')
pixel_to_img('happy/',v_happy)

v_sad = pd.read_csv('4.csv')
pixel_to_img('sad/',v_sad)

v_suprise = pd.read_csv('5.csv')
pixel_to_img('suprise/',v_suprise)

v_neutral = pd.read_csv('6.csv')
pixel_to_img('neutral/',v_neutral)

"""## Converting Images to Gray Scale Images - Batch Conversion"""

direc = os.listdir(input_folder)
for i in direc:
    img = Image.open(input_folder+i)
    grayimg = img.convert('L')
    grayimg.save(output_folder+i)

"""#### Path for Data - GrayScale"""

input_folder = './angry/'
output_folder = './angry_BW/'

input_folder = './disgust/'
output_folder = './disgust_BW/'

input_folder = './fear/'
output_folder = './fear_BW/'

input_folder = './happy/'
output_folder = './happy_BW/'

input_folder = './sad/'
output_folder = './sad_BW/'

input_folder = './suprise/'
output_folder = './suprise_BW/'

input_folder = './neutral/'
output_folder = './neutral_BW/'

