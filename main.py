from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import numpy as np
from pathlib import Path
from torch import nn, Tensor, squeeze
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision import transforms, datasets
import torchvision
from os import listdir
from matplotlib import image

print(os.getcwd())

# # im = Image.open(r'G:\17810_23812_bundle_archive\chest_xray\chest_xray\test\NORMAL\IM-0001-0001.jpeg')
# # print(im.format, im.size, im.mode)
# # im.show();
############################################# DATA EVALUATION ##############################################################
# train_normal = len(os.listdir(r'G:\17810_23812_bundle_archive\chest_xray\chest_xray\train\NORMAL'))
# train_pneumonia = len(os.listdir(r'G:\17810_23812_bundle_archive\chest_xray\chest_xray\train\PNEUMONIA'))
# val_normal = len(os.listdir(r'G:\17810_23812_bundle_archive\chest_xray\chest_xray\val\NORMAL'))
# val_pneumonia = len(os.listdir(r'G:\17810_23812_bundle_archive\chest_xray\chest_xray\val\PNEUMONIA'))
# test_normal = len(os.listdir(r'G:\17810_23812_bundle_archive\chest_xray\chest_xray\test\NORMAL'))
# test_pneumonia = len(os.listdir(r'G:\17810_23812_bundle_archive\chest_xray\chest_xray\test\PNEUMONIA'))
#
# print('BEFORE REALLOCATION:\n', 'train_normal:', train_normal, '\n', 'train_pneumonia:', train_pneumonia, '\n', 'val_normal:', val_normal, '\n', 'val_pneumonia:', val_pneumonia, '\n', 'test_normal:', test_normal, '\n', 'test_pneumonia:', test_pneumonia, '\n')
#
# normal_sum = train_normal + test_normal + val_normal
# pneumonia_sum = train_pneumonia + test_pneumonia + val_pneumonia
# print('TEST/TRAIN/VAL SUM:\n', 'normal:', normal_sum, '\n', 'pneumonia:', pneumonia_sum, '\n')
# print('EXPECTED SPREAD AFTER REALLOCATION:', 'train_normal:', round(normal_sum * 0.6), '\n', 'train_pneumonia:', round(pneumonia_sum * 0.6), '\n', 'val_normal:', round(normal_sum * 0.2), '\n', 'val_pneumonia:', round(pneumonia_sum * 0.2), '\n', 'test_normal:', round(normal_sum  * 0.2), '\n', 'test_pneumona:', round(pneumonia_sum * 0.2), '\n')
#
# joined_normal_dir = r'G:\17810_23812_bundle_archive\chest_xray\chest_xray\joined\NORMAL'
# joined_pneumonia_dir = r'G:\17810_23812_bundle_archive\chest_xray\chest_xray\joined\PNEUMONIA'

############################################# DATA REALLOCATION ##############################################################

# import os
# import shutil
# src_files = os.listdir(r"G:\17810_23812_bundle_archive\chest_xray\chest_xray\test\PNEUMONIA")
# for file_name  in src_files:
#     full_file_name = os.path.join(r"G:\17810_23812_bundle_archive\chest_xray\chest_xray\test\PNEUMONIA", file_name)
#     if os.path.isfile(full_file_name):
#         shutil.copy(full_file_name, joined_pneumonia_dir)

# normal = len(os.listdir(r'G:\17810_23812_bundle_archive\chest_xray\chest_xray\joined\NORMAL'))
# pneumonia = len(os.listdir(r'G:\17810_23812_bundle_archive\chest_xray\chest_xray\joined\PNEUMONIA'))
# print("\nexpected vs current:\n", "normal: ", normal_sum, " vs ", normal, "\npneumonia: ", pneumonia_sum, " vs ", pneumonia)
#
# joined_normal_dir = r'G:\17810_23812_bundle_archive\chest_xray\chest_xray\joined\NORMAL'
# joined_pneumonia_dir = r'G:\17810_23812_bundle_archive\chest_xray\chest_xray\joined\PNEUMONIA'
#
# src_files_normal = os.listdir(joined_normal_dir)
# src_files_pneumonia = os.listdir(joined_pneumonia_dir)
#
# train_normal_elem = []
# train_normal_elem = src_files_normal[0:950]
# val_normal_elem = src_files_normal[950:1267]
# test_normal_elem = src_files_normal[1267:]
#
# train_pneumonia_elem = src_files_pneumonia[0:2564]
# val_pneumonia_elem = src_files_pneumonia[2564:3419]
# test_pneumonia_elem = src_files_pneumonia[3419:]
#
# print("\ntrain_normal_elem:", train_normal_elem)
# print("\nval_normal_elem:", val_normal_elem)
# print("\ntest_normal_elem:", test_normal_elem)
# print("\ntrain_pneumonia_elem:", train_pneumonia_elem)
# print("\nval_pneumonia_elem:", val_pneumonia_elem)
# print("\ntest_pneumonia_elem:", test_pneumonia_elem)

# import os
# import shutil
# for file_name in test_pneumonia_elem:
#     full_file_name = os.path.join(r'G:\17810_23812_bundle_archive\chest_xray\chest_xray\joined\PNEUMONIA', file_name)
#     if os.path.isfile(full_file_name):
#         shutil.copy(full_file_name, r'G:\17810_23812_bundle_archive\chest_xray\chest_xray\rearranged\test\PNEUMONIA')

# train_normal_rearranged = len(os.listdir(r'G:\17810_23812_bundle_archive\chest_xray\chest_xray\rearranged\train\NORMAL'))
# val_normal_rearranged = len(os.listdir(r'G:\17810_23812_bundle_archive\chest_xray\chest_xray\rearranged\val\NORMAL'))
# test_normal_rearranged = len(os.listdir(r'G:\17810_23812_bundle_archive\chest_xray\chest_xray\rearranged\test\NORMAL'))
#
# train_pneumonia_rearranged = len(os.listdir(r'G:\17810_23812_bundle_archive\chest_xray\chest_xray\rearranged\train\PNEUMONIA'))
# val_pneumonia_rearranged = len(os.listdir(r'G:\17810_23812_bundle_archive\chest_xray\chest_xray\rearranged\val\PNEUMONIA'))
# test_pneumonia_rearranged = len(os.listdir(r'G:\17810_23812_bundle_archive\chest_xray\chest_xray\rearranged\test\PNEUMONIA'))
#
# print("\ntrain_normal:", train_normal)
# print("\ttrain_normal_rearranged:", train_normal_rearranged)
#
# print("\nval_normal:", val_normal)
# print("\tval_normal_rearranged:", val_normal_rearranged)
#
# print("\ntest_normal:", test_normal)
# print("\ttest_normal_rearranged:", test_normal_rearranged)
#
# print("\ntrain_pneumonia:", train_pneumonia)
# print("\ttrain_peumonia_rearranged:", train_pneumonia_rearranged)
#
# print("\nval_pneumonia:", val_pneumonia)
# print("\tval_pneumonia_rearranged:", val_pneumonia_rearranged)
#
# print("\ntest_pneumonia:", test_pneumonia)
# print("\ttest_pneumonia_rearranged:", test_pneumonia_rearranged)

# df = pd.DataFrame({
#     'normal': [train_normal_rearranged, test_normal_rearranged, val_normal_rearranged],
#     'pneumonia': [train_pneumonia_rearranged, test_pneumonia_rearranged, val_pneumonia_rearranged]
# }, index=['train', 'test', 'val'])
#
# spread = df.plot.bar(legend=True)
# plt.show()

############################################# DATA LOAD ##############################################################

train_normal_rearranged = Path(r"G:\17810_23812_bundle_archive\chest_xray\chest_xray\rearranged\train\NORMAL")
train_pneumonia_rearranged = Path(r"G:\17810_23812_bundle_archive\chest_xray\chest_xray\rearranged\train\PNEUMONIA")
val_normal_rearranged = Path(r"G:\17810_23812_bundle_archive\chest_xray\chest_xray\rearranged\val\NORMAL")
val_pneumonia_rearranged = Path(r"G:\17810_23812_bundle_archive\chest_xray\chest_xray\rearranged\val\PNEUMONIA")
test_normal_rearranged = Path(r"G:\17810_23812_bundle_archive\chest_xray\chest_xray\rearranged\test\NORMAL")
test_pneumonia_rearranged = Path(r"G:\17810_23812_bundle_archive\chest_xray\chest_xray\rearranged\test\PNEUMONIA")
BASE_PATH = Path(r"G:\17810_23812_bundle_archive\chest_xray\chest_xray")

train_path = Path(r"G:\17810_23812_bundle_archive\chest_xray\chest_xray\rearranged\train")
val_path = Path(r"G:\17810_23812_bundle_archive\chest_xray\chest_xray\rearranged\val")
test_path = Path(r"G:\17810_23812_bundle_archive\chest_xray\chest_xray\rearranged\test")
# train_normal_rearranged  = os.path.join(train_normal_rearranged, '')
# print(train_normal_rearranged)
# train_normal_rearranged = train_normal_rearranged + os.path.sep

# train_normal_labels = torch.ones(len(os.listdir(train_normal_rearranged)))
# train_pneumonia_labels = torch.zeros(len(os.listdir(train_pneumonia_rearranged)))

# img = Image.open(BASE_PATH + r"\rearranged\train\NORMAL\IM-0001-0001.jpeg")
# print(img.getbands())
# mask = np.array(img)
# shape = mask.shape
# # print(mask)
# # print(shape)
# trans = transforms.ToTensor()
# test = trans(img)
# print(test)

# train_normal_images = list()
# for filename in listdir(train_normal_rearranged):
# 	# load image
# 	img_data = image.imread(train_normal_rearranged + os.path.sep + filename)
# # 	# store loaded image
# 	train_normal_images.append(img_data)
# 	print('> loaded %s %s' % (filename, img_data.shape))
#
transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor()])

dataset_train = datasets.ImageFolder(train_path, transform=transform)
dataset_val = datasets.ImageFolder(val_path, transform=transform)
dataset_test = datasets.ImageFolder(test_path, transform=transform)


train_loader = DataLoader(dataset_train, shuffle=True, batch_size=4)


dataiter = iter(train_loader)
images, labels = dataiter.next()
#
#
# images_test = images[0].permute(1,2,0)
# images_test = squeeze(images_test, dim=2)
# print(images_test.shape)
# print(images_test, labels[0])
# plt.imshow(images_test, cmap='gray')
# plt.show()


images_grid = torchvision.utils.make_grid(images)
images_permute = images_grid.permute(1,2,0)
print(images.shape)
print(images_permute.shape)
print(labels)
plt.imshow(torchvision.utils.make_grid(images_permute), cmap='gray')
plt.show()



# class ConvNet(nn.Module):
#     def __init__(self):
#         self.conv1 = nn.Conv2d(1, 6)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(1, 254, 254)
#
#     def forward(self, x):
#         pass

# # Make a grid from batch
# out = torchvision.utils.make_grid(inputs)
#
# imshow(out, title=[class_names[x] for x in classes])


# inputs, classes = next(iter(loader['NORMAL']))
# out = torchvision.utils.make_grid(inputs)
# imshow(out, title=[class_names[x] for x in classes])

# dataset_pneumonia_train = datasets.ImageFolder(train_pneumonia_rearranged, transform=transform)
# dataset_normal_val = datasets.ImageFolder(val_normal_rearranged, transform=transform)
# dataset_pneumonia_val = datasets.ImageFolder(val_pneumonia_rearranged, transform=transform)
# dataset_normal_test = datasets.ImageFolder(test_normal_rearranged, transform=transform)
# dataset_pneumonia_test = datasets.ImageFolder(test_pneumonia_rearranged, transform=transform)


# for images, labels in dataset_train.take(1):  # only take first element of dataset
#     numpy_images = images.numpy()
#     numpy_labels = labels.numpy()
#
# print(numpy_images, numpy_labels)
# dataloader = torch.utils.data.DataLoader(dataset_normal_train, batch_size=32, shuffle=True)
# images, labels = next(iter(dataloader))
# plt.imshow()
# df = pd.DataFrame(im3_t[4:15,4:22])
# df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')
# img.show()


############################################# NEURAL NETWORK ##############################################################



