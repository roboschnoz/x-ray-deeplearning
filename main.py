from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import numpy as np
from pathlib import Path
from torch.nn import functional
from torch import nn, Tensor, squeeze, flatten, device
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision import transforms, datasets
import torchvision
import torch
from os import listdir
from matplotlib import image

print(os.getcwd())

device = torch.device('cpu')
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
classes = ('normal', 'pneumonia')
num_epochs = 4
batch_size = 4


transform = transforms.Compose([transforms.Resize(268),
                                transforms.CenterCrop(268),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor()])

dataset_train = datasets.ImageFolder(train_path, transform=transform)
dataset_val = datasets.ImageFolder(val_path, transform=transform)
dataset_test = datasets.ImageFolder(test_path, transform=transform)


train_loader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(dataset_val, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(dataset_test, shuffle=True, batch_size=batch_size)


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

###################################### SHOW BATCH
# images_grid = torchvision.utils.make_grid(images)
# images_permute = images_grid.permute(1,2,0)
# print(images.shape)
# print(images_permute.shape)
# print(labels)
# plt.imshow(torchvision.utils.make_grid(images_permute), cmap='gray')
# plt.show()

##### BLOCK START
# def run():
#     torch.multiprocessing.freeze_support()
#     print('loop')
#
# if __name__ == '__main__':
#     run()
#
# train_set = datasets.ImageFolder(train_path, transform=transforms)
# train_loader_full = DataLoader(dataset_train, batch_size=len(train_set), num_workers=1)
# data = next(iter(train_loader_full))
#
# print(data[0].mean(), data[0].std())


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 264
        self.pool = nn.MaxPool2d(2, 2) # 132
        self.conv2 = nn.Conv2d(6, 16, 5) # 128
        self.pool = nn.MaxPool2d(2, 2) # 64
        self.conv3 = nn.Conv2d(16, 32, 5) # 60
        self.pool = nn.MaxPool2d(2, 2) # 30
        self.conv4 = nn.Conv2d(32, 64, 5) # 26
        self.pool = nn.MaxPool2d(2, 2) # 13
        self.conv5 = nn.Conv2d(64, 128, 5) # 9
        self.fc1 = nn.Linear(128*9*9, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = self.pool(functional.relu(self.conv3(x)))
        x = self.pool(functional.relu(self.conv4(x)))
        x = functional.relu(self.conv5(x))
        x = x.view(-1, 128*9*9)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ConvNet().to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)


# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

#forward pass
outputs = model(images)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.view(4).float(), labels.float())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 2000 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = 'G:/x-ray_deeplearning/model'
# torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

##### BLOCK END



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



