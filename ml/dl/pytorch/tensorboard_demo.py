#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# dataset
trainset = torchvision.datasets.FashionMNIST("./data", download=True, train=True, transform=transform)
testset  = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=transform)

# data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testloader  = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


# helper function to show an image
# (used in the 'plot_classes_preds' function below
def matplotlib_imshow(img, one_channel=False):
  if one_channel:
    img = img.mean(dim=0)
  img = img / 2 + 0.5
  npimg = img.numpy()
  if one_channel:
    plt.imshow(npimg, cmap="Greys")
  else:
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5)  # input image is 1x28x28
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 4 * 4, 120)
    self.fc2 = nn.Linear(120, 80)
    self.fc3 = nn.Linear(80, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 4 * 4)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)

    return x


net = Net()

loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter("runs/fashion_mnist_experiment_1")

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# merge a batch of images into a single image(Tensor)
img_grid = torchvision.utils.make_grid(images)

# show the image batch
matplotlib_imshow(img_grid, one_channel=True)

# write an image to tensorboard
writer.add_image("four_fashion_mnist_images", img_grid)

writer.add_graph(net, images)


def select_n_random(data, labels, n=100):
  '''
  Selects n random datapoints and their corresponding labels from a dataset
  :param data:
  :param label:
  :param n:
  :return:
  '''
  assert len(data) == len(labels)
  perm = torch.randperm(len(data))
  return data[perm][:n], labels[perm][:n]


# select random images and their target indices
images, labels = select_n_random(trainset.data, trainset.targets)

# get class label for each image
class_labels = [classes[t] for t in labels]

# log embeddings
features = images.view(-1, 28 * 28)
writer.add_embedding(features, metadata=class_labels, label_img=images.unsqueeze(1))


# helper functions
def images_to_probs(net, images):
  '''
  Generates predictions and corresponding probabilities from a trained
  network and a list of images
  '''
  output = net(images)
  _, pred_class_index_tensor = torch.max(output, 1)
  pred_class_index = np.squeeze(pred_class_index_tensor.numpy())
  return pred_class_index, [F.softmax(ev, dim=0)[ci].item() for ci, ev in zip(pred_class_index, output)]


def plot_classes_preds(net, images, labels):
  '''
  Generates matplotlib Figure using a trained network, along with images
  and labels from a batch, that shows the network's top prediction along
  with its probability, alongside the actual label, coloring this
  information based on whether the prediction was correct or not.
  Uses the "images_to_probs" function.
  '''
  pred_classes, probs = images_to_probs(net, images)
  # plot the images in the batch, along with predicted and true labels
  fig = plt.figure(figsize=(12, 48))
  for idx in np.arange(4):
    ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
    matplotlib_imshow(images[idx], one_channel=True)
    title = "{0}, {1:.1f}%\n(label: {2})".format(classes[pred_classes[idx]],
                                                 probs[idx] * 100.0, classes[labels[idx]])
    color = "green" if pred_classes[idx] == labels[idx].item() else "red"
    ax.set_title(title, color=color)

  return fig


running_loss = 0.0
for epoch in range(1):
  for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = loss_func(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if i % 1000 == 999:
      # log the running loss
      writer.add_scalar("training loss", running_loss / 1000, epoch * len(trainloader) + i)

      # log a Matplotlib Figure showing the model's predictions on a random mini-batch
      writer.add_figure("predictions vs. actuals",
                        plot_classes_preds(net, inputs, labels),
                        global_step=epoch * len(trainloader) + i)
      running_loss = 0.0

print("Finished training")


# 1. gets the probability predictions in a test_size x num_classes Tensor
# 2. gets the preds in a test_size Tensor
# takes ~10 seconds to run
class_probs = []
class_preds = []

with torch.no_grad():
  for data in testloader:
    images, labels = data
    output = net(images)
    class_probs_batch = F.softmax(output, dim=1)  # [F.softmax(ev, dim=0) for ev in output]  # this is a list
    _, class_preds_batch = torch.max(output, 1)

    class_probs.append(class_probs_batch)
    class_preds.append(class_preds_batch)

test_probs = torch.cat(class_probs)
test_preds = torch.cat(class_preds)

# helper function
def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
  '''
  Takes in a "class_index" from 0 to 9 and plots the corresponding
  precision-recall curve
  '''
  tensorboard_preds = test_preds == class_index
  tensorboard_probs = test_probs[:, class_index]

  writer.add_pr_curve(classes[class_index], tensorboard_preds, tensorboard_probs,global_step=global_step)


# plot all the pr curves
for i in range(len(classes)):
  add_pr_curve_tensorboard(i, test_probs, test_preds)

writer.close()
