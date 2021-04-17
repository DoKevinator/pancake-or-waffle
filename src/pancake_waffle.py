import matplotlib.pyplot as plt 
# %matplotlib inline

import torch                                            #PyTorch deep learning framework
from torchvision import datasets, models, transforms    # dataset management from Pytorch
import torch.nn as nn                                   # neural network from Pytorch
from torch.nn import functional as F                    # special functions 
import torch.optim as optim                             # optimization library

import os                                               # used for interacting with the os
from PIL import Image                                   # tool used to view images

from tqdm.notebook import trange, tqdm                  # progress bars

from glob import glob
from math import floor

import functools
print = functools.partial(print, flush=True)

data_path = "../data"
training_path = "../data/train"
validation_path = "../data/validation"

# the size that all the test/validation images will be resized to
image_height = 300
image_width = 300

class MyModel(nn.Module):

    def __init__( self, input_dimenions=(3, image_height, image_width)):
        super().__init__()

        # total number of RGB pixels per picture
        total_rgb_pixels = input_dimenions[0] * input_dimenions[1] * input_dimenions[2]

        # Note: it's hard to figure out what's really going on in nn.Linear under
        # the hood. I'm assuming it's one of the more basic neural network
        # algorithms like Perceptron. There will be multi layers because classifying 
        # images is most definitely not a linearly separated dataset. 
        # Multilayered perceptron is important here so that we can create a 
        # "curvy line" to distinguish between the different types of images. 

        # starting input layer
        self.layer0 = nn.Linear(total_rgb_pixels, 128)

        # hidden intermediate layers 
        self.layer1 = nn.Linear(128, 64)
        self.layer2 = nn.Linear(64, 64)

        # final output layer (output is of size 2 for pancakes/waffles)
        self.layer3 = nn.Linear(64, 2)

    # forward function is where we pass in the input data. (note: the input data 
    # has already been converted into a tensor at this point)
    def forward( self, x ):

        batch_size = x.shape[0]

        # convert the RGB vector into one long vector
        x = x.view(batch_size, -1)

        # pass the input through the layers
        # relu() is a "squishification" function that's more efficient than sigmoid
        # I learned of sigmoid and relu through 3blue1brown 
        x = F.relu(self.layer0(x)) 
        x = F.relu(self.layer1(x)) 
        x = F.relu(self.layer2(x)) 
        x = F.relu(self.layer3(x)) 

        # converts the output into a probability number
        x = F.softmax(x, dim=1)

        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
model

normalize = transforms.Normalize( mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
train_transforms = transforms.Compose([
    transforms.Resize((image_height, image_width)), #resize the images
    transforms.ToTensor(), # convert image to tensor
    normalize
])
validation_transforms = transforms.Compose([
    transforms.Resize((image_height, image_width)), #resize the images
    transforms.ToTensor(), # convert image to tensor
    normalize
])

# print( "Train transforms:", train_transforms)

image_datasets = {
    'train':
        datasets.ImageFolder('../data/train', train_transforms),
    'validation':
        datasets.ImageFolder('../data/validation', validation_transforms)}

# print("==Train Dataset==\n", image_datasets["train"])
# print()
# print("==Validation Dataset==\n", image_datasets["train"])

dataloaders = {
    'train':
        torch.utils.data.DataLoader(
            image_datasets['train'],
            batch_size=8,
            shuffle=True,
            num_workers=4),
    'validation':
        torch.utils.data.DataLoader(
            image_datasets['validation'],
            batch_size=8,
            shuffle=True,
            num_workers=4)
}
# print("Train loader:", dataloaders["train"])
# print("Validation loader:", dataloaders["validation"])

# print(next(iter(dataloaders["train"])))

def train_model( model, dataloaders, loss_function, optimizer, num_epochs):

    for epoch in trange( num_epochs, desc="Total progress", unit="epoch"):
        print( 'Epoch {}/{}'.format(epoch+1, num_epochs))
        print('--------------')

        # split it up into a training section and a validation section
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # keeps track of number of hits and misses
            running_loss = 0.0
            running_corrects = 0

            # for inputs, labels in tqdm_notebook(dataloaders[phase], desc=phase, unit="batch", leave=False):
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs) #forward pass
                loss = loss_function(outputs, labels) #calculate the error from the model

                if phase == 'train':
                    optimizer.zero_grad()   #reset gradients from previous run
                    loss.backward()         #backpropagation to generate new gradients 
                    optimizer.step()        #update weights according to gradient
                
                running_loss += loss.item() * inputs.size(0)    # track misses
                _, preds = torch.max(outputs, 1)    # track hits
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            print(f'{phase} error: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        
        print()

def main():
    print("Data contents:", os.listdir(data_path))
    print("Train contents:", os.listdir(training_path))
    print("Validation contents:", os.listdir(validation_path)) 

    # _, ax = plt.subplots(1, 4, figsize=(15,60))  # to show 4 images side by side, make a "1 row x 4 column" axes
    # ax[0].imshow(Image.open("../data/train/pancakes/p-00001.jpeg"))  # show the chihuahua in the first column
    # ax[1].imshow(Image.open("../data/train/pancakes/p-00002.jpeg"))  # show the chihuahua in the second column
    # ax[2].imshow(Image.open("../data/train/waffles/w-00001.jpeg"))   # show the muffin in the third column
    # ax[3].imshow(Image.open("../data/train/waffles/w-00002.jpeg"))   # show the muffin in the fourth column
    # plt.show()


    loss_function = nn.CrossEntropyLoss()   # apparently the most common error function in deep learning? Note: need to learn more about this
    optimizer = optim.SGD(model.parameters(), lr=0.1)    # stochastic gradient descent, with learning rate of 0.1? Note: need to learn more about this

    train_model(model, dataloaders, loss_function, optimizer, num_epochs=20)

    # get all the images from our validation sets
    validation_img_paths = glob("../data/validation/**/*.jpg", recursive=True)
    images = [Image.open(img_path) for img_path in validation_img_paths]

    # put all the images together to run through our model
    validation_batch = torch.stack( [validation_transforms(img).to(device) for img in images])
    pred_logits_tensor = model(validation_batch)
    pred_probs = pred_logits_tensor.cpu().data.numpy()

    # show the probabilities for each picture
    fig, axs = plt.subplots(6, 5, figsize=(20, 20))
    for i, img in enumerate(images):
        ax = axs[floor(i/5)][i % 5]
        ax.axis('off')
        ax.set_title("{:.0f}% Pancake, {:.0f}% Waffle".format(100*pred_probs[i,0], 100*pred_probs[i,1]), fontsize=12)
        ax.imshow(img)
        
    plt.show()

if __name__ == "__main__":
    main()





