#torchvision
# Imagenet이나 CIFAR10, MNIST등과 같이 일반적으로 사용하는 데이터 셋을 위한 
# dataloader, 즉 torchvision.datasets과 이미지용 데이터 변환기인 data trans
# former인 torch.utils.data.DataLoader가 포함됨.
# 이는 boiler plate(=유사한 코드)를 매번 작성하는 것을 막아줌 

#CIFAR10
## class로  airplane, automobile, bird, cat, deer, dog, frog, horse, 
## ship, truck이 있음 
## Image Size = (# of channel, Width, Height) = (3, 32, 32) 

#Image Classifier 학습 과정 
##1. torchvision을 활용하여 CIFAR10의 Train/Test Data Set을 load하고  
##   Normalize함 
##2. Convolution Neural Network를 정의 
##3. Loss Function과 Optimizer 정의 
##4. Train Data Set을 사용해서 Convolution Neural Network 학습 
##5. Test Data Set을 사용해서 Convolution Neural Netowkr를 검사 

#1. CIFAR10을 Load하고 Normalize 

import torch
import torchvision
import torchvision.transforms as transforms 

## torchvison의 Data Set의 Output은 [0,1]의 범위를 갖는 PILImage임 
## 이를 [-1,1]의 범위로 Normalize할 것 
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        ### trnasoforms.Normalize(means of each channel, std of each channel) 
        ### -> Normalized Mean = 0 , Normalized Std = 1
        ### -> range of each channel's pixel value is [-1,1]
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) 
    ]
)

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform)

## num_workers = Data Load Multi Processing에 관한 것 
## CPU에서 작업을 GPU로 넘기기 위해 데이터를 전처리하는 과정이 너무 오래 걸리면 이 과정 중에 GPU는
## 일을 하지 않음. 즉 자원을 최대한 활용하지 못함.  반면 CPU에서 빠르게 전처리를 하여 task를 바로
## GPU에 던져줄 수 있으면 GPU는 쉬는 시간 없이 계속 일할 수 있게 됨.
## 따라서 CPU에서 작업을 빠르게 처리해야 GPU를 최대한 활용할 수 있음. 이를 위해 Single Core가 아닌
## Multi Core롤 전처리 작업을 수행하게 할 수 있음. Data Loader에서 이를 가능하게 해주는 것이 
## num_workers Paremeter임. 
## Data Processing에 무조건 많은 CPU Core를 할당해주는 게 좋지 않은 경우도 있음. 이는 특정 시스템의
## Core의 개수가 한정되어 있기 때문임. 예를들어 모든 Core를 전부 Data Load에 사용하게 된다면 다른
## 부가적인 처리를 하는데 Delay가 생기게 될 것이고 따라서 Data Loading 이외의 모든 작업에 영향을 
## 받을 수 있음. 따라서 적절한 개수를 지정해줄 필요가 있음 
## 이를 위해 학습 환경의 GPU 개수, CPU 개수, I/O 속도, 메모리 등을 고려해줘야함.
## 관련 해서 읽으면 좋은 PyTorch Site 
## https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813
## 출처 : https://jybaek.tistory.com/799
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data/',train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

##2. Convolution Neural Network 정의 
import torch.nn as nn 
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ### Input Size  = (Batch Num, # of Input Channel, Height, Width) 
        ###             = (Batch Num, 3, Height of Input, Width of Input)
        ### Output Size = (Batch Num, # of Output Channel, Height, Width) 
        ###             = (Batch Num, 6, Height of Output, Width of Width)
        ### Kernel Size = (# of Output Channel, Height, Width) = (6, 5, 5)
        ### nn.Conv2d(# of Input Channel, # of Output Channel, Kernel Height/Width, Stride, Padding,
        ###           Dilation, Groups, Bias, Padding_Mode)              
        self.conv1 = nn.Conv2d(3,6,5)
        ### Input Size  = (Batch Num, # of Input Channel, Height, Width)
        ###             = (Batch Num, 6, Height of Input, Wdith of Input)
        ### Output Size = (Batch Num, # of Output Channel, Height, Width)
        ###             = (Batch Num, 6, Height of Output, Width of Output)
        ### nn.MaxPool2d(Kernel Size, Stride, Padding, Dilation, Reutrn Indices, Ceil_Mode)
        self.pool  = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84 , 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        ### Flatten
        x=x.view(-1, 16*5*5)
        
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x 

net = Net()
        
       


##3. Loss Function과 Optimizer 정의 
### Corss-Entropy Loss와 Momentum을 갖는 SGD를 사용 
### torch.optim은 다양한 Optimization Algorithm을 구현하는 Package임 
import torch.optim as optim 

criterion = nn.CrossEntropyLoss()

###optim.SGD(params, learning rate, momentum, dampening, weight_decay, nesterov)
###Stochastic Gradient Descent를 구현(원하면 momentum 값을 지정해 줄 수 있음)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


##4. Train Data Set을 사용해서 Convolution Neural Network 학습 
###n_epochs = 2 -> Train Data Set에 대하여 2번 학습을 진행 
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(train_loader,0):
        ###Input Data
        inputs, labels = data
        
        ###Gradient를 0으로 초기화  
        optimizer.zero_grad()

        ###Forward Propagation
        outputs = net(inputs)

        ###Get Loss 
        loss = criterion(outputs, labels)

        ###Backward Propagation = 각 Tensor에 대하여 Gradient 구하기 
        loss.backward()

        ###Gradient Descent = 구한 Gradient로 Parameter 갱신하기 
        optimizer.step()

        ###통계 출력 
        running_loss += loss.item()
        if i% 2000 == 1999:
            print('[%d, %5d] loss: %.3f'%
                  (epoch + 1, i + 1, running_loss/2000))
            running_loss =0.0
print('Finished Training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

##5. Test Data Set을 사용해서 Convolution Neural Netowkr를 검사 
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' %(
    100 * correct/total))


##6. 어떤 Class를 잘 분류하고 어떤 Class를 잘 못 분류했는지 확인 
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in test_loader:
        images, labels = data 
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        ###tensor.squeeze() -> 크기가 1인 차원을 제거 
        c = (predicted == labels).squeeze()
        break
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' %(
         classes[i], 100*class_correct[i] / class_total[i]))
