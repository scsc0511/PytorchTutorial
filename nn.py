#Neural Network는 torch.nn 패키지를 사용하여 생성할 수 있음 
#nn.Modules은 layer와 output을 반환하는 forward(input) 메서드를 포함 
#forward(input)은 간단한 Feed-Forward Network로 input을 받아 여러 계층에 차례로 전달한 후 최종 output을 리턴함 

#신경망의 일반적인 학습 과정 
#1. 학습 가능한 Parameter/Weight를 갖는 신경망을 정의 
#2. 반복해서 Dataset 입력 
#3. Forward Propagation 
#4. Loss를 계산 
#5. Backpropagation = 각 Parameter/Weight에 대한 Gradient 계산 
#6. Gradient Descent = Gradient를 바탕으로 Parameter/Weight를 갱신 

#Note
#torch.nn은 mini-batch만 지원. 즉 torch.nn 패키지 전체는 하나의 Sample이 아닌 Sample들의 mini-batch만을 입력으로
#받을 수 있음  
#만약 하나의 sample만 있어 mini-batch를 구성하는 sample의 개수에 해당하는 차원이 없다면  input.unsqueeze(0)을 사용해서
#mini-batch를 구성하는 sample의 개수에 해당하는 차원을 추가  

#Neural Netwowrk 정의 
import torch 
import torch.nn as nn 
import torch.nn.functional as F

import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        #1. Convlutional Layer
        
        # Conv1 
        # Input Image의 Size     = (Batch_Size,    Channel, Height, Width) = (Batch_Size, 1, 32, 32) 
        # Output Image의 Size    = (Batch_Size,    Channel, Height, Width) = (Batch_Size, 6, 30, 30)
        # Kernel Size            = (Kernel Number, Channel, Height, Width) = (6,          1, 3,   3)
        self.conv1 = nn.Conv2d(1,6,3)

        # Conv2
        # Input Image의 Size     = (Batch_Size,    Channel, Height, Width) = (Batch_Size,  6, 15, 15)
        # Output Image의 Size    = (Batch_Size,    Channel, Height, Width) = (Batch_Size, 16, 13, 13)
        # Kernel Size            = (Kernel Number, Channel, Height, Width) = (6,           1,  3,  3)
        self.conv2 = nn.Conv2d(6,16,3)

        #2. Fully Connected Layer
        # Affine Calcuation : y = Wx + b
        
        # FC1
        # InputSize  = Height * Width * Channel = 6*6*16 (Flatten)
        # OutputSize                            = 120
        self.fc1 = nn.Linear(16*6*6, 120) 

        # FC2
        # InputSize                             = 120
        # OutputSize                            = 84 
        self.fc2 = nn.Linear(120, 84)

        # FC3
        # InputSize                             = 84
        # OutputSize                            = 10
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #1. Convolution Block #1
        # InputSize  = (Batch_Size,  1, 32, 32)
        # OutputSize = (Batch_Size,  6, 15, 15)

        # Convolution 
        x = self.conv1(x)

        # Activation Function (ReLU)
        x = F.relu(x)

        # Max Pooling
        # Kernel/Window Size = (Height, Width) = (2,2)
        x = F.max_pool2d(x, (2,2))

        #2. Convolution Block #2
        # InputSize  = (Batch_Size,   6, 15, 15)
        # OutputSize = (Batch_Size,  16, 6,  6)

        # Convolution
        x = self.conv2(x)
      
        # Activation Function (ReLU)
        x = F.relu(x)

        # Max Pooling
        # Kernel/Window Size = (Height, Width) = (2,2)
        x = F.max_pool2d(x, (2,2))
        
        #3. Flatten 
        # InputSize  = (Batch_Size, 16, 6, 6)
        # OutputSize = (Batch_Size,   16*6*6) 
        x = x.view(-1, self.num_flat_features(x))

        #4. FC Layer
        # InputSize = (Batch_Size, 16*6*6)
        # OutpuSize = (Batch_Size,     10)
      
        # Fully Connected Layer #1
        x = self.fc1(x)

        # Activation Function(ReLU)
        x = F.relu(x)
    
        # Fully Connected Layer #2
        x = self.fc2(x)

        # Activation Function(ReLU)
        x = F.relu(x)

        # Fully Connected Layer #3
        x = self.fc3(x)

        return x 

    def num_flat_features(self, x):
        # x.size() = (Batch_Size, Channel, Height, Width) = (Batch_Size, 16, 6, 6)
        size = x.size()[1:]

        # num_features = 16 x 6 x 6
        num_features = 1
        for s in size :
            num_features *= s

        return num_features

#Network의 forward() 함수만 정의하면 Gradient를 계산하는 backward() 함수는 autograd를 사용하여 자동으로 정의됨 
#forward() 함수에서 어떠한 Tensor 연산을 사용해도 됨 
#모델의 학습 가능한 Parameter들은 net.parameters()에 의해 반환됨 
##params = 각 layer의 parameter들 
##len(params) = layer의 개수 

#1. 학습 가능한 Parameter/Weight를 갖는 신경망을 정의 
net = Net()
print(net)


##2~3. Data 입력 및 Forward Propagation
optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()  #Gradient는 누적되기 때문에 수동으로 gradient를 0으로 초기화 해줘야함 


##InpusSize = (Batch_Size, Channel, Height, Width) = (1, 1, 32, 32)
input = torch.randn(1,1,32,32)
output = net(input)

#4. Loss를 계산 

##target = label for input 
target = torch.randn(10)

##output과 같은 shape로 만들기 위함 -> output.size() = (1, 10)
target = target.view(1,-1)

##MSE Loss 계산 
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)

##5. Back Propagation
loss.backward()

##6. Gradient Descent - Stochastic Gradient Descnet 
print('before')
print(list(net.parameters())[0][0])
optimizer.step()
print('after')
print(list(net.parameters())[0][0])
