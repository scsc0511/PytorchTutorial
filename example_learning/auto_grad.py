# autograd 패키지 
# 자동 미분 기능을 제공하여 신경망에서 Backpropagation 연산을 자동화할 수 
# 있게 해줌 
# Computation Graph를 정의
# Computation Graph의 Node는 Tensor, Edge는 Input Tensor로 부터 Output 
# Tensor를 만들어내는 함수가 됨. 이 Computation Graph를 통해 Backprop
# agation을 하면 Gradient를 쉽게 계산할 수 있음 
# Tensor x가 x.requires_grad의 값이 True라면 x.grad는 x의 Gradient 값을
# 갖는 또 다른 Tensor임. 

import torch 

dtype = torch.float
device = torch.device("cuda:0")

# N = BATCH_SIZE
# D_in = INPUT_SIZE
# H = HIDDEN_SIZE
# D_out = OUTPUT_SIZE
N, D_in, H, D_out = 64, 1000, 100, 10

# Input과 Output에 대응하는 Random으로 샘플링한 값을 갖는 Tensor를 생성
# requires_grad를 False로 설정하여 Backpropagation 중에 Tensor의 Gradient를
# 계산할 필요가 없음을 나타냄.(requires_grad의 디폴트 값이 False)
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype) 

# Weight에 대응하는 Random으로 샘플링한 값을 갖는 Tensor를 생성 
# requires_grad=True로 설정하여 Backpropagation 중에 이 Tensor의 Gradient
# 를 계산할 필요가 있음을 나타냄. 
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6

for t in range(500):
    # Forward Propagation : Tensor 연산을 사용하여 예상되는 y값을 계산 
    #                       이는 Tensor를 사용한 Forward Propagation과
    #                       동일하지만 Backpropgation 단계를 별도로 구
    #                       현하지 않아도 되므로 중간 값들에 대한 참조
    #                       를 갖고 있을 필요가 없음 
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Tensor 연산을 사용하여 Loss를 계산하고 출력 
    # loss는 (1,) 형태의 Tensor이며, loss.item()은 
    # loss의 Scalar 값
    loss = (y_pred - y).pow(2).sum()
    if t%100 == 99:
        print(t, loss.item())

    # autgograd를 사용하여 Backpropagation 계산을 수행 
    # 이는 requires_grad=True를 갖는 모든 Tensor에 대해
    # Loss의 Gradient를 계산함. 이후 w1.grad와 w2.grad
    # 는 각각 w1과 w1 각각에 대한 Loss의 Gradient를 갖
    # 는 Tensor가 됨
    loss.backward()

    # Gradient Descent를 사용하여 Weight를 수동으로 갱신 
    # torch.no_grad()로 감싸는 이유는 Weight의 gradient
    # 를 추적할 필요가 없는데 requires_grad의 값이 True
    # 이기 때문 
    # Gradient Descent를 위해 아래의 방법을 사용하는 대신
    # gorch.optim.SGD를 사용할 수도 있음 
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad 

        # Weigth 갱신 후에는 수동으로 Gradient를 0으로 만
        # 들어줌 
        w1.grad.zero_()
        w2.grad.zero_()

 
 
# autograd함수 정의하기 
# 내부적으로 autograd의 primitive operator는 실제로 Tensor를 조작하는 2개
# 의 함수(forward(), backward())임. forward 함수는 Input Tensor로 부터 
# Output Tensor를 계산함. backward 함수는 어떤 Scalar 값에 대한 출력 Ten
# sor의 Gradient를 전달받고 동일한 Scalar 값에 대한 Input Tensor의 변화도  
# 를 계산함. 
# PyTorch에서 torch.autograd.Function의 Subclass를 정의하고 forward와 
# backward 함수를 구현함으로써 사용자 정의 autograd 연산자를 쉽게 정의
# 할 수 있음. 그 후 instance를 생성하고 이를 함수처럼 호출하여 입력 데
# 이터를 갖는 Tensor를 전달하는 식으로 새로운 autograd 연산자를 사용할 수
# 있음

# torch.autograd.Function을 상속받아 사용자 정의 autograd Function을 구현하
# 고 Tensor 연산을 하는 Forward와 Backward 함수를 구현  
class MyReLU(torch.autograd.Function):
   
    @staticmethod 
    def forward(ctx, inp):
    ### Forward Propagation에서는 Input Tensor를 받아 Output Tensor를 
    ### 반환. ctx는 Context Object로 Backward propagation을 위한 정보
    ### 를 저장하는데 사용함. ctx.save_for_backward method를 사용함으
    ### 써 Backpropagation에서 사용할 어떤 객체도 cache해 둘 수 있음  
        ctx.save_for_backward(inp) # backward propagation에서 loss를 Inp
                                   # ut Tensor로 미분할 때 사용하기 위해 
                                   # 필요(Chain Rule에 따라 Output Tensor
                                   # 의 Gradient를 Input Tensor로 미분한 
                                   # 값과 Loss를 Output Tensor로 미분한 
                                   # 값을 곱함으로써 해당 Tensor에 대한
                                   # Gradient 계산이 이루어짐   
        return inp.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):

    ### Backward Propagation 단계에서는 Output Tensor 에 대한 Loss의 Gra
    ### ident를 계산하는 Tensor를 입력받고 이를 바타응로 Input Tensor에 
    ### 대한 Loss의 Gradient를 계산함

        inp, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[inp < 0] = 0    # ReLU는 0보다 클 때 gradient가 1이고 0
                                   # 보다 작을 때 gradient가 0이므로 ReLU
                                   # 에 대한 Derivative 값을 구하기 위해서 
                                   # Input 값이 0보다 큰 값들은 그대로 두고 
                                   # 0보다 작은 값들은 0으로 만들어 주면 됨
                                   # (Chain Rule에 의해 지금 까지 구해진 G
                                   # radeint와 곱해지므로) 
        return grad_input

dtype = torch.float
device = torch.device("cuda:0")

# N = BATCH_SIZE
# D_in = INPUT_SIZE
# H = HIDDEN_SIZE
# D_out = OUTPUT_SIZE
N, D_in, H, D_out = 64, 1000, 100, 10

# Input Tensor와 Output Tensor에 대응하는 Random으로 Sampling된 값을 갖는
# Tensor를 생성  
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Weight를 저장하기 위해 Random으로 Sampling된 값을 갖는 Tensor를 생성함
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 사용자 정의 Function을 적용하기 위해 Function.apply 메소드를 사용함
    relu = MyReLU.apply 

    # Forward Propagation : Tensor 연산을 사용하여 예상되는 y 값을 계산 
    #                       사용자 정의 autograd 연산을 사용하여 ReLU
    #                       를 계산함 
    y_pred = relu(x.mm(w1)).mm(w2)

    # Loss를 계산하고 출력 (MSE)
    loss = (y_pred - y).pow(2).sum()

    if t%100 == 99:
        print(t, loss.item())

    # autograd를 사용하여 Backpropagation 단계를 계산함
    loss.backward()

    # Gradient Descnet를 사용하여 Weight를 갱신함. 
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad 
        
        # Weight 갱신 후에는 수동으로 Gradient를 0으로 만듦
        w1.grad.zero_()
        w2.grad.zero_() 

