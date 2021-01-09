#torch.Tensor Class의 
#.requires_grad 속성을 True로 설정하면, 해당 tensor에서 이루어진 모든 연산을 추적 
#.backward()를 호출하면 모든 gradient를 자동으로 계산 
#.backward()를 통해 계산된 gradient는 .grad 속성에 누적됨 
#.detach()를 호출하면 tensor가 기록된 연산을 추적하는 것을 중지함   

#with torch.no_grad()로 코드 블럭을 감싸면 기록을 추적하는 것과 메모리를 사용하는 것을 방지함 
#이는gradient는 필요 없지만, requires_grad=True로 설정할 수 있기 때문에 학습 가능한 Parameter를 갖는 모델을 평가할 때 유용함 

#Tensor Class와 Function Class는 서로 연결되어 있으며 모든 연산 과정을 encode하여 Acyclic Graph를 생성함 
#Tensro의 grad_fn 속성을 통해 해당 Tensor를 생성한 Function을 참조할 수 있음
#사용자가 만든 Tensor는 예외적으로 grad_fn의 값이 None임 

#Derivative를 계산하기 위해서 Tensor Class의 .backward()를 호출 
#Tensor가 scalar인 경우 backward에 argument를 지정해줄 필요가 없으나 multi dimensional matrix인 경우 tensor의 shape를 
#gradient의 인자로 지정해 줘야함 
import torch 

x = torch.ones(2,2, requires_grad=True)
#print(x)

y = x+2 
#print(y)
#y는 연산에 의해 생성된 Tensor이므로 grad_fn 속성에 y Tensor를 생성한 Function의 값이 저장되어 있음   
#print(y.grad_fn)

a = torch.randn(2,2)
a = ((a*3) / (a-1))
#print(a.requires_grad)

#.requires_grad()는 기존의 Tensor의 requires_grad의 값을 변경 
#requires_grad의 디폴트 값은 false임 
a.requires_grad_(True)
#print(a.requires_grad)

#b는 연산에 의해 생성된 Tensor이므로 grad_fn 속성에 b Tensor를 생성한 Function의 값이 저장되어 있음 
b = (a*a).sum()
#print(b.grad_fn)

z=y*y*3
out = z.mean()
#print(z,out)

#out은 scalar이기 때문에 out.backward()는 out.backward(torch.tensor(1.))과 동일 
#.backward()로 gradient를 계산할 때 chain rule을 근거로 하여 vector-Jacobain product를 사용 
out.backward()
#print(x.grad)

x = torch.randn(3, requires_grad=True)

y = x*2
while y.data.norm() < 1000:
    y = y*2

#print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

#print(y)
#print(x.grad)

#print(x.requires_grad)
#print((x**2).requires_grad)

#with torch.no_grad():
#    print((x**2).requires_grad)

print(x.requires_grad)
y = x.detach()
print(x)
print(y)
print(y.requires_grad)
print(x.eq(y).all())

