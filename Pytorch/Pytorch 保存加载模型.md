# Pytorch 保存/加载模型

> 参考：https://zhuanlan.zhihu.com/p/73893187

[TOC]

## 保存加载模型基本用法

1. 保存加载整个模型

   保存整个网络模型（网络结构+权重参数）

   ```python
   torch.save(model, 'net.pkl')
   ```

   加载整个网络模型（可能比较耗时）

   ```python
   model = torch.load('net.pkl')
   ```

2. 只保存加载模型参数

   只保存模型的权重参数（速度快，占内存少）

   ```python
   torch.save(model.state_dict(), 'net_params.pkl')
   ```

   加载模型的权重参数

   ```python
   # 因为我们只保存了模型参数，所以需要先定义一个网络对象，然后再加载模型参数
   model = ClassNet()
   state_dict = torch.load('net_params.pkl')
   model.load_state_dict(state_dict)
   ```

## 保存加载自定义模型

首先需要知道我们保存的`net.pkl`到底保存了些什么，然后我们才能知道自定义需要保存的内容。

`net.pkl`是一个字典，通常包含如下内容：

1. 网络结构：输入尺寸、输出尺寸以及隐藏层信息，以便能够在加载时重建模型。
2. 模型的权重参数：包含各网络层训练后的可学习参数，可以在模型实例上调用`state_dict()`方法获取，比如前面介绍只保存模型权重参数时用到的`model.state_dict()`。
3. 优化器参数：有时保存模型的参数需要稍后接着训练，那么就必须保存优化器的状态和其所使用的超参数，也是在优化器实例上调用`state_dict()`方法来获取这些参数。
4. 其他信息：有时我们需要保存一些其他的信息，比如`epoch`、`batch_size`等超参数。

自定义需要保存的内容：

```python
# saving a checkpoint assuming the network class named ClassNet
checkpoint = {
    'model': ClassNet(),
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch':epoch
}
torch.save(checkpoint, 'checkpoint.pkl')
```

加载保存的自定义的内容

```python
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = TheOptimizerClass()
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    return model
model = load_checkpoint('checkpoint.pkl')
```

如果加载模型只是为了进行推理测试，则将每一层的 `requires_grad` 置为 `False`，即固定这些权重参数；还需要调用 **`model.eval()`** 将模型置为测试模式，主要是将 `dropout` 和 `batch normalization` 层进行固定，否则模型的预测结果每次都会不同。

如果希望继续训练，则调用 `model.train()`，以确保网络模型处于训练模式。

---

`state_dict()` 也是一个Python字典对象，`model.state_dict()` 将每一层的可学习参数映射为参数矩阵，其中只包含具有可学习参数的层(卷积层、全连接层等)。

```python
# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    # Initialize model
    model = TheModelClass()

    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
```

输出为：

```markdown
Model's state_dict:
conv1.weight            torch.Size([8, 3, 5, 5])
conv1.bias              torch.Size([8])
bn.weight               torch.Size([8])
bn.bias                 torch.Size([8])
bn.running_mean         torch.Size([8])
bn.running_var          torch.Size([8])
bn.num_batches_tracked  torch.Size([])
conv2.weight            torch.Size([16, 8, 5, 5])
conv2.bias              torch.Size([16])
fc1.weight              torch.Size([120, 400])
fc1.bias                torch.Size([120])
fc2.weight              torch.Size([10, 120])
fc2.bias                torch.Size([10])
Optimizer's state_dict:
state            {}
param_groups     [{'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [139805696932024, 139805483616008, 139805483616080, 139805483616152, 139805483616440, 139805483616512, 139805483616584, 139805483616656, 139805483616728, 139805483616800]}]
```

可以看到 `model.state_dict()` 保存了卷积层，BatchNorm层和最大池化层的信息；而 `optimizer.state_dict()` 则保存的优化器的状态和相关的超参数。

## 跨设备保存加载模型

1. Save on GPU, Load on CPU

   ```python
   device = torch.device('cpu')
   model = TheModelClass()
   model.load_state_dict(torch.load('net_params.pkl', map_location=device))
   ```

2. Save on GPU, Load on GPU

   ```python
   device = torch.device("cuda")
   model = TheModelClass()
   model.load_state_dict(torch.load('net_params.pkl'))
   model.to(device)
   ```

   在这里使用 `map_location` 参数不起作用，要使用 `model.to(torch.device("cuda"))` 将模型转换为CUDA优化的模型。

3. Save on CPU, Load on GPU

   ```python
   device = torch.device("cuda")
   model = TheModelClass()
   model.load_state_dict(torch.load('net_params.pkl', map_location="cuda"))
   model.to(device)
   """
   如果要指定哪个GPU，将cuda替换为cuda:id
   """
   ```

## CUDA的用法

在PyTorch中和GPU相关的几个函数：

```python
import torch

# 判断cuda是否可用；
print(torch.cuda.is_available())

# 获取gpu数量；
print(torch.cuda.device_count())

# 获取gpu名字；
print(torch.cuda.get_device_name(0))

# 返回当前gpu设备索引，默认从0开始；
print(torch.cuda.current_device())

# 查看tensor或者model在哪块GPU上
print(torch.tensor([0]).get_device())
```

输出：

```text
True
1
GeForce RTX 2080 Ti
0
```

有时我们需要把数据和模型从cpu移到gpu中，有以下两种方法：

```python
use_cuda = torch.cuda.is_available()

# 方法一：
if use_cuda:
    data = data.cuda()
    model.cuda()

# 方法二：
device = torch.device("cuda" if use_cuda else "cpu")
#device = torch.device("cuda:0" if use_cuda else "cpu") 指定哪个GPU
data = data.to(device)
model.to(device)
```