# Pytorch Q & A

[TOC]

### Q1：`torch.nn`与`torch.nn.functional`的区别与联系

发现`nn.Conv2d()`和`nn.functional.conv2d()`都可以用来构建模型，区别与联系？

> 参考网址：https://www.zhihu.com/question/66782101

#### 相同之处

- `nn.Xxx`和`nn.funcitonal.xxx`实际功能是相同的，即`nn.Conv2d`和`nn.functional.conv2d`都是进行卷积，`nn.Dropout`和`nn.functional.dropout`都是进行dropout。
- 运行效率近乎相同。

`nn.functional.xxx`是函数接口，而`nn.Xxx`是`nn.functional.xxx`的类封装，并且`nn.Xxx`都继承于一个共同祖先`nn.Module`。这一点导致`nn.Xxx`除了具有`nn.functional.xxx`功能之外，内部附带了`nn.Module`相关的属性和方法，例如`train(), eval(),load_state_dict, state_dict `等。

#### 差别之处

- 两者的调用方式不同

  `nn.Xxx`需要先实例化并传入参数，然后以函数调用的方式调用实例化的对象并传入输入数据。

  ```python
  inputs = torch.rand(64, 3, 244, 244)
  conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
  out = conv(inputs)
  ```

  `nn.functional.xxx`同时传入输入数据和weight,bias等其他参数

  ```python
  weight = torch.rand(64, 3, 3, 3)
  bias = torch.rand(64)
  out = nn.functional.conv2d(inputs, weight, bias, padding=1)
  ```

- `nn.Xxx`继承于`nn.Module`，能够很好的与`nn.Sequential`结合使用，而`nn.functional.xxx`无法与`nn.Sequential`结合使用。

  ```python
  fm_layer = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=64),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),
      nn.Dropout(0.2)
  )
  ```

- `nn.Xxx`不需要自己定义和管理weight；而`nn.functional.xxx`需要自己定义weight，每次调用时都需要手动传入weight，不利于代码复用。

  使用`nn.Xxx`定义一个CNN

  ```python
  class CNN(nn.Module):
      def __init__(self):
          super(CNN, self.__init__()
          self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=0)
          self.relu1 = nn.ReLU()
          self.maxpool1 = nn.Maxpool2d(kernel_size=2)
                
          self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=0)
          self.relu2 = nn.ReLU()
          self.maxpool2 = nn.Maxpool2d(kernel_size=2)
          
         	self.linear1 = nn.Linear(4*4*32, 10)
      def forward(self, x):
         	x = x.view(x.size(0), -1)
       	out = self.maxpool1(self.relu1(self.cnn1(x)))
          out = self.maxpool2(self.relu2(self.cnn2(out)))
          out = self.linear2(out.view(x.size(0), -1))
          return out
  ```

  使用`nn.functional.xxx`定义与上面相同的CNN

  ```python
  class CNN(nn.Module):
      def __init__(self):
          super(CNN, self).__init__()
          
          self.cnn1_weight = nn.Parameter(torch.rand(16, 1, 5, 5))
          self.bias1_weight = nn.Parameter(torch.rand(16))
          
          self.cnn2_weight = nn.Parameter(torch.rand(32, 16, 5, 5))
          self.bias2_weight = nn.Parameter(torch.rand(32))
          
          self.linear1_weight = nn.Parameter(torch.rand(4 * 4 * 32, 10))
          self.bias3_weight = nn.Parameter(torch.rand(10))
          
      def forward(self, x):
          x = x.view(x.size(0), -1)
          out = F.conv2d(x, self.cnn1_weight, self.bias1_weight)
          out = F.relu(out)
          out = F.max_pool2d(out)
          
          out = F.conv2d(x, self.cnn2_weight, self.bias2_weight)
          out = F.relu(out)
          out = F.max_pool2d(out)
          
          out = F.linear(x, self.linear1_weight, self.bias3_weight)
          return out
  ```

  > 好像共享参数的话，后者方便？

官方推荐：具有学习参数的（例如，conv2d, linear, batch_norm)采用`nn.Xxx`方式，没有学习参数的（例如，maxpool, loss func, activation func）等根据个人选择使用`nn.functional.xxx`或者`nn.Xxx`方式。

关于dropout，知乎作者强烈推荐使用`nn.Xxx`方式，因为一般情况下只有训练阶段才进行dropout，在eval阶段都不会进行dropout。使用`nn.Xxx`方式定义dropout，在调用`model.eval()`之后，model中所有的dropout layer都关闭，但以`nn.function.dropout`方式定义dropout，在调用`model.eval()`之后并不能关闭dropout。

### Q2：`model.named_parameters()`与`model.parameters()`

区别：

- `named_parameters()`：`for name, param in model.named_parameters():`可以访问`name`属性。

- `parameters()`：`for index, param in enumerate(model.parameters()):`

联系：

均可以更改参数的可训练属性`param.requires_grad`。

### Q3：Logits的含义

原公式：$logit(p)=log\frac{p}{1-p}$，Logti=Logistic Unit

logtis可以看作神经网络输出的**未经过归一化(Softmax等）的概率**，一般是全连接层的输出。将其结果用于分类任务计算loss时，如求cross_entropy的loss函数会设置from_logits参数。

因此，当from_logtis=False（默认情况）时，可以理解为输入的y_pre不是来自logtis，那么它就是已经经过softmax或者sigmoid归一化后的结果了；当from_logtis=True时，则可以理解为输入的y_pre是logits，那么需要对它进行softmax或者sigmoid处理再计算cross entropy。

```python
y_pre = np.array([[5, 5], [2, 8]], dtype=np.float)   # 神经网络的输出，未概率归一化
y_true = np.array([[1, 0], [0, 1]], dtype=np.float)
loss = tf.losses.binary_crossentropy(y_true=y_true, y_pred=y_pre, from_logits=True)
print(loss)

y_pre = tf.nn.sigmoid(y_pre)
loss = tf.losses.binary_crossentropy(y_true=y_true, y_pred=y_pre, from_logits=False)
print(loss)
```

### Q4：`requires_grad`、`grad_fn`、`with torch.no_grad()`、`retain_grad`

- `requires_grad`是pytorch中tensor的一个属性，如果`requires_grad=True`，在进行反向传播的时候会记录该tensor梯度信息。

- `grad_fn`也是tensor的一个属性，它记录的是tensor的运算信息，如c=a+b，那么`c.grad_fn=<AddBackward0>`。

- `with torch.no_grad():`，在该模块下，所有计算得出的tensor的`requires_grad`都自动设置为False，不参与反向传播。

- `retain_graph`：如果设置为`False`，计算图中的中间变量在计算完后就会被释放。Pytorch的机制是每次调用`.backward()`都会free掉所有buffers，模型中可能有多次`backward()`，而前一次`backward()`存储在buffer中的梯度，会因为后一次调用`backward()`被free掉。

  ==注==：

  关于`requires_grad`和`torch.no_grad()`，有两篇文章写的很好，下附链接。大意是：`with torch.no_grad()`时会**阻断该传播路线的梯度传播**，而`requires_grad`置`False`时，**不会阻断该路线的梯度传播，只是不会计算该参数的梯度**。

  在使用`with torch.no_grad()`时，虽然可以少计算一些tensor的梯度而减少计算负担，但是如果有`backward`的时候，很可能会有错误的地方，**要么很确定没有`backward`就可以用**，要么在显卡允许的情况下就不用`with torch.no_grad()`，以免出现不必要的错误。

> requires_grad与torch.no_grad()：
>
> 1. https://its201.com/article/laizi_laizi/112711521
> 2. https://www.cxyzjd.com/article/weixin_43145941/114757673
>
> retain_graph：
>
> 1. https://www.cnblogs.com/luckyscarlett/p/10555632.html

### Q5：`tensor.detach()` & `tensor.detach_()`

当我们在训练网络的时候可能希望保持一部分的网络参数不变，只对一部分的参数进行调整；或者只训练部分分支网络，并不让其梯度对主网络的梯度造成影响，这时候我们可以使用`detach()`函数来切断一些分支的反向传播。

1. `tensor.detach()`

   返回一个从当前计算图中分离下来的新的tensor，但是仍指向**原变量的存放位置**，不同之处只是`requires_grad`为false，得到的这个tensor不需要计算其梯度，不具有grad。即使之后重新将它的requires_grad置为true,它也不会具有梯度grad。

   - 正常的例子：

   ```python
   import torch
   a = torch.tensor([1, 2, 3.], requires_grad = True)
   print(a.grad)
   
   out = a.sigmoid()
   out.sum().backward()
   print(a.grad)
   ```

   输出：

   ```markdown
   None
   tensor([0.1966, 0.1050, 0.0452])
   ```

   - 使用`detach()`分离tensor但没有更改tensor，不会影响`backward()`

   ```python
   import torch
   a = torch.tensor([1, 2, 3.], requires_grad=True)
   print(a.grad)
   out = a.sigmoid()
   print(out)
   
   c = out.detach()
   print(c)
   
   out.sum().backward()
   print(a.grad)
   # c 和 out 的区别就是前者是没有梯度的，后者是有梯度的
   ```

   输出：

   ```markdown
   None
   tensor([0.7311, 0.8808, 0.9526], grad_fn=<SigmoidBackward>)
   tensor([0.7311, 0.8808, 0.9526])
   tensor([0.1966, 0.1050, 0.0452])
   ```

   下面两种情况将会报错：

   - 当使用`detach()`分离tensor，然后用分离出来的tensor去求导数，会影响`backward()`，出现错误。
   - 当使用`detach()`分离tensor并且更改这个tensor时，即使再对原来的out求导数，会影响`backward()`，出现错误。

   ==只训练部分模型==：

   - 假如A网络输出了一个Tensor类型的变量a, a要作为输入传入到B网络中，如果我想通过损失函数反向传播修改B网络的参数，但是不想修改A网络的参数，这个时候就可以使用detcah()方法

     ```python
     a = A(input)
     a = a.detach()
     
     b = B(a)
     loss = criterion(b, target)
     loss.backward()
     ```

   - 假如A网络输出了一个Tensor类型的变量a, a要作为输入传入到B网络中，如果我想通过损失函数反向传播修改A网络的参数，但是不想修改B网络的参数，这个时候又应该怎么办呢？

     ```python
     for param in B.parameters():
     	param.requires_grad = False
     
     a = A(input)
     b = B(a)
     loss = criterion(b, target)
     loss.backward()
     ```

2. `tensor.detach_()`

   将一个tensor从创建它的图中分离，并把它设置成叶子tensor

   相当于变量之间的关系本来是**x -> m -> y**,这里的叶子tensor是x，但是这个时候对m进行了m.detach_()操作,其实就是进行了两个操作：

   - 将m的grad_fn的值设置为None，这样m就不会再与前一个节点x关联，这里的关系变成x, m ->y，此时m变成叶子节点
   - 然后会将m的`requires_grad`设置为False，这样对y进行`backward()`时就不会求m的梯度。

**总结**

`detach()`和`detach_()`很像，两个的区别就是`detach_()`是对本身的更改，`detach()`则是生成了一个新的tensor

比如x -> m -> y中如果对m进行`detach()`，后面如果反悔想还是对原来的计算图进行操作还是可以的

但是如果是进行了`detach_()`，那么原来的计算图也发生了变化，就不能反悔了。



### Q6：使用Python遇到的问题

> [Pytorch有哪些坑](https://www.zhihu.com/question/67209417)

### 
