# Python Q & A

> 主要参考：
>
> - [python_interview_question(4.7k stars)](https://github.com/kenwoodjw/python_interview_question#%E6%96%87%E4%BB%B6%E6%93%8D%E4%BD%9C)
>
> - [interview_python (15.1k stars)](https://github.com/taizilongxu/interview_python)
>
> - [python-interview (31 stars)](https://github.com/wangyitao/python-interview)
>
> - [python -81](https://developer.aliyun.com/article/656961)
> - [stackoverflow about python](https://taizilongxu.gitbooks.io/stackoverflow-about-python/content/)
> - [廖雪峰Python](https://www.liaoxuefeng.com/wiki/1016959663602400/1017328655674400)

[TOC]

## 理论类

### Q1：函数可变参数`*args` 和 `**kwargs`

> 参考网站：http://showteeth.tech/posts/38814.html

这两个参数都是表示给函数传不定数量的参数（不确定最后使用这个函数的时候会传递多少参数，也叫可变参数），两者的差异主要是在：

- `*args`：接受不定量的非关键字参数，如`test('Hello','Welcome')`
- `**kwargs`：接受不定量个关键字参数，如`test(x=1,y=2)`

`*args`和`kwargs`参数关键的是最前面的`*`和`**`，至于后面的字母`args`和`kwargs`只是约定俗称的叫法。

此外，可变参数在函数调用时一定在普通参数之后，如果调换顺序会报错。

### Q2：`super().__init__()`

`super().__init__()`，就是继承父类的`init`方法。

python`class`继承时，只能继承父类中的函数，而不能继承父类中的属性。

```markdown
class Root(object):
  def __init__(self):
      self.x= '这是属性'

  def fun(self):
  	#print(self.x)
      print('这是方法')
      
class A(Root):
  def __init__(self):
      print('实例化时执行')

test = A()		#实例化类
test.fun()	#调用方法
test.x		#调用属性
```

输出：

```markdown
Traceback (most recent call last):

实例化时执行

这是方法

  File "/hom/PycharmProjects/untitled/super.py", line 17, in <module>

    test.x  # 调用属性

AttributeError: 'A' object has no attribute 'x'
```

可以看到，此时父类的方法继承成功，但是父类的属性并未继承。

```markdown
class Root(object):
  def __init__(self):
      self.x = '这是属性'

  def fun(self):
      print(self.x)
      print('这是方法')


class A(Root):
  def __init__(self):
      super(A,self).__init__()
      print('实例化时执行')


test = A()  # 实例化类
test.fun()  # 调用方法
test.x  # 调用属性
```

输出：

```markdown
实例化时执行

这是属性

这是方法
```

==注==：`super()`在python2/3中的区别：

- Python3可以直接写成`super().方法名(参数)`
- Python2必须写成`super(父类,self).方法名(参数)`

### Q3：私有属性 & 私有方法

双下划线开头，声明该属性或该方法为私有，不能在类的外部被使用或直接访问；

一种情况除外，可以使用`object._class__attrname(对象名.\_类名\_\_私有属性名)访问属性。

```text
class JustCounter:
    __secretCount = 0  # 私有变量
    publicCount = 0    # 公开变量
 
	def count(self):
        self.__secretCount += 1
        self.publicCount += 1
        print self.__secretCount
 
counter = JustCounter()
counter.count()
counter.count()
print counter.publicCount
print counter.__secretCount  # 报错，实例不能访问私有变量
```

```text
class Runoob:
    __site = "www.runoob.com"

runoob = Runoob()
print runoob._Runoob__site
```

### Q4：单下划线、双下划线、头尾下划线

- \_\_foo\_\_: 定义的是特殊方法，一般是系统定义名字 ，类似 \_\_init\_\_() 之类的。
- **_foo**: 以单下划线开头的表示的是 protected 类型的变量，即保护类型只能允许其本身与子类进行访问，不能用于 **from module import \***
- **__foo**: 双下划线的表示的是私有类型(private)的变量, 只能是允许这个类本身进行访问了。

### Q5：`\#!/usr/bin/env python` & `# -*- coding:utf-8 -*-`

- `\#!/usr/bin/env python`：指定[/usr/bin/env]目录下的python解释器执行python脚本

- `# -*- coding:utf-8 -\*-`：

  Python中默认的编码格式是 ASCII 格式，在没修改编码格式时无法正确打印汉字，在读取中文时会报错。Python3.X 源码文件默认使用utf-8编码，可以正常解析中文，一般而言，都会声明为utf-8编码。

### Q6：参数传递

```python
# Case 1
a = 1
def fun(a):
    print "func_in",id(a)   # func_in 41322472
    a = 2
    print "re-point",id(a), id(2)   # re-point 41322448 41322448
print "func_out",id(a), id(1)  # func_out 41322472 41322472
fun(a)
print a  # 1
```

```python
# Case 2
a = []
def fun(a):
    print "func_in",id(a)  # func_in 53629256
    a.append(1)
print "func_out",id(a)     # func_out 53629256
fun(a)
print a  # [1]
```

对象有两类：

- 可更改对象（mutable）：`list`、`dict`、`set`。总是会按引用传入函数，函数代码组中对变量数据结构的任何改变都会反映到调用代码中。
- 不可更改对象（immutable）：`string`、`tuples`、`numbers`。总是会按值传入函数，函数中对变量的任何修改是这个函数私有的，不会反映到调用代码中。

### Q7：元类（metaclass）

> https://taizilongxu.gitbooks.io/stackoverflow-about-python/content/part/2.html

### Q8：`@staticmethod` & `@classmethod`

> [参考](https://blog.csdn.net/GeekLeee/article/details/52624742?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-52624742-blog-9615239.pc_relevant_multi_platform_whitelistv3&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-52624742-blog-9615239.pc_relevant_multi_platform_whitelistv3&utm_relevant_index=1)

Python中，要使用某个类的方法或属性，首先需要实例化一个类再调用方法或属性。而使用`@staticmethod`或`@classmethod`，就可以不用实例化，直接`类名.方法名()`来调用。这样有利于组织代码，把某些应该属于某个类的函数给放到那个类去，同时有利于命名空间的整洁。

**两者区别**

- `@staticmethod`不需要表示自身对象的`self`或自身类的`cls`参数，就和使用普通函数一样，无法访问类的对象和属性，通常包含与类有关的逻辑代码。
- `@classmethod`也不需要`self`参数，但是第一个参数需要是表示自身类的`cls`参数。可以访问类的对象和属性，可以修改类属性，也可以实例化对象。

```python
def foo(x):
    print "executing foo(%s)"%(x)

class A(object):
    def foo(self,x):
        print "executing foo(%s,%s)"%(self,x)

    @classmethod
    def class_foo(cls,x):
        print "executing class_foo(%s,%s)"%(cls,x)

    @staticmethod
    def static_foo(x):
        print "executing static_foo(%s)"%x

a=A()
```

|       | 实例方法 | 类方法         | 静态方法        |
| ----- | -------- | -------------- | --------------- |
| a=A() | a.foo(x) | a.class_foo(x) | a.static_foo(x) |
| A     | 不可用   | A.class_foo(x) | A.static_foo(x) |

**应用场景**

- `@classmethod`

  ```python
  # 不使用@classmethod 调用类属性时只能先实例化一个对象。
  # 类代码的关系扩散到了类定义的外面，造成了代码维护的困难。
  def A(cls_obj):
      return cls_obj.param
  class Myclass(object):
      param = 0
      def __init__(self):
          Myclass.param = Myclass.param + 1
  b = Myclass()
  print(A(b))
  ```

  ```python
  # 使用@classmethod 与上述代码等效。
  class Myclass(object):
      param = 0
      def __init__(self):
          Myclass.param = Myclass.param + 1
      @classmethod
      def A(cls):
          return cls.param
  print(Myclass.A())
  ```

- `@staticmethod`

  ```python
  # 不使用@staticmethod
  # 经常有一些和类有关的功能但是运行时又不需要实例和类参与的情况下需要用到静态方法。比如更改环境变量或者修改其他类的属性。
  Ind = 'ON'
  def check():
      return Ind=='ON'
  class Myclass(object):
      def __init__(self,param):
          self.param = param
      def pparam(self):
          if check():
              print("param of Myclass", self.param)
  ```

  ```python
  # 使用@staticmethod
  IND = "ON"
  class Myclass(object):
      def __init__(self, param):
          self.param = param
      @staticmethod
      def check():
          return IND=="ON"
      def pparam(self):
          if check():
              print("param of Myclass", self.param)
  
  ```

### Q9：类变量和实例变量

```python
class Test(object):  
    num_of_instance = 0  
    def __init__(self, name):  
        self.name = name  
        Test.num_of_instance += 1  

if __name__ == '__main__':  
    print Test.num_of_instance   # 0
    t1 = Test('jack')  
    print Test.num_of_instance   # 1
    t2 = Test('lucy')  
    print t1.name , t1.num_of_instance  # jack 2
    print t2.name , t2.num_of_instance  # lucy 2
```

- 类变量：可在类的所有实例之间共享的值（也就是说，它们不是单独分配给每个实例的）。例如上例中，`num_of_instance` 就是类变量，用于跟踪存在着多少个Test 的实例。

- 实例变量：实例化之后，每个实例单独拥有的变量。

```python
# 例子2
class Person:
    name="aaa"

p1=Person()
p2=Person()
p1.name="bbb"
print p1.name  # bbb
print p2.name  # aaa
print Person.name  # aaa
```

`p1.name="bbb"`实例调用了类变量，一开始是指向类变量的，但是在实例`p1`的作用域里把类变量的引用改变了，此时`name`变成了一个实例变量，`self.name`不再引用`Person`类的类变量`name`了。而`p2`引用的是类变量。

```python
# 例子3
class Person:
    name=[]

p1=Person()
p2=Person()
p1.name.append(1)
print p1.name  # [1]
print p2.name  # [1]
print Person.name  # [1]
```

**和可更改对象、不可更改对象类似！！**

### Q10：Python自省

自省就是面向对象的语言所写的程序在运行时,所能知道对象的类型.简单一句就是运行时能够获得对象的类型.比如`type()`,`dir()`,`getattr()`,`hasattr()`,`isinstance()`.

- `type()`：返回类型
- `dir()`：返回参数的属性、方法列表
- `getattr()`：返回对象属性值
- `hasattr()`：判断对象是否包含对应的属性
- `isinstance()`：判断一个对象是否是指定的类型

### Q11：格式化字符串

### Q12：生成器 & 迭代器

> [StackOverflow](https://taizilongxu.gitbooks.io/stackoverflow-about-python/content/part/1.html)
>
> [菜鸟教程](https://www.runoob.com/python3/python3-iterator-generator.html)

### Q13：装饰器

[StackOverflow](https://stackoverflow.com/questions/739654/how-to-make-function-decorators-and-chain-them-together)

### Q14：切片编程 AOP

### Q15：鸭子类型 Duck Typing

> Google的定义
>
> With nominative typing, an object is *of a given type* if it is declared to be.
>
> In duck typing, an object is *of a given type* if it has all methods and properties required by that type.Duck typing can be viewed as a usage-based structural equivalence between a given object and the requirements of a type.

其实动态语言是相对静态语言而言的，**静态语言**的特点是在程序执行前，代码编译时从代码中就可以知道一切，比如变量的类型，方法的返回值类型。

而**动态语言**只有等到程序运行时才知道一切，变量（严格来说叫名字，就像人的名字一样）不需要指定类型，变量本身没有任何类型信息，类型信息在对象身上，对象是什么类型，必须等到程序运行时才知道，动态类型语言的优点在于方便阅读，不需要写很多类型相关的代码；缺点是不方便调试，命名不规范时会造成读不懂，不利于理解等。

动态语言中经常提到鸭子类型，所谓鸭子类型就是：如果走起路来像鸭子，叫起来也像鸭子，那么它就是鸭子（If it walks like a duck and quacks like a duck, it must be a duck）。鸭子类型是编程语言中动态类型语言中的一种设计风格，一个对象的特征不是由父类决定，而是通过对象的方法决定的。

动态语言，如：PHP、Python、JavaScript
静态语言，如：C++、Java、Golang

### Q16：为什么Python不需要函数重载？

函数重载主要是为了解决两个问题。

1。可变参数类型。

2。可变参数个数。

另外，一个基本的设计原则是，仅仅当两个函数除了参数类型和参数个数不同以外，其功能是完全相同的，此时才使用函数重载，如果两个函数的功能其实不同，那么不应当使用重载，而应当使用一个名字不同的函数。

好吧，那么对于情况 1 ，函数功能相同，但是参数类型不同，python 如何处理？答案是根本不需要处理，因为 python 可以接受任何类型的参数，如果函数的功能相同，那么不同的参数类型在 python 中很可能是相同的代码，没有必要做成两个不同函数。

那么对于情况 2 ，函数功能相同，但参数个数不同，python 如何处理？大家知道，答案就是缺省参数。对那些缺少的参数设定为缺省参数即可解决问题。因为你假设函数功能相同，那么那些缺少的参数终归是需要用的。

好了，鉴于情况 1 跟 情况 2 都有了解决方案，python 自然就不需要函数重载了。

### Q17：新式类 & 旧式类

### Q18：Python 推导式

### Q19：类中的函数：`__new__`和`__init__`

> 参考：https://stackoverflow.com/questions/674304/why-is-init-always-called-after-new

两者都是有关类的函数。

- **作用**：当我们实例化一个类时，`__new__`控制我们如何创建一个实例，`__init__`控制我们如何初始化一个实例。

- **先后**：首先得创建一个实例，然后才能对其初始化。（`__init__`函数有`self`参数）
- **返回值**：`__new__`返回一个实例，`__init__`不返回任何值。

### Q20：各类设计模式

- 单例模式...手写单例模式 4种方法

### Q21：`nonlocal` & `Global`

```python
# 没有用nonlocal 和 global
x = 0
def outer():
    x = 1
    def inner():
        x = 2
        print("inner:", x)

    inner()
    print("outer:", x)

outer()
print("global:", x)

# inner: 2
# outer: 1
# global: 0
```

```python
# nonlocal
x = 0
def outer():
    x = 1
    def inner():
        nonlocal x
        x = 2
        print("inner:", x)

    inner()
    print("outer:", x)

outer()
print("global:", x)

# inner: 2
# outer: 2
# global: 0
```

```python
# global
x = 0
def outer():
    x = 1
    def inner():
        global x
        x = 2
        print("inner:", x)

    inner()
    print("outer:", x)

outer()
print("global:", x)

# inner: 2
# outer: 1
# global: 2
```

### Q22：Python作用域

1. Local局部作用域：能够访问局部变量的作用域，比如一个函数/方法 内部

2. Enclosing嵌套作用域：包含非局部（nonlocal）、非全局（nonglobal）变量。比如函数A中嵌套了函数B，则A中的变量就是nonlocal，B中的变量是local

   ```python
   g_count = 0  # 全局作用域
   def outer():
       o_count = 1  # 闭包函数外的函数中
       def inner():
           i_count = 2  # 局部作用域
           # 这里可以访问o_count，但不能修改
   ```

3. Global全局作用域：能够访问全局变量的作用域

4. Built-in内置作用域：能够访问内置函数和内置变量的作用域。如`max`函数、`abs`函数

![img](Python Q&A.assets/1418490-20180906153626089-1835444372.png)

### Q23：高阶函数

在Python中，函数是一个对象，类也是一个对象，一切皆为对象。

作为一个对象，所以：

- 可以把它赋值给一个变量
- 可以对此赋值
- 可以给它添加属性
- 可以作为函数参数来传递它

而高阶函数：把一个函数作为参数来传递的函数，称之为高阶函数。

==函数式编程的特点之一==：允许把函数本身作为参数传入另一个函数，还允许返回一个函数。（装饰器把函数作为返回值，也是函数式编程的一种）

```python
def add(x, y, f):
    return f(x) + f(y)
```

#### `map()`

`map()`函数接收两个参数，一个是函数，一个是`Iterable`，`map`将传入的函数依次作用到序列的每个元素，并把结果作为新的`Iterator`返回。

```python
>>> def f(x):
...     return x * x
...
>>> r = map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9])
# 这里的f当然也可以用lamda匿名函数来实现
>>> list(r)
[1, 4, 9, 16, 25, 36, 49, 64, 81]
```

`map()`把运算规则`f`抽象了，使得我们显而易见得知道代码在干什么。

#### `reduce()`

`reduce`把一个函数作用在一个序列`[x1, x2, x3, ...]`上，这个函数必须接收两个参数，`reduce`把结果继续和序列的下一个元素做累积计算，其效果就是：

```python
reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)
```

```python
# reduce() 应用例子
# 把序列[1,3,5,7,9]变成整数13579
>>> from functools import reduce
>>> def fn(x, y):
...     return x * 10 + y
...
>>> reduce(fn, [1, 3, 5, 7, 9])
13579
```

#### `filter`

`filter`函数用于过滤序列。`filter`函数接收一个函数和一个序列，把传入的函数依次作用于每个元素，然后根据返回值是`True`还是`False`决定保留还是丢弃该元素。返回一个`Iterator`。

```python
def is_odd(n):
    return n % 2 == 1

list(filter(is_odd, [1, 2, 4, 5, 6, 9, 10, 15]))
# 结果: [1, 5, 9, 15]
```

#### `sorted`

`sorted()`函数也是一个高阶函数，它可以接收一个`key`函数来实现自定义的排序，key指定的函数将作用于list的每一个元素上，并根据key函数返回的结果进行排序。例如按绝对值大小排序：

```python
>>> sorted([36, 5, -12, 9, -21], key=abs)
[5, 9, -12, -21, 36]
```

```python
# 另一个例子 为了演示 reverse参数
# sorted 默认从小到大，reverse=True时从大到小
>>> sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower, reverse=True)
['Zoo', 'Credit', 'bob', 'about']
```





### Q24：

---

## 具体问题类

### Q1：进制转换

- 十进制转二进制、八进制、十六进制，分别使用`bin()`、`oct()`、`hex()`函数。

  ```python
  # 获取用户输入十进制数
  dec = int(input("输入数字："))
   
  print("十进制数为：", dec)
  print("转换为二进制为：", bin(dec))
  print("转换为八进制为：", oct(dec))
  print("转换为十六进制为：", hex(dec))
  ```

- 二进制转十进制、八进制、十六进制

  ```python
  # 二进制可以直接转十进制
  int('0b1111', 2)
  # 二进制可以直接转八进制
  oct(0b1111)
  # 二进制需通过十进制转成十六进制
  hex(int('0b1111', 2))
  ```

- 八进制转二进制、十进制、十六进制

  ```python
  # 八进制需通过十进制转成二进制
  bin(int('016',8))
  # 八进制可转十进制
  int('67', 8)
  # 八进制需通过十进制转成十六进制
  hex(int('016',8))
  ```

- 十六进制转二进制、十进制、八进制

  ```python
  # 十六进制需通过十进制转二进制
  bin(int('0xe', 16))
  # 十六进制可转十进制
  int('0xe', 16)
  # 十六进制可转八进制
  oct(0xe)
  ```

**总结**：

- 所有进制都可直接转十进制。`int()`函数输入是字符串，输出为数字。
- 所有进制都可直接转八进制。`oct()`函数输入是数字，输出为字符串。
- 转二进制需通过十进制。`bin()`函数输入是数字，输出是字符串。
- 转十六进制需通过十进制。`hex()`函数输入是数字，输出是字符串。

