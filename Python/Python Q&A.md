# Python Q & A

[TOC]



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

### 私有属性 & 私有方法

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

### Q3：单下划线、双下划线、头尾下划线

- \_\_foo\_\_: 定义的是特殊方法，一般是系统定义名字 ，类似 \_\_init\_\_() 之类的。
- **_foo**: 以单下划线开头的表示的是 protected 类型的变量，即保护类型只能允许其本身与子类进行访问，不能用于 **from module import \***
- **__foo**: 双下划线的表示的是私有类型(private)的变量, 只能是允许这个类本身进行访问了。

### Q4：`\#!/usr/bin/env python` & `# -*- coding:utf-8 -*-`

- `\#!/usr/bin/env python`：指定[/usr/bin/env]目录下的python解释器执行python脚本

- `# -*- coding:utf-8 -\*-`：

  Python中默认的编码格式是 ASCII 格式，在没修改编码格式时无法正确打印汉字，在读取中文时会报错。Python3.X 源码文件默认使用utf-8编码，可以正常解析中文，一般而言，都会声明为utf-8编码。

### Q5：参数传递

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

### Q6：元类（metaclass）

### Q7：`@staticmethod` & `@classmethod`

> [参考](https://blog.csdn.net/GeekLeee/article/details/52624742?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-52624742-blog-9615239.pc_relevant_multi_platform_whitelistv3&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-52624742-blog-9615239.pc_relevant_multi_platform_whitelistv3&utm_relevant_index=1)

Python中，要使用某个类的方法或属性，首先需要实例化一个对象再调用方法或属性。而使用`@staticmethod`或`@classmethod`，就可以不用实例化，直接`类名.方法名()`来调用。这样有利于组织代码，把某些应该属于某个类的函数给放到那个类去，同时有利于命名空间的整洁。

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

### Q8：类变量和实例变量

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

### Q9：Python自省

自省就是面向对象的语言所写的程序在运行时,所能知道对象的类型.简单一句就是运行时能够获得对象的类型.比如`type()`,`dir()`,`getattr()`,`hasattr()`,`isinstance()`.

- `type()`：返回类型
- `dir()`：返回参数的属性、方法列表
- `getattr()`：返回对象属性值
- `hasattr()`：判断对象是否包含对应的属性
- `isinstance()`：判断一个对象是否是指定的类型

### Q10：格式化字符串

### Q11：生成器 & 迭代器

> [StackOverflow](https://taizilongxu.gitbooks.io/stackoverflow-about-python/content/part/1.html)
>
> [菜鸟教程](https://www.runoob.com/python3/python3-iterator-generator.html)

### Q14：装饰器

[StackOverflow](https://stackoverflow.com/questions/739654/how-to-make-function-decorators-and-chain-them-together)

### 切片编程 AOP

### 鸭子类型 Duck Typing

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

### Q17：为什么Python不需要函数重载？

函数重载主要是为了解决两个问题。

1。可变参数类型。

2。可变参数个数。

另外，一个基本的设计原则是，仅仅当两个函数除了参数类型和参数个数不同以外，其功能是完全相同的，此时才使用函数重载，如果两个函数的功能其实不同，那么不应当使用重载，而应当使用一个名字不同的函数。

好吧，那么对于情况 1 ，函数功能相同，但是参数类型不同，python 如何处理？答案是根本不需要处理，因为 python 可以接受任何类型的参数，如果函数的功能相同，那么不同的参数类型在 python 中很可能是相同的代码，没有必要做成两个不同函数。

那么对于情况 2 ，函数功能相同，但参数个数不同，python 如何处理？大家知道，答案就是缺省参数。对那些缺少的参数设定为缺省参数即可解决问题。因为你假设函数功能相同，那么那些缺少的参数终归是需要用的。

好了，鉴于情况 1 跟 情况 2 都有了解决方案，python 自然就不需要函数重载了。

### Q18：新式类 & 旧式类