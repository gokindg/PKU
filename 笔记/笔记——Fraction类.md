week1

Fraction类



The operations for the `Fraction` type will allow a `Fraction` data object to behave like any other numeric value. 

The first method that all classes should provide is the constructor. The constructor defines the way in which data objects are created.



（下划线有隐藏的含义。）

```python
class Fraction:

    def __init__(self,top,bottom):

        self.num = top
        self.den = bottom
```

`self` is a special parameter that will always be used as a reference back to the object itself. self必须是第一个形式参数。调用时不需要写self。

```python
def show(self):
     print(self.num,"/",self.den)
>>> myf = Fraction(3,5)
>>> myf.show()
3 / 5
>>> print(myf)
<__main__.Fraction instance at 0x40bce9ac>
>>>
```



_str_is one of the set of standard methods that are provided,is the method to convert an object into a string. What we need to do is provide a “better” implementation for this method.

```python
def __str__(self):
    return str(self.num)+"/"+str(self.den)

myf = Fraction(3,5)
print(myf)
3/5
print("I ate", myf, "of the pizza")
I ate 3/5 of the pizza
myf.__str__()
'3/5'
str(myf)
'3/5'
>>>
```



We can providing the `Fraction` class with a method that overrides the addition method.In Python, this method is called `__add__` and it requires two parameters. The first, `self`, is always needed, and the second represents the other operand in the expression.

```python
def __add__(self,otherfraction):

     newnum = self.num*otherfraction.den + self.den*otherfraction.num
     newden = self.den * otherfraction.den

     return Fraction(newnum,newden)
>>> f1=Fraction(1,4)
>>> f2=Fraction(1,2)
>>> f3=f1+f2
>>> print(f3)
6/8
>>>
```

```python
def gcd(m,n):
    while m%n!=0:
        oldm=m
        oldn=n
        
        m=oldn
        n=oldm%oldn
    return n 
    
class Fraction:
    def __init__(self,top,bottom):
        self.num=top
        self.den=bottom
        
    def __str__(self):
        return str(self.num)+"/"+str(self.den)
    
    def __add__(self,otherfraction):
        newnum=self.num*otherfraction.den+self.den*otherfraction.num
        newden=self.den*otherfraction.den
        common=gcd(newnum, newden)
        return Fraction(newnum//common,newden//common)
    
a,b,c,d=map(int,input().split())
A=Fraction(a,b)
B=Fraction(c,d)
print(A+B) 

```

