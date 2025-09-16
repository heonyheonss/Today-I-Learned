# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 10:15:24 2023

@author: heon
"""

#1.1 interget 수정
x = 1
print(x)
print(type(x))
#1.2 float 수정
x = 1.5
print(x)
print(type(x))

x=1. #float
y=1 #int
z = x+y
print(z)
print(type(z))

#1.3 숫자연산
x = 2
y = 3
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x // y)
print(x % y)
print(x ** 2)

#1.4 string(문자형)
x = 'hello'
y = '1'
print(type(x))
z = int(y)
print(str(x) + y)

x = 1
y = '2'
print(x + int(y)) #3
print(str(x) + y) #'12'

x1 = 'deep'
x2 = 'deep '
y = 'learning'
print(x1 + y) # 'deeplearning'
print(x2 + y) # 'deep learning'
print(3*y)

#1.5 문자열인댁싱
print(y[1]) # 'learning' 01234567 -> 'e'
print(y[-1]) #'g'
print(y[1:4]) #ear

#1.6 boolean - True, False
t = True
f = False
print(type(t))
print(t and f)
print(t or f)
print(not t)
print(t != f)

#1.7 출력문(print)
print("hello")
print('hello')
a = 5
print(a)

a = 9
b = 10
print(f"I got {a} out of {b}.")#f""는 따옴표 안에서도 변수를 출력할 수 있게 해준다. {}사용
print("I got {} out of {}.".format(a, b))

a = 5
print('I am' + str(a) + 'years old.')
print('I am', str(a), 'years old.')

a = 4
b = 'cat'
c = 4.2123213
print(c)
print('%.1f' % c)
print('%.2f' % c)
print('%.4f' % c)
print('%d, %s, %.1f' % (a, b, c))

#1.8 라이브러리
import numpy
arr = numpy.array([1,2,3])
print(numpy.sum(arr))

import numpy as np
arr = np.array([1,2,3])
print(np.sum(arr))

from numpy import array, mean
arr = array([1,2,3])
print(mean(arr))

#2.1 List
list1 = [1,2,3,4]
print(list1, type(list1))

print(list1[0]) # 1
print(list1[-1]) # 4
print(list1[1:3]) # [2, 3]

list2 = ['math', 'english']
print(list2[0]) # math
print(list2[0][1]) # a

list3 = [1, '2', [1, 2, 3]]
print(list3)

list4 = [1, 2, 3]
list5 = [4, 5]
print(list4 + list5)
print(2*list4)

print(list4)
list4.append(list5)
print(list4)

a = [1, 2]
b = [0, 5]
c = [a, b]
print(c)
print(c[0])
print(c[0][1])
c[0][1] = 10
print(c)

a = list(range(10)) # 0~num-1 까지 나열
print(a)
a1 = list(range(1, 10)) #num1 ~ num2-1 나열
print(a1)
a2 = list(range(1, 11, 3)) #num1 ~ num2-1 까지 num3간격으로 나열
print(a2)
print(sum(a))
print(a[2:4]) # [2, 3]
print(a[2:]) # [2,3,4,5,6,7,8,9]
print(a[:2]) # [0,1]
print(a[:]) # [0,1,2,3,4,5,6,7,8,9]
print(a[:-1]) # [0,1,2,3,4,5,6,7,8]

b = [2, 10, 0, -2]
print(sorted(b)) #오름차순정렬
print(b.index(0)) #0이라는 성분의 인덱스를 알려준다
print(len(b)) # 리스트의 길이를 알려준다

# 2.2 tuple
a = (1, 2)
print(a)
print(type(a))
print(a[0])
# a[0] = 4;TypeError: 'tuple' object does not support item assignment

#2.3 Ditionary
a = {"class":['deep learning', 'machine learning'], "num_students":[40, 20]}
print(a)
print(type(a))
print(a["class"])
a["grade"] = ['A', 'B', 'C']
print(a)
print(list(a.keys())) #[class, num_student, grade]
print(a.values())
print(a.items())
print(a.get("class"))#a["class"]와 동일하지만 없는 key 값을 불러올 경우 None값 반환
print("class" in a)

#2.4 Set
animals = {'cat', 'dog'}
print('cat' in animals)
#print{'fish' in animals}
animals.add('fish')
print('fish' in animals)
print(len(animals))
animals.add('cat')
print(len(animals))
animals.remove('cat')
print(len(animals))

#3.1 if 문
x = 10
if x == 1:
    print("x == 1")
else:
    print("x != 1")

x = 10
if x == 1:
    print('x는 1입니다.')
elif x == 2:
    print('x는 2입니다.')
elif x == 3:
    print('x는 3입니다.')
else:
    print('x는 4 이상 입니다.')

# 3.2 for 문
for i in range(5):
    print(i)

animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)

animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#%d: %s' %(idx + 1, animal))

nums = [0,1,2,3,4]
squares = [x ** 2 for x in nums]
print(squares)

nums = [0,1,2,3,4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)

# 3.3 while 문
i = 0
while i < 5:
    print(i)
    i += 1

# break, continue 문
for i in range(5):
    if i == 3:
        continue
    print(i)

for i in range(5):
    if i  == 3:
        break
    print(i)

# 4. function
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1,0,1]:
    print(sign(x))

def hello(name, loud = False):
    if loud:
        print('Hello, %s!' % (name.upper()))
    else:
        print('Hello, %s' % name)

hello('Bob')
hello("Fred", loud = True)

# 5. Classes
class Greeter(object):
    def __init__(self, name):
        self.name = name
    
    def greet(self, loud=False):
        if loud:
            print('Hello, %s!' % (self.name.upper()))
        else:
            print('Hello, %s' % self.name)

g = Greeter('Fred')
g.greet()
g.greet(loud=True)
            

print("201821254 심헌")