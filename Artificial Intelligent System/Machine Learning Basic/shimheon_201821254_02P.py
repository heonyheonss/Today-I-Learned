# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:58:36 2023

@author: heon
"""
#1
print("1번 문제 결과입니다.")
x = ['Alice', 'Bob', 'Charlie']
k = 0

for student in x:
    k += 1
    print("%d등은 %s입니다." %(k, student))

#2
print("2번 문제 결과입니다.")
std_pnt = {"student":x, "point":[83, 70, 91]}
k = 0
for student in std_pnt["student"]:
    if std_pnt["point"][k] >= 90:
        print("%s의 성적은 A입니다." %(student))
    elif std_pnt["point"][k] >= 80:
        print("%s의 성적은 B입니다." %(student))
    elif std_pnt["point"][k] >= 70:
        print("%s의 성적은 C입니다." %(student))
    else:
        print("%s의 성적은 F입니다." %(student))
    k += 1

#3
print("3번 문제 결과입니다.")
def std_weight(height, gender):
    if gender == "남자":
        return ((height/100) ** 2)*22
    else:
        return ((height/100) ** 2)*21

height = 175
gender = "남자"
print("키 %dcm %s의 표준체중은 %.2f kg 입니다." %(height, gender, std_weight(height, gender)))
        