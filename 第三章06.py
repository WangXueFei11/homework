# -*- coding: utf-8 -*-
score = int(input())
if score >= 90:
    print("优秀")
elif score < 90 and score >= 75:
    print("良好")
elif score < 75 and score >= 60:
    print("合格")
elif score < 60:
    print("不合格")
