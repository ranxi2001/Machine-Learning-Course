#朴素贝叶斯分类
#计算每个类别的先验概率
p_banana=500/1000
p_orange=300/1000
p_other=200/1000

#计算每个类别的条件概率
p_long_banana=400/500
p_notlong_banana=100/500
p_sweet_banana=350/500
p_notsweet_banana=150/500
p_yellow_banana=450/500
p_notyellow_banana=50/500

p_long_orange=0/300
p_notlong_orange=300/300
p_sweet_orange=150/300
p_notsweet_orange=150/300
p_yellow_orange=300/300
p_notyellow_orange=0/300

p_long_other=100/200
p_notlong_other=100/200
p_sweet_other=150/200
p_notsweet_other=50/200
p_yellow_other=50/200
p_notyellow_other=150/200

# #计算后验概率
# p_long_sweet_yellow_banana=p_long_banana*p_sweet_banana*p_yellow_banana*p_banana
# p_notlong_sweet_yellow_banana=p_notlong_banana*p_sweet_banana*p_yellow_banana*p_banana
# p_long_notsweet_yellow_banana=p_long_banana*p_notsweet_banana*p_yellow_banana*p_banana
# p_notlong_notsweet_yellow_banana=p_notlong_banana*p_notsweet_banana*p_yellow_banana*p_banana
# p_long_sweet_notyellow_banana=p_long_banana*p_sweet_banana*p_notyellow_banana*p_banana
# p_notlong_sweet_notyellow_banana=p_notlong_banana*p_sweet_banana*p_notyellow_banana*p_banana
# p_long_notsweet_notyellow_banana=p_long_banana*p_notsweet_banana*p_notyellow_banana*p_banana
# p_notlong_notsweet_notyellow_banana=p_notlong_banana*p_notsweet_banana*p_notyellow_banana*p_banana


#计算后验概率
p_long_sweet_yellow_banana=p_long_banana*p_sweet_banana*p_yellow_banana*p_banana
print("P(Long, Sweet, Yellow | Banana) =", p_long_sweet_yellow_banana)
p_notlong_sweet_yellow_banana=p_notlong_banana*p_sweet_banana*p_yellow_banana*p_banana
print("P(Not Long, Sweet, Yellow | Banana) =", p_notlong_sweet_yellow_banana)
p_long_notsweet_yellow_banana=p_long_banana*p_notsweet_banana*p_yellow_banana*p_banana
print("P(Long, Not Sweet, Yellow | Banana) =", p_long_notsweet_yellow_banana)
p_notlong_notsweet_yellow_banana=p_notlong_banana*p_notsweet_banana*p_yellow_banana*p_banana
print("P(Not Long, Not Sweet, Yellow | Banana) =", p_notlong_notsweet_yellow_banana)
p_long_sweet_notyellow_banana=p_long_banana*p_sweet_banana*p_notyellow_banana*p_banana
print("P(Long, Sweet, Not Yellow | Banana) =", p_long_sweet_notyellow_banana)
p_notlong_sweet_notyellow_banana=p_notlong_banana*p_sweet_banana*p_notyellow_banana*p_banana
print("P(Not Long, Sweet, Not Yellow | Banana) =", p_notlong_sweet_notyellow_banana)
p_long_notsweet_notyellow_banana=p_long_banana*p_notsweet_banana*p_notyellow_banana*p_banana
print("P(Long, Not Sweet, Not Yellow | Banana) =", p_long_notsweet_notyellow_banana)
p_notlong_notsweet_notyellow_banana=p_notlong_banana*p_notsweet_banana*p_notyellow_banana*p_banana
print("P(Not Long, Not Sweet, Not Yellow | Banana) =", p_notlong_notsweet_notyellow_banana)

#计算后验概率
p_long_sweet_yellow_orange=p_long_orange*p_sweet_orange*p_yellow_orange*p_orange
print("P(Long, Sweet, Yellow | Orange) =", p_long_sweet_yellow_orange)
p_notlong_sweet_yellow_orange=p_notlong_orange*p_sweet_orange*p_yellow_orange*p_orange
print("P(Not Long, Sweet, Yellow | Orange) =", p_notlong_sweet_yellow_orange)
p_long_notsweet_yellow_orange=p_long_orange*p_notsweet_orange*p_yellow_orange*p_orange
print("P(Long, Not Sweet, Yellow | Orange) =", p_long_notsweet_yellow_orange)
p_notlong_notsweet_yellow_orange=p_notlong_orange*p_notsweet_orange*p_yellow_orange*p_orange
print("P(Not Long, Not Sweet, Yellow | Orange) =", p_notlong_notsweet_yellow_orange)
p_long_sweet_notyellow_orange=p_long_orange*p_sweet_orange*p_notyellow_orange*p_orange
print("P(Long, Sweet, Not Yellow | Orange) =", p_long_sweet_notyellow_orange)
p_notlong_sweet_notyellow_orange=p_notlong_orange*p_sweet_orange*p_notyellow_orange*p_orange
print("P(Not Long, Sweet, Not Yellow | Orange) =", p_notlong_sweet_notyellow_orange)
p_long_notsweet_notyellow_orange=p_long_orange*p_notsweet_orange*p_notyellow_orange*p_orange
print("P(Long, Not Sweet, Not Yellow | Orange) =", p_long_notsweet_notyellow_orange)
p_notlong_notsweet_notyellow_orange=p_notlong_orange*p_notsweet_orange*p_notyellow_orange*p_orange
print("P(Not Long, Not Sweet, Not Yellow | Orange) =", p_notlong_notsweet_notyellow_orange)

#计算后验概率
p_long_sweet_yellow_other=p_long_other*p_sweet_other*p_yellow_other*p_other
print("P(Long, Sweet, Yellow | Other) =", p_long_sweet_yellow_other)
p_notlong_sweet_yellow_other=p_notlong_other*p_sweet_other*p_yellow_other*p_other
print("P(Not Long, Sweet, Yellow | Other) =", p_notlong_sweet_yellow_other)
p_long_notsweet_yellow_other=p_long_other*p_notsweet_other*p_yellow_other*p_other
print("P(Long, Not Sweet, Yellow | Other) =", p_long_notsweet_yellow_other)
p_notlong_notsweet_yellow_other=p_notlong_other*p_notsweet_other*p_yellow_other*p_other
print("P(Not Long, Not Sweet, Yellow | Other) =", p_notlong_notsweet_yellow_other)
p_long_sweet_notyellow_other=p_long_other*p_sweet_other*p_notyellow_other*p_other
print("P(Long, Sweet, Not Yellow | Other) =", p_long_sweet_notyellow_other)
p_notlong_sweet_notyellow_other=p_notlong_other*p_sweet_other*p_notyellow_other*p_other
print("P(Not Long, Sweet, Not Yellow | Other) =", p_notlong_sweet_notyellow_other)
p_long_notsweet_notyellow_other=p_long_other*p_notsweet_other*p_notyellow_other*p_other
print("P(Long, Not Sweet, Not Yellow | Other) =", p_long_notsweet_notyellow_other)
p_notlong_notsweet_notyellow_other=p_notlong_other*p_notsweet_other*p_notyellow_other*p_other
print("P(Not Long, Not Sweet, Not Yellow | Other) =", p_notlong_notsweet_notyellow_other)

print("--------------------------------------贝叶斯估计")
#设置平滑参数a
a = 1

#计算每个类别的条件概率
p_long_banana=(400+a)/(500+2*a)
p_notlong_banana=(100+a)/(500+2*a)
p_sweet_banana=(350+a)/(500+2*a)
p_notsweet_banana=(150+a)/(500+2*a)
p_yellow_banana=(450+a)/(500+2*a)
p_notyellow_banana=(50+a)/(500+2*a)

p_long_orange=(0+a)/(300+2*a)
p_notlong_orange=(300+a)/(300+2*a)
p_sweet_orange=(150+a)/(300+2*a)
p_notsweet_orange=(150+a)/(300+2*a)
p_yellow_orange=(300+a)/(300+2*a)
p_notyellow_orange=(0+a)/(300+2*a)

p_long_other=(100+a)/(200+2*a)
p_notlong_other=(100+a)/(200+2*a)
p_sweet_other=(150+a)/(200+2*a)
p_notsweet_other=(50+a)/(200+2*a)
p_yellow_other=(50+a)/(200+2*a)
p_notyellow_other=(150+a)/(200+2*a)

#计算后验概率
p_long_sweet_yellow_banana=p_long_banana*p_sweet_banana*p_yellow_banana*p_banana
print("P(Long, Sweet, Yellow | Banana) =", p_long_sweet_yellow_banana)
p_long_sweet_yellow_orange=p_long_orange*p_sweet_orange*p_yellow_orange*p_orange
print("P(Long, Sweet, Yellow | Orange) =", p_long_sweet_yellow_orange)
p_long_sweet_yellow_other=p_long_other*p_sweet_other*p_yellow_other*p_other
print("P(Long, Sweet, Yellow | Other) =", p_long_sweet_yellow_other)


