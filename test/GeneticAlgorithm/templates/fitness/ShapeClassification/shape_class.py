import math

#Single Block
list = [[0,0]]
sum_x = 0
sum_y = 0
for i in list:
    sum_x += i[0]
    sum_y += i[1]
sum_x /= len(list)
sum_y /= len(list)
print "SingleBlock: Grav. (", sum_x, ", ", sum_y, ")"
d_sum_x = 0
d_sum_y = 0
for i in list:
    d_sum_x += (i[0]-sum_x)*(i[0]-sum_x)
    d_sum_y += (i[1]-sum_y)*(i[1]-sum_y)
print "SingleBlock: ", len(list), ":(", d_sum_x, ", ", d_sum_y, ")"

#QuadroBlock
list = [[0,0], [1,1], [1,0], [0,1]]
sum_x = 0.0
sum_y = 0.0
for i in list:
    sum_x += i[0]
    sum_y += i[1]
sum_x /= len(list)
sum_y /= len(list)
print "SingleBlock: Grav. (", sum_x, ", ", sum_y, ")"
d_sum_x = 0.0
d_sum_y = 0.0
for i in list:
    d_sum_x += (i[0]-sum_x)*(i[0]-sum_x)
    d_sum_y += (i[1]-sum_y)*(i[1]-sum_y)
print "SingleBlock: ", len(list), ":(", d_sum_x, ", ", d_sum_y, ")"

#Cath1
list = [[0,0], [1,1], [1,0], [0,1], [-1, 1], [1, 2], [0, -1], [2, 0]]
sum_x = 0.0
sum_y = 0.0
for i in list:
    sum_x += i[0]
    sum_y += i[1]
sum_x /= len(list)
sum_y /= len(list)
print "SingleBlock: Grav. (", sum_x, ", ", sum_y, ")"
d_sum_x = 0.0
d_sum_y = 0.0
for i in list:
    d_sum_x += (i[0]-sum_x)*(i[0]-sum_x)
    d_sum_y += (i[1]-sum_y)*(i[1]-sum_y)
print "SingleBlock: ", len(list), ":(", d_sum_x, ", ", d_sum_y, ")"

#Hollow
list = [[0,0], [1,0], [2,0], [2,1], [2, 2], [1, 2], [0, 2], [0, 1]]
sum_x = 0.0
sum_y = 0.0
for i in list:
    sum_x += i[0]
    sum_y += i[1]
sum_x /= len(list)
sum_y /= len(list)
print "SingleBlock: Grav. (", sum_x, ", ", sum_y, ")"

d_sum_x = 0.0
d_sum_y = 0.0
for i in list:
    d_sum_x += (i[0]-sum_x)*(i[0]-sum_x)
    d_sum_y += (i[1]-sum_y)*(i[1]-sum_y)
print "SingleBlock: ", len(list), ":(", d_sum_x, ", ", d_sum_y, ")"

d_sum_x = 0.0
d_sum_y = 0.0
for i in list:
    d_sum_x += (i[0]-sum_x)*(i[0]-sum_x)
    d_sum_y += (i[1]-sum_y)*(i[1]-sum_y)
print "SingleBlock: ", len(list), ":(", d_sum_x, ", ", d_sum_y, ")"

