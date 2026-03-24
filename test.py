import math

# print(math.sqrt(0.322**2 - 0.26742**2))
# print(math.sqrt(0.10138**2 + 0.15115**2))

theta = math.acos(0.098/0.118)
x = 0.064 * math.cos(theta)
y = 0.064 * math.sin(theta)
print([x, y])