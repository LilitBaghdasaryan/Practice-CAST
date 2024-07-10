class XORNetwork():
    def forward(self, x1, x2):
        tmp = x1 + x2
        x2 = -1 * x1 + -1 * x2
        x1 = tmp
        
        x1 = 1 if x1 >= 1 else 0
        x2 = 1 if x2 >= -1 else 0
        
        res = x1 + x2
        res = 1 if res >= 2 else 0
        return res
        
a = XORNetwork().forward(0, 1)
# print(a)


class BooleanNetwork():
    def __init__(self, W, T):
        self.W = W
        self.T = T
    
    def forward(self, x1, x2):
        tmp = self.W[0][0] * x1 + self.W[0][1] * x2
        x2 = self.W[1][0] * x1 + self.W[1][1] * x2
        x1 = tmp

        x1 = 1 if x1 >= self.T[0][0] else 0
        x2 = 1 if x2 >= self.T[0][1] else 0
        
        res = self.W[2][0] * x1 + self.W[2][1] * x2
        res = 1 if res >= self.T[1][0] else 0
        return res


# XOR
print('XOR')
print(BooleanNetwork([[1, 1], [-1, -1], [1, 1]], [[1, -1], [2]]).forward(0, 0))
print(BooleanNetwork([[1, 1], [-1, -1], [1, 1]], [[1, -1], [2]]).forward(0, 1))
print(BooleanNetwork([[1, 1], [-1, -1], [1, 1]], [[1, -1], [2]]).forward(1, 0))
print(BooleanNetwork([[1, 1], [-1, -1], [1, 1]], [[1, -1], [2]]).forward(1, 1), end='\n\n')

# OR
print('OR')
print(BooleanNetwork([[1, 1], [-1, -1], [1, 1]], [[1, 1], [1]]).forward(0, 0))
print(BooleanNetwork([[1, 1], [-1, -1], [1, 1]], [[1, 1], [1]]).forward(0, 1))
print(BooleanNetwork([[1, 1], [-1, -1], [1, 1]], [[1, 1], [1]]).forward(1, 0))
print(BooleanNetwork([[1, 1], [-1, -1], [1, 1]], [[1, 1], [1]]).forward(1, 1), end='\n\n')

# NOR
print('NOR')
print(BooleanNetwork([[1, 1], [-1, -1], [1, 1]], [[0, 0], [2]]).forward(0, 0))
print(BooleanNetwork([[1, 1], [-1, -1], [1, 1]], [[0, 0], [2]]).forward(0, 1))
print(BooleanNetwork([[1, 1], [-1, -1], [1, 1]], [[0, 0], [2]]).forward(1, 0))
print(BooleanNetwork([[1, 1], [-1, -1], [1, 1]], [[0, 0], [2]]).forward(1, 1), end='\n\n')
