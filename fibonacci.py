class Solution:
    def fib(self, n: int) -> int:
        if n < 2:
            return n
        return self.fib(n - 1) + self.fib(n - 2)


class Solution:
    def fib(self, n: int) -> int:
        i0 = 0
        i1 = 1
        i2 = 1
        if n == 0:
            return i0

        for i in range(3, n + 1):
            tmp = i2
            i2 = i1 + i2
            i1 = tmp
        return i2

