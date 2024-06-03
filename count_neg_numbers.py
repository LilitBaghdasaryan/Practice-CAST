class Solution:
    def countNegatives(self, grid: List[List[int]]) -> int:
        border_x = 0
        border_y = 0
        x = len(grid) - 1
        y = len(grid[0]) - 1
        count = 0
        while x >= border_x:
            while y >= border_y:
                if grid[x][y] < 0:
                    count += 1
                else:
                    border_y = y
                y -= 1
            x -= 1
            y = len(grid[0]) - 1
        return count
