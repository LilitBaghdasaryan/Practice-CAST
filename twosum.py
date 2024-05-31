class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        i = 0
        j = len(nums) - 1
        while i < len(nums):
            while j > i:
                if nums[i] + nums[j] == target:
                    return i, j
                else:
                    j -= 1
            i += 1
            j = len(nums) - 1
        return
