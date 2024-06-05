class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        
        if len(nums) == 1:
            return [nums]

        perms = self.permuteUnique(nums[1:])
        to_insert = nums[0]
        new_nums = nums[1:]
        l = []

        for perm in perms:
            for i in range(len(perm) + 1):
                item = []
                item.extend(perm[:i])
                item.append(to_insert)
                item.extend(perm[i:])
                l.append(item)
        
        l_no_dub = []
        for item in l:
            if item not in l_no_dub:
                l_no_dub.append(item)
        return l_no_dub
