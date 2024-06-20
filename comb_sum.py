class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []

        def helper(i, potential_comb):
            if sum(potential_comb) == target:
                res.append(potential_comb.copy())
                return res
            if sum(potential_comb) > target or i >= len(candidates):
                i = len(candidates)
                return res
            if sum(potential_comb) < target:
                potential_comb.append(candidates[i])
                helper(i, potential_comb)
                potential_comb.pop()
                helper(i + 1, potential_comb)
            return res
                
        return helper(0, [])
