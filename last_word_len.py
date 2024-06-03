class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        tokens = s.split()
        return len(tokens[-1])
