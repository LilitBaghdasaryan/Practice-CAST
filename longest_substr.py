class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:

        if len(s) == 0:
            return 0

        LS = []
        LS.append(s[0])
        rep_index = 0

        for i in range(1, len(s)):
            if s[i] in LS[i - 1]:
                rep_index = LS[i - 1].index(s[i])
                if s[i] == s[i - 1]:
                    LS.append(s[i])
                else:
                    LS.append(LS[i - 1][rep_index + 1:] + s[i])
            else:
                LS.append(LS[i - 1] + s[i])
            print(i, LS[i])
        LS.sort(key = len)
        return len(LS[-1])
