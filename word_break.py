class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:

        DP = [False for _ in range(len(s))]   
        DP.append(True)
        i = len(s) - 1
        last_len = 0

        while i >= 0:
            for word in wordDict:
                print(word, s[i:i + len(word)])
                if s[i:i + len(word)] == word:
                    DP[i] = DP[i + len(word)]
                    last_len = len(word)
                    if DP[i]:
                        break
            i -= 1
            
        if DP[0]:
            return DP[last_len]
        return False
