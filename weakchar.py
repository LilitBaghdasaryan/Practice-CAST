class Solution:
    def numberOfWeakCharacters(self, properties: List[List[int]]) -> int:
        properties.sort(key=lambda x: x[1])
        properties.sort(key=lambda x: x[0])
        print(properties)
        count = 0
        max_ad = [properties[-1][0], properties[-1][1]]
        tmp0, tmp1 = 0,0
        changed = True

        for i in range(len(properties) - 1, 0, -1):
            if max_ad[1] < properties[i][1] and changed:
                tmp1 = properties[i][1]
                tmp0 = properties[i][0]
                changed = False
            
            
            if (tmp0 > properties[i - 1][0]):
                max_ad[0] = tmp0
                max_ad[1] = tmp1
                changed = True

            if (max_ad[1] > properties[i - 1][1]) and (max_ad[0] != properties[i - 1][0]):
                count +=1 

            print(max_ad, properties[i - 1])
            print()
            
        return count
