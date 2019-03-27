/// author: kingil
/// theme: interview prepare leetcode solution
/// time: 2019-03

## 无重复字符的最长子串

//解法一 往死了循环
//注意n*n的遍历，从i-j中间取子串，然后用一个k去检测是不是有重复的。由于是一个个遍历，所以只需考虑每个串里新加进来的j是否和之前的每一个字符相同。
class Solution {
public:
    int lengthOfLongestSubstring(std::string s) {
        if(s.length()<=1)
        {
            return s.length();
        }
        int temp = 0;
        //abcabcbb
        for (int i=0;i< s.length();i++)
        {
            int count = 0;
            for(int j = i+1 ;j< s.length();j++)
            {
                bool flag = false;
                for(int k=j-1;k >= i;k--)
                {
                    if(s[k]==s[j])
                    {
                        flag = true;
                    }
                }
                if(!flag)
                {
                    count ++;
                }
                else
                {
                    break;
                }
            }
            if(count > temp)
            {
                temp = count;
            }
        }
        return temp+1;
    }
};

//解法二，用要给map记录每个子串出现的字母，用一个map，记录之前出现字母的位置下标。然后出现重复的时候，从该字母下标的下一位开始遍历。
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int maxLength = 0;
        map<char, int> m;
        int len = 0;
        for (int i = 0; i < s.size(); i ++) {
            if (m.count(s[i]) == 0) {
                len += 1;
                m[s[i]] = i;
            } else {
                len = 0;
                i = m[s[i]];
                m.clear();
            }
            if (maxLength < len) {
                    maxLength = len;
                }
        }
        return maxLength;
    }
};

## 判断s2是否包含s1，只含小写字母
关键利用substr，然后判断两个字符串排列是否一样，可以利用26个字母表的映射，创建两个int[26]，看每个字母出现的次数是否相同；
class Solution {
public:
     bool checkInclusion(std::string s1, std::string s2)
    {     
        bool flag = false;
        int length1 = s1.size();
        int length2 = s2.size();
        if(length2<length1)
        {
            return false;
        }
        
        for(int i =0;i<=(length2 - length1);i++)
        {
            if(WithSameElement(s1,s2.substr(i,length1) ) )
            {
                flag = true;
            }
        }
        return flag;
    }
    private:
    bool WithSameElement(std::string a,std::string b)
    {
        if(a.size()!=b.size())
        {
            return false;
        }
        int index1[26] = {0};
        int index2[26] = {0};
        for(int i =0;i<a.size();i++)
        {
            index1[int(a[i]-'a')]++;
            index2[int(b[i]-'a')]++;
        }
        for(int i =0;i< 26;i++)
        {
            if(index2[i]!= index1[i])
            {
                return false;
            }
        }
        return true;
    }
};

## 最长公共前缀
首先，排序。然后从最小的子串开始，判断是不是和排序后的第一位和最后一位字母相应的子串相同。
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        int n = strs.size();
        if(n==0)
        {
            return "";
        }
        if(n == 1)
        {
            return strs[0];
        }
        int k =0;
        int mark = 0;
        sort(strs.begin(),strs.end());
        int min =strs[0].size();
        for(int i =1;i<n;i++)
        {
            if(strs[i].size()< min)
            {
                min = strs[i].size();
                mark = i;
            }
        }
        
        for (int i = strs[mark].size();i>0;i--)
        {
            if(strs[mark].substr(0,i) == strs[0].substr(0,i) && strs[mark].substr(0,i) == strs[n-1].substr(0,i))
            {
                k = i;
                break;
            }
        }
        if(k!=0)
        {
            return strs[mark].substr(0,k);
        }
        else
        {
            return "";
        }

    }
};

## 字符串相乘
/**
        num1的第i位(高位从0开始)和num2的第j位相乘的结果在乘积中的位置是[i+j, i+j+1]
        例: 123 * 45,  123的第1位 2 和45的第0位 4 乘积 08 存放在结果的第[1, 2]位中
          index:    0 1 2 3 4  
              
                        1 2 3
                    *     4 5
                    ---------
                          1 5
                        1 0
                      0 5
                    ---------
                      0 6 1 5
                        1 2
                      0 8
                    0 4
                    ---------
                    0 5 5 3 5
        这样我们就可以单独都对每一位进行相乘计算把结果存入相应的index中        
        **/
class Solution {
public:
	string multiply(string num1, string num2) {
		int length1 = num1.size();
		int length2 = num2.size();
        if(length1 == 0 || length2==0)
            return "";
		int * index = new int[length1 + length2];
		for (int i = 0;i<length1 + length2;i++)
		{
			index[i] = 0;
		}
		for (int i = length1 - 1; i >= 0; i--)
		{
			for (int j = length2 - 1; j >= 0; j--)
			{
				int result = 0;
				result = (num1[i] - '0') * (num2[j] - '0');
				result += index[i + j + 1];
				index[i + j] += result / 10;    //add bit
				index[i + j + 1] = result % 10;  //result bit
			}
		}
		//producte the result
		int k = 0;
        bool flag = false;
		for (int i = 0; i< length1 + length2; i++)
		{
			if (index[i] != 0)
			{
                flag = true;
				k = i;
				break;
			}
		}
        if(!flag)
        {
            k = length1 + length2 -1;
        }
		string  s = "";
		for (int i = k; i<length1 + length2; i++)
		{
			stringstream temp;
			string str;
			temp << index[i];
			temp >> str;
			s.append(str);
		}
		return s;
	}
};




