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

## 字符串的排列
看s2是否包含s1的排列。关键利用substr，然后判断两个字符串排列是否一样，可以利用26个字母表的映射，创建两个int[26]，看每个字母出现的次数是否相同；
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

## 翻转字符串
//解法一 两个栈
class Solution
{
public:
    std::string reverseWords(std::string s)
    {        
        std::stack<char> stk1;
        std::stack<char> stk2;
        std::string s2;
        //首先将字符串全部退入栈1
        for(int i = 0;i<s.size();i++)
        {
            stk1.push(s[i]);
        }
        //借助另外一个栈，将栈1的推入栈2，如果遇到空的，就把2清空，将结果放入目标字符串
        do
        {
            char temp =stk1.top();
            //std::cout<<temp;
            if( temp != ' ')
            {
                stk2.push(temp);
            }
            else
            {
                if(!stk2.empty())
                {
                    do
                {
                    s2.append(1,stk2.top());
                    stk2.pop();
                }
                while(!stk2.empty());
                
                s2.append(1,' ');
                }                
            }
            stk1.pop();
        }
        while(!stk1.empty());
        //最后剩一个单词
        do
        {
            s2.append(1,stk2.top());
            stk2.pop();
        }
        while(!stk2.empty());
        return s2;
    }
};

## 三数之和
class Solution {
public:
	vector<vector<int>> threeSum(vector<int>& nums) {
		vector<vector<int>> res;
		sort(nums.begin(),nums.end());
        if (nums.empty() || nums.back() < 0 || nums.front() > 0) return {};
		for (int i = 0; i< nums.size(); ++i){
			if (nums[i]>0) break;
			if(i>0 && nums[i] == nums[i-1] ) continue;
			int target = 0 - nums[i];
			int j = i + 1;
			int k = nums.size()-1;
			while (j<k){
				if (nums[j] + nums[k] == target){
					res.push_back({ nums[i],nums[j],nums[k] });
					while (j<k && nums[j] == nums[j + 1]) ++j;
					while (j<k && nums[k] == nums[k - 1]) --k;
					++j;--k;
				}else if (nums[j] + nums[k] < target)	++j;
				else  --k;
			}
		}
		return res;
	}
};

## 岛屿最大面积
递归解决，已经便利过的给个标记，如果面积是1，递归四个方向
class Solution {
public:
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        int area = 0,maxarea = 0;
        for (int i =0;i<grid.size();i++)
        {
            for(int j =0;j<grid[0].size();j++)
            {
                if(grid[i][j] ==1)
                {
                    area = dfs(grid,i,j);
                    if(area > maxarea) maxarea = area;
                }
            }
        }
        return maxarea;
    }
    int dfs(vector<vector<int>>& grid,int x,int y)
    {
        if( x<0 || y<0 || x >= grid.size() || y >= grid[0].size() || grid[x][y]!=1 )
            return 0;
        else
        {
            grid[x][y] = 2;
            return dfs(grid,x-1,y) + dfs(grid,x,y-1) + dfs(grid,x+1,y) + dfs(grid,x,y+1) +1;
        }
    }
};

## 搜索旋转数列中的某个值
二分查找，一半是有序的
class Solution {
public:
    int search(vector<int>& nums, int target) {
        return mysearch(nums, 0, nums.size() - 1, target);
    }
    private:
        int mysearch(vector<int>& nums, int low, int high, int target) {
        if (low > high)
            return -1;
        int mid = (low + high) / 2;
        if (nums[mid] == target)
            return mid;
        if (nums[mid] < nums[high]) {
            if (nums[mid] < target && target <= nums[high])
                return mysearch(nums, mid + 1, high, target);
            else
                return mysearch(nums, low, mid - 1, target);
        } else {
            if (nums[low] <= target && target < nums[mid])
                return mysearch(nums, low, mid - 1, target);
            else
                return mysearch(nums, mid + 1, high, target);
        }
    }
};

##三角形最小路径和
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        //1 直接从上往下取最小，最后出现问题
        //2 这个问题应该是从下往上的贪心问题？好像也不对[-1][1,3][1,-2,-3]
        //3 这是一个典型的动态规划问题啊，中间节点记录到该点的最短路径：A 从上往下，需要n*n的空间/原地解决 B 从下往上，借助n的空间
        int length =triangle.size();
        if(length== 0)
            return 0;
        vector<int> mark(length+1,0);
        for(int i = length-1;i>=0;i--)
        {
            for(int j = 0;j<=i;j++)
            {
                mark[j] = min(mark[j],mark[j+1]) + triangle[i][j];
            }
        }
        return mark[0];
    }
};
## 最大子序和，返回的是和
动态规划问题
当前最大的子序和 ＝ max(当前元素的值，前i－1个元素的最大子序和)
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        if(nums.size() == 0) return NULL;
        int res = INT_MIN;
        int f_n = -1;
        for(int i = 0; i < nums.size(); ++i){
            f_n = max(nums[i], f_n + nums[i]);
            res = max(f_n, res);
        }
        return res;
    }
};
## 合并区间问题
/**
 * Definition for an interval.
 * struct Interval {
 *     int start;
 *     int end;
 *     Interval() : start(0), end(0) {}
 *     Interval(int s, int e) : start(s), end(e) {}
 * };
 */
class Solution {
public:
    vector<Interval> merge(vector<Interval>& intervals) {
        //[[1,3],[2,6],[8,10],[15,18]]
        //首先根据首位进行排序
        //记录一个头和一个尾巴，注意如果下一个头小于上一个尾巴，要判断是不是尾巴的大小
        if(intervals.size()<1)
            return intervals;
        vector<Interval> result;
        Interval mark(0,0);
        for(int i =0;i<intervals.size();i++)
        {
            int max = intervals[i].start;
            int k = i;
            for(int j = i+1;j<intervals.size();j++)
            {
                if(intervals[j].start< max)
                {
                    max = intervals[j].start;
                    k = j;
                }
            }
            mark = {intervals[i].start,intervals[i].end};
            intervals[i] = intervals[k];
            intervals[k] = mark;
        }
        int start = intervals[0].start;
        int end = intervals[0].end;
        for(int i =0;i<intervals.size();i++)
        {
            if(intervals[i].start<= end)
                end = max(end,intervals[i].end);
            else
            {
                Interval temp(start,end);
                result.push_back(temp);
                start = intervals[i].start;
                end = intervals[i].end;
            }
        }
        
        Interval newtemp(start,end);
        result.push_back(newtemp);
        return result;
    }
};
##接雨水问题
class Solution {
public:
    int trap(vector<int>& height) {
        //找每一个柱子左右两边的最高点，柱子上能盛的水就是最低点min（left，right）－height
        int length = height.size();
        vector<int> left(length+2,0);
        vector<int> right(length+2,0);
        for(int i =1;i<length+1;i++)
        {
            left[i] = max(left[i-1],height[i-1]);
            right[length+1-i] = max(right[length +2-i],height[length-i]);
        }
        int sum = 0;
        for(int i =1;i<length +1;i++)
        {
            sum = sum + min(left[i],right[i]) - height[i-1];
        }
        return sum;
    }
};
## 平方根
class Solution {
public:
    int mySqrt(int x )
    {
        long y = x;
        long result = 0;
        if(y < 2 ) return y;
        for(long i = 0;i <= y;i++)
        {
            if(i*i <=y) continue;
            else
            {
                result = i-1;
                break;
            }
        }
        return result;
    }
};

## 朋友圈
事件复杂度还是在n*n，借助一个辅助向量做遍历标记。
class Solution {
public:
    int findCircleNum(vector<vector<int>>& M) {
        //DFS
        vector<int> mark(M.size(),0);
        int count = 0;
        for(int i = 0;i< M.size();i++)
        {
            if(mark[i] == 0)
            {
                count++;
                dfs(M,i,mark);
            }
        }
        return count;
    }
    void dfs(vector<vector<int>>& M,int i,vector<int>& mark)
    {
        mark[i] = 1;
        for(int j = 0;j<M.size();j++)
        {
            if(M[i][j] ==1 && mark[j] == 0) dfs(M,j,mark);
        }
    }
};

## 最长连续序列，未排序
先排序，然后直接算。
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        if(nums.size()<=1)
        {
            return nums.size();
        }
        sort(nums.begin(),nums.end());
        int length = 0,maxlength = 0;
        //0,1,1,2
        for(int i = 0;i<nums.size()-1;i++)
        {
            if( nums[i]+1 == nums[i+1]) 
                length ++;
            else if(nums[i] == nums[i+1])
                continue;
            else
            {              
                length = 0;
            }
            if(length > maxlength)
                    maxlength = length;
        }
        return maxlength+1;
    }
};

## 两链表数字相加
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        int carry = 0;
        ListNode* dummy = new ListNode(-1);
        ListNode* p = dummy;
        while (l1 || l2)
        {
            int val1 = l1 ? l1->val : 0;
            int val2 = l2 ? l2->val : 0;
            int val = val1 + val2 + carry;
            carry = val / 10;
            val %= 10;
            p->next = new ListNode(val);
            if (l1)
                l1 = l1->next;
            if (l2)
                l2 = l2->next;
            p = p->next;
        }
        if (carry)
            p->next = new ListNode(carry);
        return dummy->next;
    }
};

## 翻转链表
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        //非递归算法
        if(head == NULL || head->next == NULL)
            return head;
        ListNode * pre = head;
        ListNode * p  = head->next;
        ListNode * next = NULL;
        while(p != NULL)
        {
            next  = p->next;
            p->next = pre;
            pre = p;
            p = next;
        }
        head->next = NULL;
        return pre;
        // //递归算法
        // if(head == NULL || head->next == NULL)
        //     return head;
        // ListNode Node= reverse(head);
        // head->next = NULL;
        // return Node;
    }
    ListNode * reverse(ListNode * p)
    {
        if(p->next == NULL) return p;
        else
        {
            ListNode * next = p->next;
            ListNode * tail = reverse(next);
            next->next = p;
            return tail;
        }
    }
    
    

};

## 最小的栈

class MinStack {
    int  * items = NULL;
    int count;
    int * minitems = NULL;
    int mincount;
    int size = 1000;
    /** initialize your data structure here. */
    public:
    MinStack() {
        items = new int[size];
        minitems = new int[size];
        count = 0;
        mincount = 0;
    }
    
    void push(int x) {
        if(count == size || mincount == size) return;
        if(mincount == 0  || minitems[mincount -1 ] >= x)
        {
            minitems[mincount] = x;
            ++mincount;
        }
        items[count] = x;
        ++count;
    }
    
    void pop() {
        if(count == 0 || mincount == 0)
            return;
        if(items[count -1] == minitems[mincount -1])
            --mincount;
        --count;
    }
    
    int top() {
        if(count ==0 || mincount == 0) return -1;
        return items[count -1];
    }
    
    int getMin() {
        if(mincount == 0 ) return -1;
        return minitems[mincount -1];
    }
};


## 最大的正方形
动态规划问题，右下角要成为正方形的一个角，则左上方三个数字都必须是正方形的角。
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        if(matrix.size()==0)
        {
            return 0;
        }
       int ans  =0 ;
        vector<vector<int>> my;
        my.resize(matrix.size());
	    for (int i = 0; i < matrix.size(); ++i)
		    my[i].resize(matrix[0].size());
        
        for(int i=1;i<matrix.size();i++)
        {
        for(int j=1;j<matrix[0].size();j++)
        {
            if(my[i][j]==0)continue;
            my[i][j]=min(min(my[i-1][j],my[i][j-1]),my[i-1][j-1])+1;             //动态方程
        }
        }
        
    for(int i=1;i<matrix.size();i++)
    {
        for(int j=1;j<matrix[0].size();j++)
        {
            if(my[i][j]>ans)ans=my[i][j];                  //找到最大值
        }
    }
        return ans*ans;
    }
};

## 股票1 只买入和卖出一次

第i天卖出的最大收益 ＝ 
max(前i－1天的最大收益，p[i] - 前i－1天的最小价格)

class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int current = 0;
        int min = 0;
        for(int i =0;i<prices.size();i++)
        {
            min = findmin(prices,i);
            if(current<(prices[i] - min))
            current = prices[i] - min;
        } 
        return current;
    }
    int findmin(vector<int>& prices,int x)
    {
        int temp = prices[0];
        for(int i = 0;i< x;i++)
        {
            if(prices[i]<temp)
                temp = prices[i];
        }
        return temp;
    }
};
## 股票2 买入和卖出多次，当天可以买入再卖出
只要后一天比前一天大，就前一天买入，第二天卖出。最大收益就是这样每一天的集合。
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int current = 0;
        int min = 0;
        for(int i =0;i<prices.size();i++)
        {
            min = findmin(prices,i);
            if(current<(prices[i] - min))
            current = prices[i] - min;
        } 
        return current;
    }
    int findmin(vector<int>& prices,int x)
    {
        int temp = prices[0];
        for(int i = 0;i< x;i++)
        {
            if(prices[i]<temp)
                temp = prices[i];
        }
        return temp;
    }
};
## 合并两个有序链表
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode * result = new ListNode(0);
        ListNode * temp = result;
        while(l1 != NULL && l2!=NULL)
        {
            if(l1->val <= l2->val)
            {
                
                temp->next = l1;
                temp= temp->next;
                l1 = l1->next;
            }
            else
            {
                temp->next = l2;
                temp= temp->next;
                l2 = l2->next;
            }
        }
        if(l1 == NULL) {
            temp->next = l2;
        }else {
            temp->next = l1;
        }
        return result->next;
    }
};

## 未排序的数组中找到第 k 个最大的元素
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        sort(nums.begin(),nums.end());
        if(k==1)
            return nums[nums.size()-1];
        int temp = 1,i = nums.size()-1;
        while(i>=0)
        {
            if(i == (nums.size()-1)) i--;
            else
            {
                temp++;
                if(temp == k)
                    break;
                i--;
            }
        }
        return nums[i];
    }
};

## 最长连续递增的子序列
注意，必须是连续的，所以这个相对来说比较简单，事件复杂度
n遍历即可
class Solution {
public:
    int findLengthOfLCIS(vector<int>& nums) 
    {
        if(nums.size()<1)return 0;
        if(nums.size()==1)return 1;
        int maxlength = 0,length =0,i =0,j =0;
        while (i < nums.size())
        {
            if(i < (nums.size()-1))
            {
                if(nums[i]<nums[i+1])
                {
                    ++i;
                }
                else
                {       
                length = i -j;
                if(length > maxlength)
                    maxlength = length;
                j = i+1;
                ++i;
                }
            }
            else
                {       
                length = i -j;
                if(length > maxlength)
                    maxlength = length;
                j = i+1;
                ++i;
                }
        }
        return maxlength+1;
    }
};

## 俄罗斯套娃信封问题
注意最终转化为最长的递增序列，而不是子序列
class Solution {
public:
     static bool cmp(pair<int,int> p1,pair<int,int> p2){
        if(p1.first == p2.first) {
            return p1.second  > p2.second;
        }        
        else{
            return p1.first < p2.first;
        }
    }
    int maxEnvelopes(vector<pair<int, int>>& e) 
    {
        //第一种思路：n*n的时间复杂度(其实是nlogn)，以宽度进行排序，然后比较高度，最终给出可能的最大信封数量。
        //同样是先对宽度进行排序，从小到大，宽度相同的高度从大到小。
        //因为宽度已经有序了，所以接下来问题就转化成求最长的递增子序列的长度（不是必须连续）
         if(e.size() == 0) return 0;
        sort(e.begin(),e.end(),cmp);
        vector<int> dp(e.size(),1);
        int mx = 1;
        for(int i = 1; i < e.size(); i++)
        {
            for(int j = i - 1; j >= 0; j--)
                if(e[j].second < e[i].second && dp[j] + 1 > dp[i])
                {
                    dp[i] = dp[j] + 1;
                    mx = max(dp[i],mx);
                }
        }
        return mx;
        }
};

## 全o(1)的数据结构

Inc(key) - 插入一个新的值为 1 的 key。或者使一个存在的 key 增加一，保证 key 不为空字符串。
Dec(key) - 如果这个 key 的值是 1，那么把他从数据结构中移除掉。否者使一个存在的 key 值减一。如果这个 key 不存在，这个函数不做任何事情。key 保证不为空字符串。
GetMaxKey() - 返回 key 中值最大的任意一个。如果没有元素存在，返回一个空字符串""。
GetMinKey() - 返回 key 中值最小的任意一个。如果没有元素存在，返回一个空字符串""。

class AllOne {
public:
    /** Initialize your data structure here. */
    unordered_map<string, int> nummap;      /*保存所有key-value*/
    list<string> keylist;                   /*双向链表存储所有key 用于取最大最小值 front存最大 back存最小*/
    AllOne() {

    }
    
    /** Inserts a new key <Key> with value 1. Or increments an existing key by 1. */
    void inc(string key) {
        if(nummap.find(key) == nummap.end())
        {
            keylist.push_back(key);
            nummap[key] = 1;
        }
        else
        {
            nummap[key] += 1;
            if(nummap[key] >= nummap[keylist.front()])
            {
                keylist.remove(key);
                keylist.push_front(key);
            }
            else if(key == keylist.back())
            {
                keylist.pop_back();
                if(nummap[key] > nummap[keylist.back()])
                {
                    list<string>::iterator it = keylist.end();
                    --it;
                    keylist.insert(it, key);
                }
                else
                    keylist.push_back(key);
            }
        }  
    }
    
    /** Decrements an existing key by 1. If Key's value is 1, remove it from the data structure. */
    void dec(string key) {
        if(nummap.find(key) == nummap.end())
            return;
        if(nummap[key] == 1)
        {
            nummap.erase(key);
            keylist.remove(key);
        }
        else
        {
            nummap[key] -= 1;
            if(nummap[key] < nummap[keylist.back()])
            {
                keylist.remove(key);
                keylist.push_back(key);
            }
            else if(key == keylist.front())
            {
                keylist.pop_front();
                if(nummap[key] < nummap[keylist.front()])
                {
                    list<string>::iterator it = keylist.begin();
                    ++it;
                    keylist.insert(it, key);
                }
                else
                    keylist.push_front(key);
            }
        }
    }
    
    /** Returns one of the keys with maximal value. */
    string getMaxKey() {
        if(keylist.empty())
            return "";
        else 
            return keylist.front(); 
    }
    
    /** Returns one of the keys with Minimal value. */
    string getMinKey() {
        if(keylist.empty())
            return "";
        else 
            return keylist.back();
    }
};

/**
 * Your AllOne object will be instantiated and called as such:
 * AllOne obj = new AllOne();
 * obj.inc(key);
 * obj.dec(key);
 * string param_3 = obj.getMaxKey();
 * string param_4 = obj.getMinKey();
    */



## 三角形最小路径和

class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        //1 直接从上往下取最小，最后出现问题
        //2 这个问题应该是从下往上的贪心问题？好像也不对[-1][1,3][1,-2,-3]
        //3 这是一个典型的动态规划问题啊，中间节点记录到该点的最短路径：A 从上往下，需要n*n的空间/原地解决 B 从下往上，借助n的空间
        // vector <int> element(triangle.size(),0);
        // int j = 0;
        // for(int i = 0;i<triangle.size();i++)
        // {
        //     if(i == 0)
        //     {
        //         element[i] = triangle[0][0];
        //         j = 0;
        //     }
        //     else if(triangle[i][j] < triangle[i][j+1])
        //     {
        //         element[i] = triangle[i][j];
        //         j = j;
        //     }
        //     else
        //     {
        //         element[i] = triangle[i][j+1];
        //         j = j+1;
        //     }
        // }
        // int sum = 0;
        // for(int i = 0;i<element.size();i++)
        // {
        //     sum+=element[i];
        // }
        // return sum;
        int length =triangle.size();
        if(length== 0)
            return 0;
        vector<int> mark(length+1,0);
        for(int i = length-1;i>=0;i--)
        {
            for(int j = 0;j<=i;j++)
            {
                mark[j] = min(mark[j],mark[j+1]) + triangle[i][j];
            }
        }
        return mark[0];
    }
};

## 接雨水

class Solution {
public:
    int trap(vector<int>& height) {
        //找每一个柱子左右两边的最高点，柱子上能盛的水就是最低点min（left，right）－height
        int length = height.size();
        vector<int> left(length+2,0);
        vector<int> right(length+2,0);
        for(int i =1;i<length+1;i++)
        {
            left[i] = max(left[i-1],height[i-1]);
            right[length+1-i] = max(right[length +2-i],height[length-i]);
        }
        int sum = 0;
        for(int i =1;i<length +1;i++)
        {
            sum = sum + min(left[i],right[i]) - height[i-1];
        }
        return sum;
    }
};

## 第k个排列

class Solution {
public:
    string getPermutation(int n, int k) {
        //https://www.cnblogs.com/ariel-dreamland/p/9149577.html
        
        //牛逼，说白了就是找规律，一个数字一个数字分析
        string res;
        string num = "123456789";
        vector<int> f(n, 1);
        for (int i = 1; i < n; ++i) 
            f[i] = f[i - 1] * i;
        --k;
        for (int i = n; i >= 1; --i) {
            int j = k / f[i - 1];
            k %= f[i - 1];
            res.push_back(num[j]);
            num.erase(j, 1);
            //删除j位置的一位
        }
        return res;
    }
};

## 复原IP

给定一个只包含数字的字符串，复原它并返回所有可能的 IP 地址格式。
输入: "25525511135"
输出: ["255.255.11.135", "255.255.111.35"]

//思路：每次从字符串前分别取 1,2,3 个字符，如果符合要求，就加入当前的 ip_parts 中，再在余下的字符中，找 ip 的下一部分。当发现 ip_parts 长度为 4，且没有剩余的字符时，将其加入到结果中。如果长度已经为 4，但余下字符尚不为空，说明 ip_parts 中的解不对，抛弃之。

//

class Solution {
public:
    vector<string> restoreIpAddresses(string s) {
        vector<string> result;
        restore(s,4,"",result);
        return result;
    }
    void restore(string s,int leftset,string out,vector<string>& res)
    {
        if(leftset == 0)
        {
            if(s.empty()) res.push_back(out);
        }
        //["255.255.11.135", "255.255.111.35"]
        else
        {
            for(int i = 1;i<= 3;++i)
            {
                if(s.size() >= i && isValid(s.substr(0,i)))
                {
                    if(leftset == 1) 
                        restore(s.substr(i),leftset -1,out+s.substr(0,i),res );
                    else
                        restore(s.substr(i),leftset-1,out+s.substr(0,i)+".",res);
                }
                
            }
        }
    }
    bool isValid(string s)
    {
        if (s.empty() || s.size() > 3 || (s.size() > 1 && s[0] == '0')) return false;
        int res = atoi(s.c_str());
        return res <= 255 && res >= 0;
    }
};



## 简化路径

<https://leetcode-cn.com/problems/simplify-path/>

以 Unix 风格给出一个文件的**绝对路径**，你需要简化它。或者换句话说，将其转换为规范路径。

在 Unix 风格的文件系统中，一个点（`.`）表示当前目录本身；此外，两个点 （`..`） 表示将目录切换到上一级（指向父目录）；两者都可以是复杂相对路径的组成部分。更多信息请参阅：[Linux / Unix中的绝对路径 vs 相对路径](https://blog.csdn.net/u011327334/article/details/50355600)

请注意，返回的规范路径必须始终以斜杠 `/` 开头，并且两个目录名之间必须只有一个斜杠 `/`。最后一个目录名（如果存在）**不能**以 `/` 结尾。此外，规范路径必须是表示绝对路径的**最短**字符串。

思路：

思路简单，但是比较暴力

class Solution {
public:
    string simplifyPath(string path) {
        stack<string> sta;
        vector<string> vec = split(path);
        for(int i=0;i<vec.size();i++){
            if(vec[i] == ".."){
                if(sta.size() != 0) sta.pop();
            }else if(vec[i] == "."){
                continue;
            }else{
                sta.push(vec[i]);
            }
        }
        string s = "";
        while(sta.size() != 0){
            s = "/" + sta.top() + s;
            sta.pop();
        }
        if(s == "") s = "/";
        return s;
        

    }
    
    vector<string> split(string path){
        vector<string> vec;
        string s = "";
        for(int i=0;i<path.length();i++){
            if(path[i] == '/'){
                if(s != "") vec.push_back(s);
                s = "";
            }else{
                s += path[i];
            }
        }
        if(s != "") vec.push_back(s);
        return vec;
    }
};



## 最小栈

设计一个支持 push，pop，top 操作，并能在常数时间内检索到最小元素的栈。

- push(x) -- 将元素 x 推入栈中。
- pop() -- 删除栈顶的元素。
- top() -- 获取栈顶元素。
- getMin() -- 检索栈中的最小元素。

class MinStack {
    int  * items = NULL;
    int count;
    int * minitems = NULL;
    int mincount;
    int size = 1000;
    /** initialize your data structure here. */
    public:
    MinStack() {
        items = new int[size];
        minitems = new int[size];
        count = 0;
        mincount = 0;
    }
    
    void push(int x) {
        if(count == size || mincount == size) return;
        if(mincount == 0  || minitems[mincount -1 ] >= x)
        {
            minitems[mincount] = x;
            ++mincount;
        }
        items[count] = x;
        ++count;
    }
    
    void pop() {
        if(count == 0 || mincount == 0)
            return;
        if(items[count -1] == minitems[mincount -1])
            --mincount;
        --count;
    }
    
    int top() {
        if(count ==0 || mincount == 0) return -1;
        return items[count -1];
    }
    
    int getMin() {
        if(mincount == 0 ) return -1;
        return minitems[mincount -1];
    }
};





## 

##



## 



## 



## 

##



## 



## 



## 




