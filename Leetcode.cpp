 /*

*/


///Leetcode solutions
///线性表


//2.1.1 remove duplicate from sortd array
class solution
{
	int removedumplicate(vector<int>& nums)
	{
		if(nums.empty()) return 0;
		int index = 0;
		for(int i =1;i < nums.size();i++)
		{
			if(nums[index] != nums[i])
				index = index + 1;
				nums[index] = nums[i];
		}
	}
}


//2.1.3 search in sorted array
//二分查找

public int search(const vector<int>& nums,int target)
{
	int first = 0;
	int last = nums.size;
	while(first != last)
	{
		int mid = first + (last - first)/2 ;
		if(nums[mid] == target)
			return mid;
		if(nums[first] < nums[mid])
		{
			if(nunms[first]<target && target <nums[mid])
				last = mid;
			else
				first = mid + 1;
		}
		else
		{
			if(nums[mid]< target && target < nums[last])
				first = mid + 1;
			else
				last = mid;
		}
	}
	
	return -1;
}

2.1.5 median of two arrays
//两个排好序的数列，找到所有元素第k个大的元素

//第一个思路，记录当前最大的元素为第m大的元素
//两个指针，分别指向最开始的元素，比较当前元素的大小，若大小不一样，相应小的元素列往前移，并且累计的第m的值+1
//但是还是O(m+n)的复杂度

// 时间复杂度O(log(m+n)) 空间复杂度 O(log(m+n))
class Solution 
{
	//
public: double findMedianSortedArrays(const vector<int>& A, const vector<int>& B) 
	{
	const int m = A.size();
	const int n = B.size();
	int total = m + n;
	//首先，考虑m+n的奇偶性，然后开始执行递归
	if (total & 0x1)
	return find_kth(A.begin(), m, B.begin(), n, total / 2 + 1);
	else
	return (find_kth(A.begin(), m, B.begin(), n, total / 2)
	+ find_kth(A.begin(), m, B.begin(), n, total / 2 + 1)) / 2.0;
	}
	
private: static int find_kth(std::vector<int>::const_iterator A, int m, B, int n, int k) 
	{
		//递归终止条件，一个为空，直接返回；第A[k/2] == B[k/2] 找到我们要查找的数字
		
		//a假设m总是小于等于n的，如果不是，将A 和B 的次序颠倒进行二分查找即可
		if (m > n) return find_kth(B, n, A, m, k);
		//个数小的一个首先为空，即返回当前查找两个数组的B数组的当前的第K个即可
		if (m == 0) return *(B + k - 1);
		//特殊情况，如果只找第一个，也就是最小的元素，直接返回
		if (k == 1) return min(*A, *B);
		//现在比较A[k/2] B[k/2] 的大小，注意这里有个前提是m > k/2,n>k/2,现在m<=n,所以有可能m<k/2
		//如果m>k/2,k/2都落在AB数组范围内，正常进行划分查找就可以
		//如果m<k/2,直接取A数组前m个数字，也就是A下一次成空。B取以k-m为二分查找的分界线
		//divide k into two parts
		int ia = min(k / 2, m), ib = k - ia;
		if (*(A + ia - 1) < *(B + ib - 1))
		return find_kth(A + ia, m - ia, B, n, k - ia);
		else if (*(A + ia - 1) > *(B + ib - 1))
		return find_kth(A, m, B + ib, n - ib, k - ib);
		else
		return A[ia - 1];
	}
}

2.1.6 无序表里找连续序列最大的长度，时间空间均为n

//解法一 先排序，再找，复杂度为nlogn

//解法二 哈希表
class Solution 

	public:
	int longestConsecutive(const vector<int> &nums) 
	{
		unordered_map<int, bool> used;
		//初始化表，标注每一个元素为false
		for (auto i : nums) used[i] = false;
			int longest = 0;
		for (auto i : nums) 
		{
			if (used[i]) continue;
			int length = 1;
			used[i] = true;
			//顺藤摸瓜，一个方向上的连续数字都做标记
			for (int j = i + 1; used.find(j) != used.end(); ++j) //判断该元素j是否存在于该map中
			{
				used[j] = true;
				++length;
			}
			for (int j = i - 1; used.find(j) != used.end(); --j) 
			{
				used[j] = true;
				++length;
			}
			longest = max(longest, length);
		}
		return longest;
	}
}

//解法三

2.1.7 给定一个数组，一个数字，找出两个数字加起来等于该数字，返回大小下下标--2SUM

//解法一 暴力搜索，o(n*n)
//解法二 hash 存储下每个数对应的下标
vector<int> twoSum(vector<int>&nums,int target)
{
	unordered_map<int,int> mapping;
	vector<int> result;
	for (int i = 0;i<nums.size();i++)
	{
		mappimg[nums[i]] = i;//存储的是数组里每个元素的下标
	}
	for(int i =0;i<nums.size();i++)
	{
		int gap = target - nums[i];
		if(mapping.find(gap)!= mapping.end() && mapping[gap]>i)//避免重复查找
		{
			result.push_back(i+1);
			result.push_back(mapping[gap]+1);
			break;			
		}
		return result;
	}
}

//解法三 排序 然后头指针、尾指针从前到后夹逼
//2 sum
int i = starting; //头指针
int j = num.size() - 1; //尾指针
while(i < j) {
    int sum = num[i] + num[j];
    if(sum == target) {
        store num[i] and num[j] somewhere;
        if(we need only one such pair of numbers)
            break;
        otherwise
            do ++i, --j;
    }
    else if(sum < target)
        ++i;
    else
        --j;
}


2.1.8 3SUM

//解法 先排序，再左右夹逼，跳过重复的数字，时间O(n*n) 空间O(1) 简化成2SUM问题

2.1.11 给出序列，删除指定value

//解法一 直接便利 时间n
//解法二 利用vector
public : removeElement(vector<int> &nums,int target)
{
	return distance(nums.begin,remove(nums.begin(),nums.end(),target))
}


2.1.12 下一个排列算法

//解法一 不知道什么解法
//https://www.jianshu.com/p/0fb544271bb5

//解法二 
//1 从右到左，找到第一个破坏升序排列的数字，记为 PartitionNumber
//从右到左，几下第一个比这个PartitionNumber大的数字，记为ChangeNumber
//交换这两个Number
//PartitionIndex右边的所有序列，逆序操作
/*
6--8--7--4--3--2(6 - partionNumber,7-changeNumber,0-partionIndex)
7--8--6--4--3--2(swap)
7--2--3--4--6--8(reverse)
*/ 

2.1.13 继承2.1.12 找到第K个序列

//解法一 暴力使用 下一个排列算法
//解法二 康拓编码???

2.1.14 判断数独是否有效

//行、列、九宫格都要检查，思路是设置一个bool[9]的数组，分别检测和即可，有重复数字返回false

2.1.15 雨量计算器

//解法一 时间n 空间n
//对于每一个柱子，分别找左边和右边的最高的柱子
//比如柱子数组是a[length]
//记录左右最大柱子的数组为b[length*2]

for (i = 0;i<length;i++)
{
	if(i = 0)
	{
		b[2*i] = 0;
	}
	if(i = length -1)
	{
		b[2*i + 1 ] = 0;
	}
	b[left] = b [2*i] =  a[i-1];
	b[right] = b[2*2 +1]  = a[i+1];
	//left search
	for(j = i-1;j>=0;j--)
	{
		if (a [j] > left)
			b [2*i] = a[j];	
	}
	//...
}

//解法二 利用最大值，分两半，时间n，空间1
int trap(const vector<int>& A) 
{
	const int n = A.size();
	int max = 0; 
	for (int i = 0; i < n; i++)	
		if(A[i] > A[max])
			max = i;
	int water = 0;
	for (int i = 0, peak = 0; i < max; i++)
	{
		if (A[i] > peak) 
			peak = A[i];
		else 
			water += peak - A[i];
	}
		
	for (int i = n - 1, top = 0; i > max; i--)
	{
		if (A[i] > top)
			top = A[i];
		else 
			water += top - A[i];
	}	
	return water;
}


2.1.16 图像反转-顺时针90度

//思路一 对角线一次，水平翻转一次
//时间n*n 空间1


2.1.17 给定一个用数组表示每一位的数字，进行加以操作

//思考 正常直接给最后一位加一就行，但是需要考虑进位问题，所以算法如下
public :vector<int>plusone (vector<int> &digits)
{
	int n = digits.size;
	//确保遇到9的下一位一定进位
	for (i = n-1 ;i>=0;i--)
	{
		if(digits[i] == 9) digits[i] = 0;
		else
		{
			digits[i] += 1;
			return digits;
		}
	}
	//如果是0，直接返回1
	if(digits.front == 0)
		digits.insert(digits.begin(),1);
	return digits;
}

2.1.18 爬楼梯，n级台阶，一次1-2，多少种爬法

//解法一 递归 太慢 动态规划
int lift(int n)
{
    int f;
    if(n==1||n==0||n==2)
	{
         return n;
    }
    else
	{
        return lift(n-1)+lift(n-2);
    }
}
//解法二 迭代 每一级台阶方法等于前两级方法的和，所以一直替换pre和cur

int climbStairs(int n) 
{
	int prev = 0;
	int cur = 1;
	for(int i = 1; i <= n ; ++i)
	{
		int tmp = cur;
		cur += prev;
		prev = tmp;
	}
	return cur;
}

//解法三 数学公式，斐波那契数列的通项公式

2.1.19 格雷码 gray code

2.1.20 设置矩阵为零
//m*n的矩阵，检测到为零元素后，将他的行列都设置成零，要求空间复杂度为1

//解法一 空间mn，每一位都标记，太蠢了

//解法二 空间m+n，两个bool数组，记录每行每列是否存在0，然后对应行列置零即可

//解法三 解法对空间有要求，所以考虑复用原来行列的空间，复用原矩阵的第一行和第一列。
//但是第一行和第一列单独放两个标志来解决
void setzeroes(int[][]matrix)
{
	if(matrix == null || matrix.length == 0|| matrix[0].length ==0)
		return;
	bool row = false;
	bool col = false;
	for(int i = 0;i<m;i++)
	{
		if(matrix[i][0] == 0)
		{
			col = true;
			break;
		}
	}
	for(int i = 0;i<m;i++)
	{
		if(matrix[0][i] == 0)
		{
			row = true;
			break;
		}
	}
	
	for(i =1;i<matrix.length;i++)
	{
		for(j = 1;j<matrix[0].length;i++)
		{
			if(matrix[i][j]==0)
			{
				matrix[i][0] = 0;
				matrix[0][j] = 0;
			}
		}
	}
	for(i =1;i<matrix.length;i++)
	{
		for(j = 1;j<matrix[0].length;i++)
		{
			if(matrix[i][0]==0 || matrix[0][j]==0)
			{
				matrix[i][j] = 0;
			}
		}
	}
	
	if(col)
	{
		//第一列置零
	}
	if(col)
	{
		//第一行置零
	}	
}


//2.1.21 gas station

/*
已知gas[i] 和cost[i]
解法一 把每一个点当作开头，一个一个试，中间走不下去了，这个点就不能当作起点，pass 
解法二 关键是分析这道题的特征，如果总的油量小于总的里程数，肯定无解。
其次，对于每一个当前的点，如果出现到达下一个点油量不够，则这个点无论如何都不能当作起点。如果可以，继续往下走。
*/

public:	int canCompleteCircuit(vector<int> &gas, vector<int> &cost) 	
{		
	int sum = 0;		
	int total = 0;		
	int j = -1;		
	for(int i = 0; i < gas.size() ; ++i)		
	{			
		sum += gas[i]- cost[i];			
		total += gas[i]- cost[i];			
		if(sum < 0)			
		{				
			j=i; 
			sum = 0; 			
		}		
	}		
	if(total<0) 
		return -1;		
	else 
		return j+1;	
}

//2.1.22 糖果解法-贪心算法

/*
初始化所有小孩糖数目为1
从前往后扫描，如果第i个小孩等级比第i-1个高，那么i的糖数目等于i-1的糖数目+1；
从后往前扫描，如果第i个的小孩的等级比i+1个小孩高,但是糖的数目却小或者相等，那么i的糖数目等于i+1的糖数目+1。
该算法时间复杂度为O(N)。
之所以两次扫描，即可达成要求，是因为：
第一遍，保证了每一点比他左边candy更多(如果得分更高的话)。
第二遍，保证每一点比他右边candy更多(如果得分更高的话)，同时也会保证比他左边的candy更多，因为当前位置的candy只增不减。
*/

public class Solution {
    public int candy(int[] rating) {
        int len=rating.length;
        int [] candy=new int[len];
        for(int i=0;i<len;i++){
            candy[i]=1;
        }
        for(int i=1;i<len;i++){
            if(rating[i]>rating[i-1])
                candy[i]=candy[i-1]+1;
        }
        for(int i=len-2;i>=0;i--){
            if((rating[i]>rating[i+1])&&(candy[i]<=candy[i+1]))
                candy[i]=candy[i+1]+1;
        }
        int num=0;
        for(int i=0;i<len;i++){
            num+=candy[i];
        }
        return num;
    }
}

//2.1.23 single number I
//给定数字数组，找到数组中唯一一个出现一次的元素，线性时间复杂，不多用空间
//所有数字异或运算，最后剩下的数字就是那个


//2.1.24 single number II
/*给定数组，只有一个是一个，其他都是三个

解法一
思路其实跟上面一题一样，如果你能想到数字在计算机中的表示规律，也就是位运算，那就基本比较好处理了；
统计每一位上所有数字中1(0)出现的次数，如果对3求余是1的话，就表示唯一的那个数字对应位置也是1；
所以算法如下：
*/

class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int res = 0;
        for (int i = 0; i < 32; ++i) {
            int sum = 0;
            for (int j = 0; j < nums.size(); ++j) {
                sum += (nums[j] >> i) & 1;
            }
            res |= (sum % 3) << i;
        }
        return res;
    }
};


/*解法二
用二进制去模拟三进制的运算，也就是说用二进制去模拟single number中的解题思路，
使最终剩下的数字为想要的单个数字
*/
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int one = 0, two = 0, three = 0;
        for (int i = 0; i < nums.size(); ++i) {
            two |= one & nums[i];
            one ^= nums[i];
            three = one & two;
            one &= ~three;
            two &= ~three;
        }
        return one;
    }
}；

/*解法三
更加巧妙，但是目前还没看懂
大致的思路是，一个数字一个数字进行*运算，如果组里没有该元素A，则A*0 = A，进入ones；
如果组里有两个元素，则A*A = 0；然后进入twos组里；
如果出现三次，则ones和twos组里都不存在这个数字；
最后，出现三次的数字都不在了，出现一次的就是ones 的数字


还有一种说法是
根据上面解法的思路，我们把数组中数字的每一位累加起来对3取余，
剩下的结果就是那个单独数组该位上的数字，由于我们累加的过程都要对3取余，
那么每一位上累加的过程就是0->1->2->0，换成二进制的表示为00->01->10->00，那么我们可以写出对应关系：

00 (+) 1 = 01
01 (+) 1 = 10
10 (+) 1 = 00 ( mod 3)

那么我们用ab来表示开始的状态，对于加1操作后，得到的新状态的ab的算法如下：

b = b xor r & ~a;
a = a xor r & ~b;

我们这里的ab就是上面的三种状态00，01，10的十位和各位，
刚开始的时候，a和b都是0，当此时遇到数字1的时候，b更新为1，a更新为0，就是01的状态；
再次遇到1的时候，b更新为0，a更新为1，就是10的状态；
再次遇到1的时候，b更新为0，a更新为0，就是00的状态，相当于重置了；最后的结果保存在b中。

参考：
http://www.cnblogs.com/grandyang/p/4263927.html
https://leetcode.com/problems/single-number-ii/discuss/43294/challenge-me-thx
*/
public int singleNumber(int[] A) {
    int ones = 0, twos = 0;
    for(int i = 0; i < A.length; i++){
        ones = (ones ^ A[i]) & ~twos;
        twos = (twos ^ A[i]) & ~ones;
    }
    return ones;
}


///链表
///单链表定义
struct ListNode 
{
int val;
ListNode *next;
ListNode(int x) : val(x), next(nullptr) { }
};

//2.2.1 add two numbers
public ListNode* addtwonumbers(ListNode *L1,ListNode *L2)
{
	ListNode * dummy;
	ListNode * new = dummy;
	for(ListNode *p1 = L1,*p2 = L2;p1 != null ||p2 != null; p1 = p1->next,p2 = p2->next)
	{
		//注意，这里没有考虑特殊情况
		int a = p1->value;
		int b = p2->value;
		int value = (carry + a + b) % 10;
		int carry = (carry + a + b) / 10;
		new->next = new ListNode(value);
	}
	if(carry > 0)
	{
		new->next = new ListNode(carry);
	}
	return dummy;
}


//2.2.2 reverse link list II
//reverse list from position m to n 1<m<n<length
解法略

/*
常用的链表反向算法一，迭代法，注意保护好头指针
*/
ListNode *reverseBetween(ListNode *head, int m, int n) 
{
	ListNode dummy(-1);
	dummy.next = head;
	ListNode *prev = &dummy;
	for (int i = 0; i < m-1; ++i)
		prev = prev->next;
	ListNode* const head2 = prev;
	prev = head2->next;
	ListNode *cur = prev->next;
	for (int i = m; i < n; ++i) 
	{
		prev->next = cur->next;
		cur->next = head2->next;
		head2->next = cur; //头插法
		cur = prev->next;
	}
	return dummy.next;
}
 
/*
常用的链表反向算法二，递归法，注意保护好头指针
*/
static Node reverseLinkedList(Node node) 
{
    if (node == null || node.next == null) 
	{
        return node;
    } 
	else 
	{
        Node headNode = reverseLinkedList(node.next);
        node.next.next = node;
        node.next = null;
        return headNode;
    }
}


