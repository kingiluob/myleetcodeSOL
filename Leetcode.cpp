///Leetcode solutions

///2.1.1 remove duplicate from sortd array
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
