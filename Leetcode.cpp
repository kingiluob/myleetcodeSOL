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
//两个排好序的数列，找到
s
// ᬥ䬣฼ᱱᏕ O(log(m+n))喌⾩䬣฼ᱱᏕ O(log(m+n))
class Solution {
public:
double findMedianSortedArrays(const vector<int>& A, const vector<int>& B) {
const int m = A.size();
const int n = B.size();
int total = m + n;
if (total & 0x1)
return find_kth(A.begin(), m, B.begin(), n, total / 2 + 1);
else
return (find_kth(A.begin(), m, B.begin(), n, total / 2)
+ find_kth(A.begin(), m, B.begin(), n, total / 2 + 1)) / 2.0;
}
private:
static int find_kth(std::vector<int>::const_iterator A, int m,
std::vector<int>::const_iterator B, int n, int k) {
//always assume that m is equal or smaller than n
if (m > n) return find_kth(B, n, A, m, k);
if (m == 0) return *(B + k - 1);
if (k == 1) return min(*A, *B);
//divide k into two parts
int ia = min(k / 2, m), ib = k - ia;
if (*(A + ia - 1) < *(B + ib - 1))
return find_kth(A + ia, m - ia, B, n, k - ia);
else if (*(A + ia - 1) > *(B + ib - 1))
return find_kth(A, m, B + ib, n - ib, k - ib);
else
return A[ia - 1];
}
};