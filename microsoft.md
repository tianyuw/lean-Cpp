__Why you choose?__
Although I'm currently working in automotive company, but I'm very interested in coding and information technology. I always want to work in a famous IT company like microsoft so I can sharp on my coding skills, and working on so many interesting & challenging projects At the same time, learn from other experienced people around me.
I also know that the microsoft is putting great amount of effort on Artifitial intelligence area especially Deep Learning technology, which is what I am currently working on and have great passion with.




__What is you most exciting working experience/project?__
The working experience / project that I feel most exciting / proud of is the first project I ever did in my career. The name of that project is called US Traffic Sign Detection. The purpose of that project is to build a computer vision based system that can detect and recognize different kinds of US traffic signs with high efficiency and high accuracy.




## 

```c++
2. 1. verticalprint 给一个整数 vertical 打印出每个整数的每个数字值。不允许用额外的空间
例子： input: 123456	
output: 
1
2
3
4
5
6
这道题主要就是一个求余数和递归的思路


void verticalPrint(int n)
{
	// 1. corner cases
    // n can be positive or negative number
	if(n < 0)
    	cout << "n must be positive number" << endl;
    else if(n < 10)
    	cout << n << endl;
    else
    {
    	// recursively print in reverse order
    	verticalPrint(n/10);
    	cout << n % 10 << endl;
    }
}
```


__253. Meeting Rooms 2__
```c++
Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), find the minimum number of conference rooms required.

For example,
Given [[0, 30],[5, 10],[15, 20]],
return 2.

what is decltype?
decltype is a way to specify a type: You give it an expression, and decltype gives you back a type which corresponds to the type of the expression. Specifically, decltype(e) is the following type:

If e is the name of a variable, i.e. an "id-expression", then the resulting type is the type of the variable.
Otherwise, if e evaluates to an lvalue of type T, then the resulting type is T &, and if e evaluates to an rvalue of type T, then the resulting type is T.



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
    int minMeetingRooms(vector<Interval>& intervals) {
        // 1. corner cases
        if(intervals.empty() || intervals.size() == 1)
        	return intervals.size();
        // 2. sort the input vector
        using Transition = pair<int, int>;
        auto cmp = [](const Transition & p1, const Transition & p2){
        	return (p1.first > p2.first || (p1.first == p2.first && p1.second > p2.second)); 
        };
        priority_queue<Transition, vector<Transition>, decltype(cmp)> Q(cmp);
        // 3. Push meeting start and ends in priority queue
        for(const auto & interv : intervals)
        {
        	Q.push(make_pair(interv.start, 1));
            Q.push(make_pair(interv.end, -1));
        }
        
        int maxRoom = 0;
        int cnt = 0;
        
        // 4. pop the queue and count how many rooms are in use.
        while(!Q.empty())
        {
        	cnt += Q.top().second;
            maxRoom = max(maxRoom, cnt);
            Q.pop();
        }
        return maxRoom;
    }
};

```


__75. Sort Colors__
```c++
Given an array with n objects colored red, white or blue, sort them so that objects of the same color are adjacent, with the colors in the order red, white and blue.

Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.

Note:
You are not suppose to use the library's sort function for this problem.

使用三个指针，p1表示红色和白色的分界线，p2表示白色和蓝色的分界线，i表示当前元素

即0~p1-1是红色的，p1~i-1表示白色的，p2+1~n-1表示蓝色的

1）如果当前元素是红色的，则和p1所指向的元素进行交换，由于交换以后i所指的颜色是白色的，则i直接遍历下一个元素

2）如果当前元素是蓝色的，则和p2所指向的元素进行交换，由于交换以后i所指的颜色可能是白色的，也可能是红色的，因此需要回退i

注意i应该在[p1,p2]之间

test case中有一个是[1, 0], 注意这种test case的时候i <= q就有必要了
i  ==  q的时候有必要判断一下.

class Solution {
public:
    void sortColors(vector<int>& nums) {
        // 1. corner cases
        if(nums.empty())
            return;
        // 2. three pointer 
        int p = 0, q = nums.size() - 1, i = 0;
        while(i <= q)
        {
            if(nums[i] == 0) swap(nums[p++], nums[i++]);
            else if(nums[i] == 2) swap(nums[q--], nums[i]);
            else i++;
        }
    }
};

```


__61. Rotate List__
```c++
Given a list, rotate the list to the right by k places, where k is non-negative.

For example:
Given 1->2->3->4->5->NULL and k = 2,
return 4->5->1->2->3->NULL.




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
    ListNode* rotateRight(ListNode* head, int k) {
        if(!head) return head;
        ListNode* cur = head;
        int ctr = 1;
        while(cur->next) {
            cur = cur->next;
            ctr++;
        }
        cur->next = head;
        cur = head;
        int i = 0;
        while(i<(ctr-(k+1)%ctr)) {
            cur = cur->next;
            i++;
        }
        ListNode* ret = cur->next;
        cur->next = nullptr;
        return ret;
    }
};
```



电影题，从文件里度input，input是每个电影的分数，取每个电影分数最高的5个，算平均，然后输出每个电影的平均值。分数range是 1- 100, 用heap就行了
```c++
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <queue>
#include <iostream>
#include <cstdlib>

using namespace std;

vector<float> calculateScore(string fileName, int topK)
{
    vector<float> result;
    // 1. corner cases
    if (topK < 1)
        return result;

    // 2. build a vector of min heap storing scores of each movie
    unordered_map<int, priority_queue<int, vector<int>, std::greater<int>>> score_heap;

    // 3. read from file, and store scores
    ifstream infile(fileName);
    string line;
    int id, rate;
    while (getline(infile, line))
    {
        int spaceLoc = line.find(" ");
        id = atoi(line.substr(0, spaceLoc).c_str());
        rate = atoi(line.substr(spaceLoc).c_str());
        if (score_heap[id].size() < topK)
            score_heap[id].push(rate);
        else
        {
            if (rate > score_heap[id].top())
            {
                score_heap[id].pop();
                score_heap[id].push(rate);
            }
        }
    }
    int element_num = score_heap.size();
    result.resize(element_num);
    // 4. calculate the average score of each movie
    for (auto element : score_heap)
    {
        int sum = 0;
        for (int i = 0; i < topK; i++)
        {
            sum += element.second.top();
            element.second.pop();
        }
        result[element.first] = (float)sum / (float)topK;
    }
    return result;
}


int main()
{
    vector<float> aveScore;
    aveScore = calculateScore("test.txt", 2);

    return 0;
}




```

__136. Single Number__
```c++
Given an array of integers, every element appears twice except for one. Find that single one.

Note:
Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?

和find duplicate正好相反，但是仍然可以用排序或hash table解决。排序以后，对每个坐标i，查找A[i-1], A[i+1]中是否有等于A[i]的，没有则为要找的数。或者用hash table/set来记录扫描过的数字。如果A[i]不在hash table中，则插入，如果已经在，则在hash table中删除，最后table中剩下的就是要找的数。但排序法事件复杂度是O(nlogn)，而hash table尽管是O(n)事件复杂度，需要o(n)的extra memory。

这题的终极解法是利用位运算中的异或：x^x = 0, x^0 = x。并且异或有交换律：1^1^0 = 0 = 1^0^1。所以如果将全部数字进行异或运算，所有重复元素都会被消除，最后的结果便是那个唯一的数。


class Solution
{
public: 
	int singleNumber(vector<int> & nums)
	{
		// 1. corner cases
		if(nums.empty()) return 0;
		if(nums.size() == 1) return nums[0];
		// 2. apply XOR to all elements.
		int result = 0;
		for(const auto & num : nums)
			result ^= num;
		return result;
	}
};
```

__137. Single Number 2__
```c++
Given an array of integers, every element appears three times except for one, which appears exactly once. Find that single one.

Note:
Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?


由于x^x^x = x，无法直接利用I的方法来解。但可以应用类似的思路，即利用位运算来消除重复3次的数。以一个数组[14 14 14 9]为例，将每个数字以二进制表达：

1110
1110
1110
1001
_____
4331    对每一位进行求和
1001    对每一位的和做%3运算，来消去所有重复3次的数

因为int 的位数有32位, 所以对所有位进行同样的操作.
从最高位往最低位处理,

class Solution
{
public:
	int singleNumber(vector<int> & nums)
	{
		// 1. corner cases
		if(nums.empty()) return 0;
		if(nums.size() == 1) return nums[0];
		// 2. bit manipulation
		int result = 0;
		for(int i = 31; i >= 0; i--)
		{
			// build mask
			int sum = 0,  mask = 1 << i;
			for(const auto & num : nums)
			{
				if(mask & num) 
					sum++;
			}
			// 先左移再写, 要是先写再左移的话会多移动一位.
			result <<= 1; 
			result += (sum%3);
			
		}
		return result;
	}
};
```
__442. Find All Duplicates in an Array__
```c++
Given an array of integers, 1 ≤ a[i] ≤ n (n = size of array), some elements appear twice and others appear once.

Find all the elements that appear twice in this array.

Could you do it without extra space and in O(n) runtime?

Example:
Input:
[4,3,2,7,8,2,3,1]

Output:
[2,3]

这道题的思路是, 对array进行循环, 因为array中的每个元素都是从1-n的, 所以我们可以去访问array[array[i] - 1] 对应位置上的元素.

1 < elem < n, so 0 < elem - 1 < n-1

我们每走到一个位置i, 就去查看array[array[i] - 1]位置上的元素是正还是负, 如果是正数, 我们就把他置为负. 如果再次在array中访问到了相同的元素, array[array[i] - 1]就已经是负数了, 这个时候我们就知道这个数之前出现过.




class Solution {
public:
    vector<int> findDuplicates(vector<int>& nums) {
        // 1. corner cases
        if(nums.empty())
            return nums;
        // 2. iterating through the array
        vector<int> result;
        for(int i = 0; i < nums.size(); i++)
        {
            if(nums[abs(nums[i]) - 1] >= 0)
                nums[abs(nums[i]) - 1] = -nums[abs(nums[i]) - 1];
            else
                result.push_back(abs(nums[i]));
        }
        return result;
    }
};


```


__268. Missing Number__
```c++
Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.

For example,
Given nums = [0, 1, 3] return 2.

Note:
Your algorithm should run in linear runtime complexity. Could you implement it using only constant extra space complexity?



比如对于上面的例子, 由于0 ^ 1 ^ 3 ^ 0 ^ 1 ^ 2 ^ 3 = 2
所以我们只需要让nums中的每个element和0-n的所有数进行异或, 就可以得到结果.

class Solution
{
public:
	int missingNumber(vector<int> & nums)
	{
		// 1. corner cases
		if(nums.empty())
			return 0;
		// 2. XOR operation
		int result, size = nums.size();
		for(int i = 0; i < size; i++)
		{
			result ^= i;
			result ^= nums[i];
		}
		result ^= size;
		return result;
	}
};
```

__116. Populating Next Right Pointers in Each Node__
```c++
Given a binary tree

    struct TreeLinkNode {
      TreeLinkNode *left;
      TreeLinkNode *right;
      TreeLinkNode *next;
    }
Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.

Note:

You may only use constant extra space.
You may assume that it is a perfect binary tree (ie, all leaves are at the same level, and every parent has two children).
For example,
Given the following perfect binary tree,
         1
       /  \
      2    3
     / \  / \
    4  5  6  7
After calling your function, the tree should look like:
         1 -> NULL
       /  \
      2 -> 3 -> NULL
     / \  / \
    4->5->6->7 -> NULL


/**
 * Definition for binary tree with next pointer.
 * struct TreeLinkNode {
 *  int val;
 *  TreeLinkNode *left, *right, *next;
 *  TreeLinkNode(int x) : val(x), left(NULL), right(NULL), next(NULL) {}
 * };
 */
BFS solution:
class Solution {
public:8
    void connect(TreeLinkNode *root) {
        // corner cases 
        if(root == NULL)
        	return;
        // level order traversal
        queue<TreeLinkNode * > Q;
        Q.push(root);
        TreeLinkNode * temp;
        while(!Q.empty())
        {
        	int size = Q.size();
        	for(int i = 0; i < size; i++)
        	{
            	temp = Q.front();
            	Q.pop();
                if(i == Q.size() - 1)
                	temp -> next = NULL;
                else
                	temp -> next = Q.front();
                if(temp -> left != NULL) Q.push(temp -> left);
                if(temp -> right != NULL) Q.push(temp -> right);
            }
        }
    }
};

DFS solution:
class Solution {
public:
    void connect(TreeLinkNode *root) {
  		if(root == NULL)
        	return;
        if(root -> left) root -> left -> next = root -> right;
        if(root -> right) root -> right -> next = (root -> next) ? root -> next -> left : NULL;
      	connect(root -> left);
        connect(root -> right);
    }
};


```

__151.Reverse Words in a String__
```c++
Given an input string, reverse the string word by word.

For example,
Given s = "the sky is blue",
return "blue is sky the".

Update (2015-02-12):
For C programmers: Try to solve it in-place in O(1) space.


Algorithm:

1) Reverse the individual words, we get the below string.
     "i ekil siht margorp yrev hcum"
2) Reverse the whole string from start to end and you get the desired output.
     "much very program this like i"

class Solution {
private:
	void reverse(string & s, int start, int end)
    {
    	char temp;
        while(start < end)
        {
        	temp = s[start];
            s[start] = s[end];
            s[end] = temp;
            start ++;
            end --;
        }
    }
public:
    void reverseWords(string &s) {
        //corner cases
        if(s.empty())
        	return;
        // step1: reverse individual words
        int word_begin = 0;
        for(int i = 0; i < s.size(); i++)
        {
        	if(i == s.size() - 1)
            	reverse(s, word_begin, i);
            else if(s[i] == ' ')
            {
            	reverse(s, word_begin, i - 1);
                word_begin = i + 1;
            }
        }
        // step2: reverse the whole string
        reverse(s, 0, s.size() - 1);
    }
};
```


__171. Excel Sheet Column Number__
```c++
Given a column title as appear in an Excel sheet, return its corresponding column number.

For example:

    A -> 1
    B -> 2
    C -> 3
    ...
    Z -> 26
    AA -> 27
    AB -> 28 
    
    
这道题其实是要把26进制数转化为10 进制数。
从右向左一位一位转

class Solution
{
public:
	int titleToNumber(string s)
    {
    	int result = 0;
    	for(int i = s.size() - 1; i >= 0; i--)
        {
        	result += (s[i] - 'A' + 1) * pow(26, s.size() - 1 - i);
        }
    	return result;
    }
};
```

__165. Compare Version Numbers__
```c++
Compare two version numbers version1 and version2.
If version1 > version2 return 1, if version1 < version2 return -1, otherwise return 0.

You may assume that the version strings are non-empty and contain only digits and the . character.
The . character does not represent a decimal point and is used to separate number sequences.
For instance, 2.5 is not "two and a half" or "half way to version three", it is the fifth second-level revision of the second first-level revision.

Here is an example of version numbers ordering:

0.1 < 1.1 < 1.2 < 13.37

class Solution {
public:
    int compareVersion(string version1, string version2) {
  		// vnum stores each numeric part of version
        int vnum1 = 0, vnum2 = 0;
        
        // loop until both string are processed
        for(int i = 0, j = 0; (i < version1.size() || j < version2.size()); )
        {
        	// storing numeric part of version 1 in vnum1
            while(i < version1.size() && version1[i] != '.')
            {
            	vnum1 = vnum1 * 10 + (version1[i] - '0');
            	i++;
            }
        	// storing numeric part of version 2 in vnum2
            while(i < version2.size() && version2[i] != '.')
            {
            	vnum2 = vnum2 * 10 + (version2[i] - '0');
                j++;
            }
        	if(vnum1 < vnum2) return -1;
            if(vnum1 > vnum2) return 1;
            
            // if equal, reset variables and go for next numeric
            vnum1 = 0;
            vnum2 = 0;
            i++;
            j++;
        }
        return 0;
    }
};
```

__138. Copy List with Random Pointer__
```c++
A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null.

Return a deep copy of the list.


The algorithm is composed of the follow three steps which are also 3 iteration rounds.

Iterate the original list and duplicate each node. The duplicate
of each node follows its original immediately.
Iterate the new list and assign the random pointer for each
duplicated node.
Restore the original list and extract the duplicated nodes.
The algorithm is implemented as follows:


RandomListNode * copyRandomList(RandomListNode *head)
{
	if(!head) return NULL;
    RandomListNode * run = head;
    // step1: Insert the copy of each node after it
    while(run)
    {
    	RandomListNode * copy = new RandomListNode(run -> label);
    	copy -> next = run -> next;
        run -> next = copy;
        run = run -> next -> next;
    }
	// step2: Set the random pointer for each copy
    run = head;
    while(run)
    {
    	if(run -> random)
        	run -> next ->random = run -> random -> next;
        run = run -> next -> next;
    }
    // step3: Extract the copy list
    RandomListNode * new_head = new RandomListNode(0);
    RandomListNode * new_run;
    run = head;
    new_head -> next = head -> next;
    while(run)
    {
    	new_run = run -> next;
        run -> next = new_run -> next;
        if(run -> next)
        	new_run -> next = new_run -> next -> next;
        run = run -> next;
    }
	return new_head -> next;
}
```
__273. Integer to English Words__

```c++
Convert a non-negative integer to its english words representation. Given input is guaranteed to be less than 231 - 1.

For example,
123 -> "One Hundred Twenty Three"
12345 -> "Twelve Thousand Three Hundred Forty Five"
1234567 -> "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven"


class Solution{
public:
	string numberToWords(int num)
	{
		// corner cases
		if(num == 0)
			return "Zero";
		int quotient, rem = num;
		vector<string> units = {"Billion", "Million", "Thousand"};
		string res = "";
		int  i = 0;
		// this will conduct 4 iteration, 
		// Billion, Million, Thousand, None
		for(int div = 1000000000; div >= 1; div /= 1000)
		{
			quotient = rem / div; // get how many units, if quotient equal 0, means no that unit
			rem %= div; // get remaining number
			res += threeDigitsToString(quotient) + ((i < 3 && quotient > 0) ? " " + units[i] : "") + ((div > 1 && quotient > 0 && rem > 0) ? " " : ""); // quotient need greater than 0 
			i++; // i will equal 0, 1, 2, 3
		}
		return res;
	}
	
	string threeDigitsToString(int num)
	{
		vector<string> units = {"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"};
		vector<string> tens = {"Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};
        	vector<string> decades = {"Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};
		string str;
		int rem, quotient;
		quotient = num / 100; //how many hundred
		rem = num % 100;
		if(quotient > 0) str += units[quotient] + " Hundred" + ((rem > 0) ? " " : "");
		if(rem >=10 && rem <20) 
		{
			str += tens[rem - 10];
			return str;
		}
		else if(rem >= 20)
		{
			quotient = rem / 10;
			rem = rem % 10;
			str += decades[quotient - 2] + (rem >0 ? " " : "");		
		}
		// in case only one digit 
		str += units[rem];
		return str;
	}
};

```

__348. Design Tic-Tac-Toe__
```c++
Design a Tic-tac-toe game that is played between two players on a n x n grid.

You may assume the following rules:

A move is guaranteed to be valid and is placed on an empty block.
Once a winning condition is reached, no more moves is allowed.
A player who succeeds in placing n of their marks in a horizontal, vertical, or diagonal row wins the game.
Example:
Given n = 3, assume that player 1 is "X" and player 2 is "O" in the board.

TicTacToe toe = new TicTacToe(3);

toe.move(0, 0, 1); -> Returns 0 (no one wins)
|X| | |
| | | |    // Player 1 makes a move at (0, 0).
| | | |

toe.move(0, 2, 2); -> Returns 0 (no one wins)
|X| |O|
| | | |    // Player 2 makes a move at (0, 2).
| | | |

toe.move(2, 2, 1); -> Returns 0 (no one wins)
|X| |O|
| | | |    // Player 1 makes a move at (2, 2).
| | |X|

toe.move(1, 1, 2); -> Returns 0 (no one wins)
|X| |O|
| |O| |    // Player 2 makes a move at (1, 1).
| | |X|

toe.move(2, 0, 1); -> Returns 0 (no one wins)
|X| |O|
| |O| |    // Player 1 makes a move at (2, 0).
|X| |X|

toe.move(1, 0, 2); -> Returns 0 (no one wins)
|X| |O|
|O|O| |    // Player 2 makes a move at (1, 0).
|X| |X|

toe.move(2, 1, 1); -> Returns 1 (player 1 wins)
|X| |O|
|O|O| |    // Player 1 makes a move at (2, 1).
|X|X|X|


player1 用 +1 记录, player2 用 -1 记录.
用vert和hori两个vector来记录每一行和每一列的和,如果某一行或者某一列的和为+3或-3的话就说明有一方赢了
diagonal 和 second_diagonal要用两个变量单独记录.




class TicTacToe {
private:
	int default_n;
	vector<int> vert;
	vector<int> horz;
	int diagonal;
	int second_diagonal;
	
	bool win(int c)
	{
		return c == default_n || c == -default_n;
	}

public:
    /** Initialize your data structure here. */
    TicTacToe(int n) {
    	       default_n = n;
    	       vert.resize(n, 0); // resize can initialize vector
    	       horz.resize(n, 0);
    	       diagonal = 0;
    	       second_diagonal = 0;
    }
    
    /** Player {player} makes a move at ({row}, {col}).
        @param row The row of the board.
        @param col The column of the board.
        @param player The player, can be either 1 or 2.
        @return The current winning condition, can be either:
                0: No one wins.
                1: Player 1 wins.
                2: Player 2 wins. */
    int move(int row, int col, int player) {
        // player == 1 then use +1, otherwise is -1
        int step = player == 1? 1 : -1;
        vert[row] += step;
        if(win(vert[row])) return player;
        horz[col] += step;
        if(win(horz[col])) return player;
        // diagonal count only at special condition
        if(row == col) 
        {
        	diagonal += step;
        	if(win(diagonal)) return player;
        }
        if(row + col == default_n - 1)
        {
        	second_diagonal += step;
        	if(win(second_diagonal)) return player;
        }
        return 0; // if no one win, return 0;
    }
};

/**
 * Your TicTacToe object will be instantiated and called as such:
 * TicTacToe obj = new TicTacToe(n);
 * int param_1 = obj.move(row,col,player);
 */
```



__54. Spiral Matrix__
```c++
Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.

For example,
Given the following matrix:

[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
You should return [1,2,3,6,9,8,7,4,5].

class Solution
{
public:
	vector<int> spiralOrder(vector<vector<int>> & matrix)
	{
		vector<int> result;
		// corner cases
		if(matrix.empty() || matrix[0].empty())
			return result;
		int rows = matrix.size();
		int cols = matrix[0].size();
		int T = 0, B = rows - 1;
		int L = 0, R = cols - 1;
		int dir = 0;
		while(T <= B && R >= L) // notice here
		{
			// go left to right
			if(dir == 0)
			{
				for(int i = L; i <= R; i++) // <=
					result.push_back(matrix[T][i]);
				T++;
			}
			// go top to bottom
			else if(dir == 1)
			{
				for(int i = T; i <= B; i++) // <=
					result.push_back(matrix[i][R]);
				R--;
			}
			// go right to left
			else if(dir == 2)
			{
				for(int i = R; i >=L; i--)
					result.push_back(matrix[B][i]);
				B--;
			}
			// go bottom to top
			else if(dir == 3)
			{
				for(int i = B; i >= T; i--)
					result.push_back(matrix[i][L]);
				L++;
			}
			dir = (dir + 1) % 4;
		}
		return result;
	}
};
```
__445. Add Two Numbers 2__
```c++
You are given two non-empty linked lists representing two non-negative integers. The most significant digit comes first and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Follow up:
What if you cannot modify the input lists? In other words, reversing the lists is not allowed.

Example:

Input: (7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 8 -> 0 -> 7

这道题是之前那道Add Two Numbers的拓展，我们可以看到这道题的最高位在链表首位置，如果我们给链表翻转一下的话就跟之前的题目一样了，这里我们来看一些不修改链表顺序的方法。由于加法需要从最低位开始运算，而最低位在链表末尾，链表只能从前往后遍历，没法取到前面的元素，那怎么办呢？我们可以利用栈来保存所有的元素，然后利用栈的后进先出的特点就可以从后往前取数字了，我们首先遍历两个链表，将所有数字分别压入两个栈s1和s2中，我们建立一个值为0的res节点，然后开始循环，如果栈不为空，则将栈顶数字加入sum中，然后将res节点值赋为sum%10，然后新建一个进位节点head，赋值为sum/10，如果没有进位，那么就是0，然后我们head后面连上res，将res指向head，这样循环退出后，我们只要看res的值是否为0，为0返回res->next，不为0则返回res即可，参见代码如下：




/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution{
public:
	ListNode * addTwoNumbers(ListNode * l1, ListNode * l2)
	{
		stack<int> s1, s2;
		while(l1)
		{
			s1.push(l1 -> val);
			l1 = l1 -> next;
		}
		while(l2)
		{
			s2.push(l2 -> val);
			l2 = l2 -> next;
		}
		
		int sum = 0;
		ListNode * res = new ListNode(0); // new node 
		while(!s1.empty() || !s2.empty())
		{
			if(!s1.empty()) {sum += s1.top(); s1.pop();}
			if(!s2.empty()) {sum += s2.top(); s2.pop();}
			res -> val = sum % 10; // construct current node 
			ListNode * head = new ListNode(sum / 10);
			head -> next = res;
			res = head; // processing res in each loop
			sum /= 10; // sum will carry the carry
		}
		return res -> val == 0 ? res ->next : res; // it is possible that the highest digits has carry. 
	}
};
```

__200. Number of Islands__

```c++
Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

Example 1:

11110
11010
11000
00000
Answer: 1

Example 2:

11000
11000
00100
00011
Answer: 3

// DFS solution
class Solution{
private:
	void DFS(vector<vector<char>> & grid, vector<vector<bool>> & visited, int row, int col)
	{
		vector<pair<int, int>> steps = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}; // four direction
		visited[row][col] = 1; // visited
		int new_row, new_col;
		int rows = grid.size(), cols = grid[0].size();
		for(const auto & step : steps)
		{
			new_row = row + step.first;
			new_col = col + step.second;
			if(new_row < rows && new_row >= 0 && new_col < cols && new_col >= 0 && !visited[new_row][new_col] && grid[new_row][new_col] == '1')
				DFS(grid, visited, new_row, new_col);
		}
	}
public:
	int numIslands(vector<vector<char>> & grid)
	{
		// corner cases
		if(grid.empty() || grid[0].empty())
			return 0;
		vector<vector<bool>> visited(grid.size(), vector<bool>(grid[0].size(), 0)); // visited initialized to all 0
		// count how many islands
		int count_islands = 0, rows = grid.size(), cols = grid[0].size();
		for(int i = 0; i < rows; i++)
		{
			for(int j = 0; j < cols; j++)
			{
				if(!visited[i][j] && grid[i][j] == '1')
				{
					DFS(grid, visited, i, j);
					count_islands++;
				}	
			}
		}
		return count_islands;
	}
};
```

__88. Merge Sorted Array__
```c++
Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.

Note:
You may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional elements from nums2. The number of elements initialized in nums1 and nums2 are m and n respectively.

把nums2中的数一个一个放入到nums1中去,当nums2放完的时候,merge就结束了
这道题需要特殊考虑 nums1 = [4, 5, 6], nums2 = [1, 2, 3]的情况

class Solution{
public:
	void merge(vector<int> & nums1, int m, vector<int> & nums2, int n)
	{
		// Three pointers
		int i = m - 1, j = n - 1, tar = m + n -1;
		while(j >= 0)
		{
			nums1[tar -- ] = (i >= 0 && nums1[i] > nums2[j]) ? nums1[i --] : nums2[j--];
		}
	}
};
```

__236. Lowest Common Ancestor of a Binary Tree__
```c++
Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes v and w as the lowest node in T that has both v and w as descendants (where we allow a node to be a descendant of itself).”

        _______3______
       /              \
    ___5__          ___1__
   /      \        /      \
   6      _2       0       8
         /  \
         7   4
For example, the lowest common ancestor (LCA) of nodes 5 and 1 is 3. Another example is LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
 
 两个node可以像上图中5,1 一样在different subtree中, 或者 像5,4一样在同一个subtree中.
 
The idea is to traverse the tree starting from root. If any of the given keys (n1 and n2) matches with root, then root is LCA (assuming that both keys are present). If root doesn’t match with any of the keys, we recur for left and right subtree. The node which has one key present in its left subtree and the other key present in right subtree is the LCA. If both keys lie in left subtree, then left subtree has LCA also, otherwise LCA lies in right subtree.
 
 // if we are sure that these two nodes exist in the tree
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
     		   // corner cases
     		   if(!root || !p || !q) return NULL;
     		   
     		   if(p == root || q == root)
     		   	return root;
     		   	
     		   TreeNode * left = lowestCommonAncestor(root -> left, p, q);
     		   TreeNode * right = lowestCommonAncestor(root -> right, p, q);
     		   
     		   if(left && right) return root;
     		   
     		   return (!left && right)? right : left;
    }
};

在上面的方法中如果有一个node在tree中, 有一个node不在的话. 也会返回一个LCA.
下面的程序可以解决这个问题

// if we are not sure that whether these two nodes exist in the tree or not
// use v1 and v2 to indicate whether two nodes exist in the tree
class Solution{
public: 
	TreeNode * lowestCommonAncestor(TreeNode * root, TreeNode * p, TreeNode * q)
	{
		// corner cases
		if(!root || !p || !q) return NULL;
		
		bool v1 = false, v2 = false;
		
		TreeNode * lca = helper(root, p, q, v1, v2);
		
		if(v1 && v2 || v1 && find(lca, q) || v2 && find(lca, p))
			return lca;
			
		return false;
	}
	TreeNode * helper(TreeNode * root, TreeNode * p, TreeNode * q, bool & v1, bool & v2)
	{
		if(!root) return NULL;
		
		if(root == p)
		{
			v1 = true;
			return root;
		}
		if(root == q)
		{
			v2 = true;
			return root;
		}
		
		TreeNode * left = helper(root -> left, p, q, v1, v2);
		TreeNode * right = helper(root -> right, p, q, v1, v2);
		
		if(left && right) return root;
		
		return (!left && right) ? right : left;
	}
	bool find(TreeNode * root, TreeNode * p)
	{
		if(!root || !p)
			return false;
		if(root == p || find(root -> left, p) || find(root -> right, p)) return true;
		
		return false; 
	}
};
```

__48. Rotate Image__
```c++
You are given an n x n 2D matrix representing an image.

Rotate the image by 90 degrees (clockwise).

Follow up:
Could you do this in-place?

class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
    		//corner cases
    		if(matrix.empty() || matrix[0].empty())   
      			return;
      		int first, last, temp, size = matrix.size();
      		// rotate the matrix layer by layer
      		for(int layer = 0; layer < size/2; layer ++)
      		{
      			first = layer;
      			// first is the same, but last will only scan to n-1 element
      			last = size - first - 1;
      			for(int i = first; i < last; i++)
      			{
      				int offset = i - first;
      				// move top to temp;
      				temp = matrix[first][i];
      				// move left to top
      				matrix[first][i] = matrix[last - offset][first];
      				// move bottom to left
      				matrix[last - offset][first] = matrix[last][last - offset];
      				// move right to bottom
      				matrix[last][last - offset] = matrix[i][last];
      				// move temp to right
      				matrix[i][last] = temp;
      			}
      		}
    }
};

```

__53. Maximam Subarray__
```c++
Find the contiguous subarray within an array (containing at least one number) which has the largest sum.

For example, given the array [-2,1,-3,4,-1,2,1,-5,4],
the contiguous subarray [4,-1,2,1] has the largest sum = 6.


这里用DP求解这个问题, 我们遍历整个array一次, 没到一个位置i我们都需要计算以nums[i]结尾的sum的最大值.
同时用一个max_by_now变量来存储所见过的最大值中最大的那个.

class Solution{
public:
	int maxSubArray(vector<int> & nums)
	{
		// corner cases
		if(nums.empty()) return 0;
		int current_max = nums[0], max_by_now = nums[0];
		for(int i = 1; i < nums.size(); i++) // take care here loop from 1
		{
			current_max = max(current_max + nums[i], nums[i]);
			max_by_now = max(max_by_now, current_max);
		}
		return max_by_now;
	}
};
```

__103. Binary Tree ZigZag Level Order Traversal__

```c++
Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).

For example:
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its zigzag level order traversal as:
[
  [3],
  [20,9],
  [15,7]
]
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution{
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
    	// corner cases
    	vector<vector<int>> result;
    	if(!root) return result;
    	
    	stack<TreeNode *> curLevel;
    	stack<TreeNode *> nextLevel;
    	curLevel.push(root);
    	bool left2right = true; // variable to save the TreeNode saving order
    	while(!curLevel.empty())
    	{
    	    result.push_back(vector<int>());
    	    while(!curLevel.empty())
    	    {
    		    TreeNode * current = curLevel.top();
    		    curLevel.pop();
    		    result.back().push_back(current -> val);
    		    if(!left2right)
    		    {
    			    if(current -> right) nextLevel.push(current -> right);
    			    if(current -> left) nextLevel.push(current -> left);
    		    }
    		    else if(left2right)
    		    {
    			    if(current -> left) nextLevel.push(current -> left);
    			    if(current -> right) nextLevel.push(current -> right);
    		    }
    	    }
    	    left2right = !left2right;
    	    curLevel = nextLevel;
    	    while(!nextLevel.empty()) nextLevel.pop();
    	}
    	return result;
    }
};
```

__73. Set Matrix Zeros__
```c++
Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in place.

click to show follow up.

Follow up:
Did you use extra space?
A straight forward solution using O(mn) space is probably a bad idea.
A simple improvement uses O(m + n) space, but still not the best solution.
Could you devise a constant space solution?

O(m+n)解法：用两个bool数组O(n)和O(m)，分别记录每行和每列的是否需要被置0。最后根据这两个数组来置0整个矩阵。
O(1)解法：用第0行和第0列来记录第1 ~ m-1行和第1 ~ n-1列是否需要置0。而用两个变量记录第0行和第0列是否需要置0。

class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        // corner cases
        if(matrix.empty() || matrix[0].empty())
        	return;
        // check the first row and col
        bool isFirstRowZero = false;
        bool isFirstColZero = false;
        int rows = matrix.size(); 
        int cols = matrix[0].size();
        for(int i = 0; i < rows; i++)
        {
        	if(matrix[i][0] == 0) 
        	{
        		isFirstColZero = true;
        		matrix[i][0] = 1; // set it to no zero value
        	}
        }
        for(int i = 0; i < cols; i++)
        {
        	if(matrix[0][i] == 0)
        	{
        		isFirstRowZero = true;
        		matrix[0][i] = 1;
        	}
        }
        // check rest of the matrix for 0
        for(int i = 0; i < rows; i++)
        {
        	for(int j = 0; j < cols; j++)
        	{
        		if(matrix[i][j] == 0)
        		{
        			matrix[i][0] = matrix[0][j] = 0;
        		}
        	}
        }
        
       // set rows and cols
       for(int i = 1; i < rows; i++)
       {
       		if(matrix[i][0] == 0)
       			setRow(matrix, i);
       }
       for(int i = 1; i < cols; i++)
       {
       		if(matrix[0][i] == 0)
       			setCol(matrix, i);
       }
	if(isFirstRowZero) setRow(matrix, 0);
	if(isFirstColZero) setCol(matrix, 0);
    }
};
```

__297. Serialize and Deserialize Binary Tree__
```c++
Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

For example, you may serialize the following tree

    1
   / \
  2   3
     / \
    4   5
as "[1,2,3,null,null,4,5]", just the same as how LeetCode OJ serializes a binary tree. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.
Note: Do not use class member/global/static variables to store states. Your serialize and deserialize algorithms should be stateless.

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Codec {
private:
	void serialize(TreeNode * root, stringstream & out)
	{
		if(root)
		{
			out << root -> val << " ";
			serialize(root -> left, out);
			serialize(root -> right, out);
		}
		else
		{
			out << "# "; // for NULL encode
		}
	}
	TreeNode * deserialize(stringstream & in)
	{
		string val;
		in >> val; // here we get string
		if(val == "#")
			return NULL;
		TreeNode * root = new TreeNode(stoi(val)); // string convert to int
		root -> left = deserialize(in);
		root -> right = deserialize(in);
		return root;
	}
public:
    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        stringstream out;
        serialize(root, out);
        return out.str();
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        stringstream in(data);
        return deserialize(in);
    }
};

// Your Codec object will be instantiated and called as such:
// Codec codec;
// codec.deserialize(codec.serialize(root));
```

__238. Product of Array Except Self__
```c++
Given an array of n integers where n > 1, nums, return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].

Solve it without division and in O(n).

For example, given [1,2,3,4], return [24,12,8,6].

Follow up:
Could you solve it with constant space complexity? (Note: The output array does not count as extra space for the purpose of space complexity analysis.)

solve it in two round
first round construct left vector storing the product before i, exclude i
first round construct right vector storing the product after i, exclude i

class Solution{
public:
	vector<int> productExceptSelf(vector<int> & nums)
	{
		int sz = nums.size();
		vector<int> result(sz, 0), left(sz, 0), right(sz, 0);
		// corner cases
		if(nums.empty() || nums.size() == 1)
			return result;
		// construct left product
		left[0] = 1; right[sz - 1] = 1;
		for(int i = 1; i < sz; i++)
			left[i] = left[i-1] * nums[i-1];
		for(int i = sz-2; i >= 0; i--)
			right[i] = right[i+1] * nums[i+1];
		for(int i = 0; i < sz; i++)
			result[i] = left[i] * right[i];
		return result;
	}
};
```

__141. Linked List Cycle__
```c++
Given a linked list, determine if it has a cycle in it.

Follow up:
Can you solve it without using extra space?

这道题就是判断一个链表是否存在环，非常简单的一道题目，我们使用两个指针，一个每次走两步，一个每次走一步，如果一段时间之后这两个指针能重合，那么铁定存在环了。

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
    bool hasCycle(ListNode *head) {
        if(head == NULL || head -> next == NULL)
        	return false;
        ListNode * fast = head;
        ListNode * slow = head;
        
        while(fast -> next != NULL && fast -> next -> next != NULL)
        {
        	fast = fast -> next -> next;
            slow = slow -> next;
        	if(slow == fast)
            	return true;
        }
        return false;
    }
};
```

__160. Linked List Cycle 2__
```c++
Given a linked list, return the node where the cycle begins. If there is no cycle, return null.

Note: Do not modify the linked list.

Follow up:
Can you solve it without using extra space?


紧跟着第一题，这题不光要求出是否有环，而且还需要得到这个环开始的节点。譬如下面这个，起点就是n2。

        n6-----------n5
        |            |
  n1--- n2---n3--- n4|
我们仍然可以使用两个指针fast和slow，fast走两步，slow走一步，判断是否有环，当有环重合之后，譬如上面在n5重合了，那么如何得到n2呢？

首先我们知道，fast每次比slow多走一步，所以重合的时候，fast移动的距离是slot的两倍，我们假设n1到n2距离为a，n2到n5距离为b，n5到n2距离为c，fast走动距离为a + b + c + b，而slow为a + b，有方程a + b + c + b = 2 x (a + b)，可以知道a = c，所以我们只需要在重合之后，一个指针从n1，而另一个指针从n5，都每次走一步，那么就可以在n2重合了。
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
    ListNode *detectCycle(ListNode *head) {
        // corner cases
        if(!head) return NULL;
        // find cycle
        ListNode * fast = head, * slow = head;
        while(fast -> next != NULL && fast -> next -> next != NULL)
        {
        	fast = fast -> next -> next;
            slow = slow -> next;
        	if(fast == slow)
        	{
            	slow = head;
                while(slow != fast)
                {
                	slow = slow -> next;
                    fast = fast -> fast;
                }
				return slow;
        	}
        }
        return NULL;
    }
};
```

__106. Intersection of Two Linked Lists__
```c++
Write a program to find the node at which the intersection of two singly linked lists begins.


For example, the following two linked lists:

A:          a1 → a2
                   ↘
                     c1 → c2 → c3
                   ↗            
B:     b1 → b2 → b3
begin to intersect at node c1.


Notes:

If the two linked lists have no intersection at all, return null.
The linked lists must retain their original structure after the function returns.
You may assume there are no cycles anywhere in the entire linked structure.
Your code should preferably run in O(n) time and use only O(1) memory.

有以下几种思路：
（1）暴力破解，遍历链表A的所有节点，并且对于每个节点，都与链表B中的所有节点比较，退出条件是在B中找到第一个相等的节点。时间复杂度O(lengthA*lengthB)，空间复杂度O(1)。

（2）哈希表。遍历链表A，并且将节点存储到哈希表中。接着遍历链表B，对于B中的每个节点，查找哈希表，如果在哈希表中找到了，说明是交集开始的那个节点。时间复杂度O(lengthA+lengthB)，空间复杂度O(lengthA)或O(lengthB)。

（3）双指针法，指针pa、pb分别指向链表A和B的首节点。
遍历链表A，记录其长度lengthA，遍历链表B，记录其长度lengthB。
因为两个链表的长度可能不相同，比如题目所给的case，lengthA=5，lengthB=6，则作差得到lengthB-lengthA=1，将指针pb从链表B的首节点开始走1步，即指向了第二个节点，pa指向链表A首节点，然后它们同时走，每次都走一步，当它们相等时，就是交集的节点。
时间复杂度O(lengthA+lengthB)，空间复杂度O(1)。双指针法的代码如下：

ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
       ListNode *pa=headA,*pb=headB;
       int lengthA=0,lengthB=0;
       while(pa) {pa=pa->next;lengthA++;}
       while(pb) {pb=pb->next;lengthB++;}
       if(lengthA<=lengthB){
           int n=lengthB-lengthA;
           pa=headA;pb=headB;
           while(n) {pb=pb->next;n--;}
       }else{
           int n=lengthA-lengthB;
           pa=headA;pb=headB;
           while(n) {pa=pa->next;n--;}
       }
       while(pa!=pb){
           pa=pa->next;
           pb=pb->next;
       }
       return pa;
   }
```

__168. Excel Sheet Column Title__
```c++
Given a positive integer, return its corresponding column title as appear in an Excel sheet.

For example:

    1 -> A
    2 -> B
    3 -> C
    ...
    26 -> Z
    27 -> AA
    28 -> AB 

class Solution {
public:
    string convertToTitle(int n) {
        string res = "";
        while(n != 0)
        {
        	n--;
        	res = (char)(n % 26 + 'A') + res;
        	n /= 26;
        }
        return res;
    }
};
```

__12. Integer to Roman__
```c++
Given an integer, convert it to a roman numeral.

Input is guaranteed to be within the range from 1 to 3999.

思路：

这题需要一些背景知识，首先要知道罗马数字是怎么表示的：

http://en.wikipedia.org/wiki/Roman_numerals

I: 1
V: 5
X: 10
L: 50
C: 100
D: 500
M: 1000

字母可以重复，但不超过三次，当需要超过三次时，用与下一位的组合表示：
I: 1, II: 2, III: 3, IV: 4
C: 100, CC: 200, CCC: 300, CD: 400

s = 3978
3978/1000 = 3: MMM
978>(1000-100), 998/900 = 1: CM
78<(100-10), 78/50 = 1 :L
28<(50-10), 28/10 = XX
8<(100-1), 8/5 = 1: V
3<(5-1), 3/1 = 3: III
ret = MMMCMLXXVIII

所以可以将单个罗马字符扩展成组合形式，来避免需要额外处理类似IX这种特殊情况。
I: 1
IV: 4
V: 5
IX: 9
X: 10
XL: 40
L: 50
XC: 90
C: 100
CD: 400
D: 500
CM: 900
M: 1000


class Solution{
public: 
	string intToRoman(int num)
	{
		string dict[13] = {"M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"};
		int val[13] = {1000,900,500,400,100,90,50,40,10,9,5,4,1};
		string ret;
		for(int i = 0; i < 13; i++)
		{
			if(num >= val[i])
			{
				int count = num / val[i];
				num %= val[i];
				for(int j = 0; j < count; j++)
				{
					ret += dict[i];
				}
			}
		}
		return ret;
	}
};
```

__13. Roman to Integer__
```c++
Given a roman numeral, convert it to an integer.

Input is guaranteed to be within the range from 1 to 3999.
I, IV, V, IX, X, XL, L, XC, C, CD, D, CM, M

class Solution{
public: 
	int romanToInt(string s)
	{
		string dict[] = {"M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"};
		int num[] = {1000,900,500,400,100,90,50,40,10,9,5,4,1};
		int s_idx = 0, dict_idx = 0, res = 0;
		while(s_idx < s.size() && dict_idx < 13)
        {
        	string target = dict[dict_idx];
            string cur = s.substr(s_idx, target.size());
            if(cur == target)
            {
            	res += num[dict_idx];
            	s_idx += target.size(); // 注意这里只更新s_idx,而dict_idx不更新. 会有XX在一起的情况出现. 要多次判断某个字符.
            }
        	else
            {
            	dict_idx ++;
            }
        }
		return res;
	}
};
```

__21. Merge Two Sort Lists__
```c++
Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.

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
        ListNode dummy(INT_MIN);
        ListNode * tail = &dummy;
        while(l1 && l2)
        {
        	if(l1 -> val < l2 -> val)
            {
            	tail -> next = l1;
                l1 = l1 -> next;
            }
            else
            {
            	tail -> next = l2;
                l2 = l2 -> next;
            }
			tail = tail -> next;
        }
        tail -> next = (!l1) ? l2 : l1;
        return dummy.next;
    }
};


```

__117. Populating Next Right Pointers in Each Node 2__
```c++
Follow up for problem "Populating Next Right Pointers in Each Node".

What if the given tree could be any binary tree? Would your previous solution still work?

Note:

You may only use constant extra space.
For example,
Given the following binary tree,
         1
       /  \
      2    3
     / \    \
    4   5    7
After calling your function, the tree should look like:
         1 -> NULL
       /  \
      2 -> 3 -> NULL
     / \    \
    4-> 5 -> 7 -> NULL

这道题要求用constant space

/**
 * Definition for binary tree with next pointer.
 * struct TreeLinkNode {
 *  int val;
 *  TreeLinkNode *left, *right, *next;
 *  TreeLinkNode(int x) : val(x), left(NULL), right(NULL), next(NULL) {}
 * };
 */
class Solution {
public:
	// based on level order traversal
    void connect(TreeLinkNode *root) {
        TreeLinkNode * head = NULL; // head of the next level;
        TreeLinkNode * prev = NULL; // the leading node on the next level
        TreeLinkNode * cur = root; // current node of current level
        
        // traverse all level
        while(cur != NULL)
        {	// traverse all nodes on current level
        	while(cur != NULL)
            {
            	// left child
                if(cur -> left)
                {
                	if(prev != NULL)
                    	prev -> next = cur -> left;
                    else
                    	head = cur -> left;
                    prev = cur -> left;
                }
                // right child
                if(cur -> right != NULL)
                {
                	if(prev != NULL)
                    	prev -> next = cur -> right;
                   	else
                    	head = cur -> right;
                    prev = cur -> right;
                }
                // move to next node
				cur = cur -> next;
            }
            // go to new level
            cur = head;
            prev = NULL;
            head = NULL;
        }
    }
};
```

__8. String to Integer(atoi)__
```c++
Implement atoi to convert a string to an integer.

Hint: Carefully consider all possible input cases. If you want a challenge, please do not see below and ask yourself what are the possible input cases.

Notes: It is intended for this problem to be specified vaguely (ie, no given input specs). You are responsible to gather all the input requirements up front.

细节实现题，这种题主要考察corner case是否都考虑全了，需要和面试官多交流。这里需要考虑的有：

1. 起始空格、非数字字符。
2. 正负号。
3. 0为数字中的最高位
4. overflow/underflow
5. 末尾非数字字符

根据以上枚举的特殊情况，判断的流程如下：

0. 用一个unsigned long long ret来记录结果，用bool isNeg记录正负。

1. 跳过所有起始空格，找到第一个非空格字符c0
(a) c0非数字字符（包括\0）也非正负号，返回0。
(b) c0为+或-，若下一个字符c1为数字，用isNeg记录下正或负，并移到c1。否则返回0。
(c) c0为数字，则更新ret

2. 经过1已经找到了第一个数字字符，以及结果的正负。此时只需要逐个扫描后面的数字字符，并更新ret即可。终止的条件有以下几个：
(a) 扫到一个非数字字符，包括\0和空格。返回ret
(b) ret已经超出int的范围，返回相应的INT_MIN/INT_MAX

isdigit used for check whether is a decimal digit character.


class Solution{
public: 
	int atoi(string str)
    {
    	// sign 
    	bool isNeg = false;
        // unsigned long long for number
        unsigned long long ret = 0;
        
        int idx = 0;
        
        // skip leading white spaces
        while(str[idx] == ' ') idx++;
        
        // first none white space char must be + - or digit
        if(!isdigit(str[idx]) || str[idx] == '+' || str[idx] == '-') return 0;
        
        // for + -, next char must be digit
        if(str[idx] == '+' || str[idx] == '-')
        {
        	if(!isdigit(str[idx + 1])) return 0;
        	if(str[idx] == '-') isNeg = true;
        	idx++;
        }
    	
        while(isdigit(str[idx]))
        {
        	ret = ret * 10 + str[idx] - '0';
            if(ret > (unsigned long long)INT_MAX)
            	return isNeg ? INT_MIN : INT_MAX;
            idx++;
        }
    	return isNeg ? -(int)ret : (int)ret;
    }
};
```

__208. Implement Trie(Prefix Tree)__
```c++
Implement a trie with insert, search, and startsWith methods.

Note:
You may assume that all inputs are consist of lowercase letters a-z.

字典树主要有如下三点性质：

1. 根节点不包含字符，除根节点意外每个节点只包含一个字符。

2. 从根节点到某一个节点，路径上经过的字符连接起来，为该节点对应的字符串。

3. 每个节点的所有子节点包含的字符串不相同。

 

字母树的插入（Insert）、删除（ Delete）和查找（Find）都非常简单，用一个一重循环即可，即第i 次循环找到前i 个字母所对应的子树，然后进行相应的操作。实现这棵字母树，我们用最常见的数组保存（静态开辟内存）即可，当然也可以开动态的指针类型（动态开辟内存）。至于结点对儿子的指向，一般有三种方法：

1、对每个结点开一个字母集大小的数组，对应的下标是儿子所表示的字母，内容则是这个儿子对应在大数组上的位置，即标号；

2、对每个结点挂一个链表，按一定顺序记录每个儿子是谁；

3、使用左儿子右兄弟表示法记录这棵树。

三种方法，各有特点。第一种易实现，但实际的空间要求较大；第二种，较易实现，空间要求相对较小，但比较费时；第三种，空间要求最小，但相对费时且不易写。

 

我们先来看第一种实现方法，这种方法实现起来简单直观，字母的字典树每个节点要定义一个大小为26的子节点指针数组，然后用一个标志符用来记录到当前位置为止是否为一个词，初始化的时候讲26个子节点都赋为空。那么insert操作只需要对于要插入的字符串的每一个字符算出其的位置，然后找是否存在这个子节点，若不存在则新建一个，然后再查找下一个。查找词和找前缀操作跟insert操作都很类似，不同点在于若不存在子节点，则返回false。查找次最后还要看标识位，而找前缀直接返回true即可。代码如下：


class TrieNode{
public:
	// Initialize your data structure here
    vector<TrieNode*>child;
    bool isWord;
    TrieNode() : isWord(false){
        child.resize(26, NULL);
    }
};


class Trie {
private:
	TrieNode * root;
public:
    /** Initialize your data structure here. */
    Trie() {
        root = new TrieNode();
    }
    
    /** Inserts a word into the trie. */
    void insert(string word) {
        TrieNode * p = root; // p will traverse
        for(auto & a : word)
        {
        	int letter = a - 'a';
        	if(!p -> child[letter]) p ->child[letter] = new TrieNode();
            p = p -> child[letter];
        }
        p -> isWord = true;
    }
    
    /** Returns if the word is in the trie. */
    bool search(string word) {
        TrieNode * p = root;
        for(auto & a : word)
        {
        	int letter = a - 'a';
        	if(!p -> child[letter]) return false;
        	p = p -> child[letter];
        }
        return p -> isWord;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(string prefix) {
        TrieNode * p = root;
        for(auto & a : prefix)
        {
        	int i = a - 'a';
            if(!p -> child[i]) return false;
            p = p -> child[i];
        }
        return true;
    }
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.insert(word);
 * bool param_2 = obj.search(word);
 * bool param_3 = obj.startsWith(prefix);
 */
```

__212. Word Search ||__
```c++
Given a 2D board and a list of words from the dictionary, find all words in the board.

Each word must be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.

For example,
Given words = ["oath","pea","eat","rain"] and board =

[
  ['o','a','a','n'],
  ['e','t','a','e'],
  ['i','h','k','r'],
  ['i','f','l','v']
]
Return ["eat","oath"].
Note:
You may assume that all inputs are consist of lowercase letters a-z.

这道题是在之前那道Word Search 词语搜索的基础上做了些拓展，之前是给一个单词让判断是否存在，现在是给了一堆单词，让返回所有存在的单词，在这道题最开始更新的几个小时内，用brute force是可以通过OJ的，就是在之前那题的基础上多加一个for循环而已，但是后来出题者其实是想考察字典树的应用，所以加了一个超大的test case，以至于brute force无法通过，强制我们必须要用字典树来求解。LeetCode中有关字典树的题还有 Implement Trie (Prefix Tree) 实现字典树(前缀树)和Add and Search Word - Data structure design 添加和查找单词-数据结构设计，那么我们在这题中只要实现字典树中的insert功能就行了，查找单词和前缀就没有必要了，然后DFS的思路跟之前那道Word Search 词语搜索基本相同，请参见代码如下：

class Solution{
public:
	struct TrieNode{
    	vector<TrieNode * >child;
        string str;
        TrieNode() : str(""){
        	child.resize(26, NULL);
        }
    };
	struct Trie
    {
    	TrieNode *root;
        Trie(){root = new TrieNode();
        void insert(string s)
        {
        	TrieNode * p = root;
            for(auto & a : s)
            {
            	int i = a - 'a';
                if(!p -> child[i])
                	p -> child[i] = new TrieNode();
                p = p -> child[i];
            }
        	p -> str = s;
        }
    };
	vector<string> findWords(vector<vector<char>>& board, vector<string>& words) 
    {
    	vector<string> res;
    	// corner cases
        if(board.empty() || board[0].empty() || words.empty()) return res;
        vector<vector<bool>> visit(board.size(), vector<bool>(board[0].size(), false));
        Trie T;
        for(auto & a: words) T.insert(a);
        for(int i = 0; i < board.size(); i++)
        {
        	for(int j = 0; j < board[0].size(); j++)
            {
            	int letter = board[i][j] - 'a';
                if(T.child[letter])
                search(board, T.child[letter], i, j, visit, res);
            }
        }
    }
    void search(vector<vector<char>>& board, TrieNode *p, int i, int j, vector<vector<bool>> & visit, vector<string>&res)
    {
    	if(!p -> str.empty())
        {
        	res.push_back(p -> str);
            p -> str.clear(); // for words like 'ab' and 'abc'
    	}
    	vector<pair<int>> steps = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        visit[i][j] = true; // visited
        int rows = board.size();
        int cols = board[0].size();
        for(const auto & step : steps)
        {
        	int nx = i + step.first, ny = j + step.second;
            int letter = board[nx][ny] - 'a';
            if(nx >= 0 && nx < rows && ny >= 0 && ny < cols && !visit[nx][ny] && p -> child[letter])
            	search(board, p -> child[letter], nx, ny, visit, res);
        }
        visit[i][j] = false; // backtracking 
    }
};
```

__33. Search in Rotated Sorted Array__
```c++
Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).

You are given a target value to search. If found in the array return its index, otherwise return -1.

You may assume no duplicate exists in the array.


class Solution {
public:
    int search(vector<int>& nums, int target) {
        // corner cases
        if(nums.empty()) return -1;
        
        // fisrt find the location of pivot
        int left = 0, right = nums.size() - 1, mid, realmid, pivot;
        if(nums[left] < nums[right])
        	pivot = 0;
        else
        {
        	while(left +1 < right)
        	{
        		mid = left + (right - left) / 2;
           	 	if(nums[mid] > nums[left])
            		left = mid;
            	else
            		right = mid;
        	}
        	pivot = (nums[left] < nums[right]) ? left : right;
        }
        
        // using pivot to find target
        left = 0; right = nums.size() - 1;
        while(left <= right)
        {
        	mid = left + (right - left) / 2;
            realmid = (mid + pivot) % nums.size();
            if(nums[realmid] == target)	return realmid;
            else if(nums[realmid] < target) left = mid + 1;
            else right = mid - 1;
        }
        return -1;
    }
};
```


__232. Implement Queue using Stacks__
···c++
Implement the following operations of a queue using stacks.

push(x) -- Push element x to the back of queue.
pop() -- Removes the element from in front of queue.
peek() -- Get the front element.
empty() -- Return whether the queue is empty.
Notes:
You must use only standard operations of a stack -- which means only push to top, peek/pop from top, size, and is empty operations are valid.
Depending on your language, stack may not be supported natively. You may simulate a stack by using a list or deque (double-ended queue), as long as you use only standard operations of a stack.
You may assume that all operations are valid (for example, no pop or peek operations will be called on an empty queue).

enQueue(q,  x)
  1) Push x to stack1 (assuming size of stacks is unlimited).

deQueue(q)
  1) If both stacks are empty then error.
  2) If stack2 is empty
       While stack1 is not empty, push everything from stack1 to stack2.
  3) Pop the element from stack2 and return it.

class MyQueue {
stack<int> sk1, sk2;
public:
    /** Initialize your data structure here. */
    MyQueue() {
        
    }
    
    /** Push element x to the back of queue. */
    void push(int x) {
        sk1.push(x);
    }
    
    void exchange()
    {
    	while(!sk1.empty())
        {
        	sk2.push(sk1.top());
            sk1.pop();
        }
    }
    
    
    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        int i;
        if(sk2.empty())
        	exchange();
        i = sk2.top();
        sk2.pop();
        return i;
    }
    
    /** Get the front element. */
    int peek() {
        if(sk2.empty()) exchange();
        return sk2.top();
    }
    
    /** Returns whether the queue is empty. */
    bool empty() {
        return sk1.empty() && sk2.empty();
    }
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue obj = new MyQueue();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.peek();
 * bool param_4 = obj.empty();
 */
···


__24. Swap Nodes in Pairs__
```c++
Given a linked list, swap every two adjacent nodes and return its head.

For example,
Given 1->2->3->4, you should return the list as 2->1->4->3.

Your algorithm should use only constant space. You may not modify the values in the list, only nodes itself can be changed.

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
    ListNode* swapPairs(ListNode* head) {
        ListNode * dummy = new ListNode(0);
        dummy -> next = head;
        ListNode * cur = dummy;
        
        while(cur -> next && cur -> next -> next)
        {
        	swap2(cur);
            cur = cur -> next -> next;
        }
        return dummy -> next;
    }
    void swap2(ListNode * cur)
    {
    	ListNode * temp = cur -> next;
        cur -> next = temp -> next;
        temp -> next = temp -> next -> next;
        cur -> next -> next = temp;
    }
};
```

__191. Number of 1 Bits__
```c++
Write a function that takes an unsigned integer and returns the number of ’1' bits it has (also known as the Hamming weight).
For example, the 32-bit integer ’11' has binary representation 00000000000000000000000000001011, so the function should return 3.

class Solution{
public:
	int hammingWeight(uint32_t n)
    {
    	int count = 0;
        while(n != 0)
        {
        	n &= (n-1);
            count ++;
        }
    	return count;
    }
};
```

__26. Remove Duplicates from Sorted Array__
```c++
Given a sorted array, remove the duplicates in place such that each element appear only once and return the new length.

Do not allocate extra space for another array, you must do this in place with constant memory.

For example,
Given input array nums = [1,1,2],

Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively. It doesn't matter what you leave beyond the new length.

class Solution {
public:
    int removeDuplicates(vector<int> & nums)
    {
        // corner cases
        if(nums.empty())
            return 0;
	    int cur = 1;
        for(int i = 1; i < nums.size(); i++)
        {
    	    if(nums[i] > nums[i - 1])
    		    nums[cur++] = nums[i];
        }
        return cur;
    }
};


```
__215. Kth Largest Elment in an Array__
```c++
Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.

For example,
Given [3,2,1,5,6,4] and k = 2, return 5.

Note: 
You may assume k is always valid, 1 ≤ k ≤ array's length.


class Solution{
public: 
	int findKthLargest(vector<int> & nums, int k)
    {
    	int left = 0, right = nums.size() - 1;
        while(true)
        {
        	int pos = partition(nums, left, right);
            if(pos == k - 1) return nums[pos];
            if(pos > k-1) right = pos - 1;
            else left = pos + 1;
        }
    }
    int partition(vector<int> & nums, int left, int right)
    {
    	int pivot = nums[left];
        int l = left + 1, r = right;
        while(l <= r)
        {
        	if(nums[l] < pivot && nums[r] > pivot)
            	swap(nums[l++], nums[r--]);
            if(nums[l] >= pivot) l++;
            if(nums[r] <= pivot) r--;
        }
    	swap(nums[left], nums[r]);
        return r;
    }
};
```
__189. Rotate Array__
```c++
Rotate an array of n elements to the right by k steps.

For example, with n = 7 and k = 3, the array [1,2,3,4,5,6,7] is rotated to [5,6,7,1,2,3,4].

Note:
Try to come up as many solutions as you can, there are at least 3 different ways to solve this problem.

class Solution{
public:
	void rotate(vector<int> & nums, int k)
    {	
    	if(nums.empty() || k < 0) return;
        k = k % nums.size();
        reverse(nums, 0, nums.size() - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.size() - 1);
    }
    void reverse(vector<int> & nums, int left, int right)
    {
    	int temp;
    	while(left <= right)
        {
        	temp = nums[left];
            nums[left++] = nums[right];
            nums[right--] = temp;
        }
    }
};
```


__71. Simplify Path__
```c++
Given an absolute path for a file (Unix-style), simplify it.

For example,
path = "/home/", => "/home"
path = "/a/./b/../../c/", => "/c"
click to show corner cases.

Corner Cases:
Did you consider the case where path = "/../"?
In this case, you should return "/".
Another corner case is the path might contain multiple slashes '/' together, such as "/home//foo/".
In this case, you should ignore redundant slashes and return "/home/foo".

归下类的话，有四种字符串：
1. "/"：为目录分隔符，用来分隔两个目录。
2. "."：当前目录
3. ".."：上层目录
4. 其他字符串：目录名

简化的核心是要找出所有的目录，并且如果遇到".."，需要删除上一个目录。


class Solution{
public: 
	string simplifyPath(string path)
    {
    	string ret, curDir;
    	vector<string> allDir;
    	path.push_back('/');
        for(int i = 0 ; i < path.size(); i++)
        {
        	if(path[i] == '/')
            {
            	if(curDir.empty())
                	continue;
                else if(curDir == ".")
                	curDir.clear();
                else if(curDir == "..")
                {
                	if(!allDir.empty()) allDir.pop_back();
                    curDir.clear();
                }
                else
                {
                	allDir.push_back(curDir);
                    curDir.clear();
                }
            }
            else
            {
            	curDir.push_back(path[i]);
            }

        }
        for(int i = 0; i < allDir.size(); i++)
        {
        	ret.append("/" + allDir[i]);
        }
    	if(ret.empty()) ret = "/";
        return ret;
    }
};
```

__173. Binary Search Tree Iterator__
```c++
Implement an iterator over a binary search tree (BST). Your iterator will be initialized with the root node of a BST.

Calling next() will return the next smallest number in the BST.

Note: next() and hasNext() should run in average O(1) time and uses O(h) memory, where h is the height of the tree.


/**
 * Definition for binary tree
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class BSTIterator {
public:
	stack<TreeNode *> iter;
    BSTIterator(TreeNode *root) {
        while(root != NULL)
        {
        	iter.push(root);
        	root = root -> left;
        }
    }

    /** @return whether we have a next smallest number */
    bool hasNext() {
        if(iter.empty()) return false;
        return true;
    }

    /** @return the next smallest number */
    int next() {
        TreeNode * node = iter.top();
        iter.pop();
        int ret = node -> val;
        if(node -> right)
        {
        	node = node -> right;
        	while(node)
            {
            	iter.push(node);
                node = node -> left;
            }
        }
        return ret;
    }
};

/**
 * Your BSTIterator will be called like this:
 * BSTIterator i = BSTIterator(root);
 * while (i.hasNext()) cout << i.next();
 */
```

__15. 3Sum__
```c++
Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

Note: The solution set must not contain duplicate triplets.

For example, given array S = [-1, 0, 1, 2, -1, -4],

A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]

题目要求的是a+b+c=0，问题可以推广到a+b+c=target。3sum问题可以转化为2sum问题：对于任意一个A[i]，在数组中的其他数中解2sum问题，目标为target-A[i]。与2sum那题不同，这题要求返回的不是index而是数字本身，并且解不唯一。同时要求解排序并去重。

对排序来说，2sum中的双指针法更为方便，因为算法本身就用到排序。双指针排序法本身会去除一些重复的可能性：

(1, 2, 3, 4), target = 6
在扫描1时，解(2, 3, 4)的2sum = 5问题，找到一个解(1, 2, 3)。
在扫描2时，应当只对后面的数解2sum问题，即对(3, 4)解2sum = 4问题。这样避免再次重复找到解(1, 2, 3)。

但当存在重复数字时，光靠排序仍然无法去重：

(1, 2, 2, 2, 3, 4), target = 9
扫描第一个2时，解(2, 2, 3, 4)中的2sum=7问题，得到解(2, 3, 4)
扫描第二个2时，解(2, 3, 4)中的2sum=7问题，仍然会得到(2, 3, 4)

去除因重复数字而造成重复解有两个办法，一是将结果存到一个hash table中。由于STL的hash table (unordered_set, unordered_map)并不能为vector类型，除非自己提供一个hash function，颇为不便，也增加额外存储空间。而另一种方法就是在扫描数组时跳过重复的数字。上例中，只扫描1, 2, 3, 4来求相应的2sum问题。进一步简化，可以只扫描1, 2。因为3已经是倒数第二个数字，不可能有以它为最小数字的解。

class Solution{
public:
	vector<vector<int>> threeSum(vector<int> & nums)
    {
    	int target = 0;
    	vector<vector<int>> allSol;
        if(nums.size() < 3) return allSol;
        sort(nums.begin(), nums.end());
        for(int i = 0; i < nums.size() - 2; i++) // need three element
        {
        	if(i > 0 && nums[i] == nums[i-1]) continue;
            int left = i+1, right = nums.size()-1;
            while(left < right)
            {
				int cur = nums[left] + nums[right];
                int tar = target - nums[i];
                if(cur == tar)
                {
                	allSol.push_back(vector<int>());
                    allSol.back().push_back(nums[left]);
                    allSol.back().push_back(nums[right]);
                    allSol.back().push_back(nums[i]);
                    left++;
                    right--;
                    while(nums[left] == nums[left - 1]) left++;
                    while(nums[right] == nums[right + 1]) right--;
                }
                else if(cur < tar)
                	left++;
                else
                	right--;
            }
        }
        return allSol;
    }
};
```

__387. First Unique Character in a string__
```c++
Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, return -1.

Examples:

s = "leetcode"
return 0.

s = "loveleetcode",
return 2.

这道题确实没有什么难度，我们只要用哈希表建立每个字符和其出现次数的映射，然后按顺序遍历字符串，找到第一个出现次数为1的字符，返回其位置即可，参见代码如下：

class Solution {
public:
    int firstUniqChar(string s) {
        unordered_map<char, int> m;
        for (char c : s) ++m[c];
        for (int i = 0; i < s.size(); ++i) {
            if (m[s[i]] == 1) return i;
        }
        return -1;
    }
};
```


__23. Merge k Sorted Lists__
```c++
思路1： priority queue

将每个list的最小节点放入一个priority queue (min heap)中。之后每从queue中取出一个节点，则将该节点在其list中的下一个节点插入，以此类推直到全部节点都经过priority queue。由于priority queue的大小为始终为k，而每次插入的复杂度是log k，一共插入过nk个节点。时间复杂度为O(nk logk)，空间复杂度为O(k)。


class Solution{
public: 
	ListNode * mergeKLists(vector<ListNode * > & lists)
    {
    	auto cmpNode = [](ListNode * p1, ListNode * p2){return p1 -> val > p2 -> val;};
   		priority_queue<ListNode *, vector<ListNode * >, decltype(cmpNode)> Q(cmp); 	
    	ListNode * dummy = new ListNode(0), *tail = dummy;
        
        // push all k head of linkedList Node to min heap
        for(int i = 0; i < lists.size(); i++)
        	if(lists[i]) Q.push(lists[i]);
            
       	while(!Q.empty())
        {
        	tail -> next = Q.top();
            Q.pop();
            tail = tail -> next;
            if(tail -> next) Q.push(tail -> next);
        }
		return dummy -> next;
    }
};
```


__25. Reverse Nodes in k-Group__
```c++
Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.

k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes in the end should remain as it is.

You may not alter the values in the nodes, only nodes itself may be changed.

Only constant memory is allowed.

For example,
Given this linked list: 1->2->3->4->5

For k = 2, you should return: 2->1->4->3->5

For k = 3, you should return: 3->2->1->4->5







```

__270. Closest Binary Search Tree Value__
```c++
Given a non-empty binary search tree and a target value, find the value in the BST that is closest to the target.

Note:
Given target value is a floating point.
You are guaranteed to have only one unique value in the BST that is closest to the target.


/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int closestValue(TreeNode* root, double target) {
    	int a = root -> val; // notice here!!
        if(target < root -> val && root -> left) a = closestValue(root -> left, target);
        else if(target > root -> val && root -> right) a = closestValue(root -> right, target);
        return (abs(a - target) < abs(root -> val - target)) ? a : root -> val;
    }
};





```












