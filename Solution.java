package myproj;

//import java.util.Scanner;
import java.util.*;
import java.lang.reflect.Field;
import java.util.Map.Entry;

import java.io.*;
import java.text.*;
import java.math.*;

public class Solution {
	public static class CPoint{
		double x;
		double y;
		public CPoint(double a, double b){
			x = a;
			y = b;
		}
	}
	public static CPoint[] findClose(CPoint[] arr, int k){
		if (k==0) return new CPoint[0];
		PriorityQueue<CPoint> minHeap = new PriorityQueue<CPoint>(k, new Comparator<CPoint>(){
			@Override
			public int compare(CPoint a, CPoint b){
				double disa = a.x * a.x + a.y * a.y;
				double disb = b.x * b.x + b.y * b.y;
				if (disb > disa){
					return -1;
				}else if(disb < disa){
					return 1;
				}else{
					return 0;
				}
			}
		});
		for (CPoint p : arr){
			minHeap.offer(p);
		}
		CPoint[] ret = new CPoint[k];
		for (int i = 0; i < k; i++){
			ret[i] = minHeap.poll();
		}
		return ret;
	}
	
	public static class JNode{
		int val;
		JNode next;
		JNode(int x){
			val=x;
		}
	}
	public static JNode reverseHalf(JNode head){
		if (head==null || head.next==null || head.next.next==null) return head;
		JNode fast = head.next;
		JNode slow = head;
		while (fast.next!=null && fast.next.next!=null){
			slow = slow.next;
			fast = fast.next.next;
		}
		JNode secondHead = slow.next;
		JNode p1 = secondHead;
		JNode p2 = p1.next;
		while(p1!=null && p2!=null){
			JNode tmp = p2.next;
			p2.next = p1;
			p1 = p2;
			p2 = tmp;
		}
		secondHead.next = null;
		slow.next = (p2==null?p1:p2);
		JNode loop = head;
		while (loop!=null){
			System.out.print(loop.val);
			loop = loop.next;
			
		}
		return head;
	}
	
	public static class Container {
		public double first;
		public double second;
    }

	public static Container findOptimalWeights(double capacity, double[]weights, int 
			numOfContainers){ 
        Container res = new Container(); 
        if(weights == null || weights.length <= 0) return res;
        Arrays.sort(weights); //weights.sort();
        if(weights[0] >= capacity) return res;
        int begin = 0, end = weights.length-1;
        while(begin < end && weights[begin] + weights[end] > capacity){
                end--;
        }
        while(begin < end && weights[begin] + weights[end] <= capacity){
                begin++;
        }
        res.first = --begin; //预定义double类？
        res.second = end;
        System.out.print(res.first+" "+res.second);
        return res;
}

	
	public static int[] twoSum(int[] nums, int target) {
        int[] ret = new int[2];
        if(nums==null || nums.length<2) return ret;
        //copy original list and sort
        int[] copylist = new int[nums.length];
        System.arraycopy(nums,0,copylist,0,nums.length);
        Arrays.sort(copylist);
        int low = 0;
        int high = copylist.length-1;
        while (low<high){
            if(copylist[low] + copylist[high] < target){
                low++;
            }
            else if(copylist[low] + copylist[high] > target){
                high--;
            }
            else{
                ret[0] = low;
                ret[1] = high;
                break;
            }
        }
        //find index from original list
        int i;
        for(i = 0;i<nums.length&&copylist[ret[0]]!=nums[i];i++){}
        ret[0] = i; 
        for(i = nums.length-1;i>=0&&copylist[ret[1]]!=nums[i];i--){}
        ret[1] = i;
        Arrays.sort(ret);
        return ret;
	}
	
	public static int[] windowSum(int[] nums, int n){
		if(nums==null||nums.length<n) return new int[0];
		int[] ret = new int[nums.length-n+1];
		int sum=0;
		for (int i = 0;i<nums.length-n+1;i++){
			if(i==0){
				for (int j = 0;j < n;j++){
					sum+=nums[i+j];
				}
			}
			else{
				sum = sum + nums[i+n-1]- nums[i];
			}
			ret[i] = sum;
		}
		return ret;
	}
	
	public static void print(int row){
		for(int i = 1; i<=row;i++){
			char ch = 'a';
			char print = ch;
			for (int j = 0;j<i;j++){
				System.out.print((print++));
			}
			System.out.println("");
		}
	}
	
	private static int maxMin;
	public static int minPathMax(int[][] mat){
		maxMin = Integer.MIN_VALUE;
        dfs(mat, 0, 0, Integer.MAX_VALUE);
        return maxMin;
	}
	private static void dfs(int[][] mat, int i, int j, int minSofar){
        if(mat.length==0 || mat[0].length==0) return;
        int M = mat.length, N = mat[0].length;
        if(i==M || j==N) return;

        minSofar = Math.min(minSofar, mat[i][j]);
        if(i==M-1 && j==N-1) {
                maxMin = Math.max(minSofar, maxMin);
        }        

        dfs(mat, i+1, j, minSofar);
        dfs(mat, i, j+1, minSofar);
}
	
	public static class Process{
		int arrivalTime;
		int executeTime;
		Process(int arr, int exc){
			arrivalTime = arr;
			executeTime = exc;
		}
	}
	
	public static float waitingTimeRobin(int[] arrival, int[] run, int q){
		if (arrival == null || run == null || run.length != arrival.length){
			return 0;
		}
		Queue<Process> queue = new LinkedList<Process>();
		int curTime = 0;
		int waitTime = 0;
		int nextProIdx = 0;
		while (!queue.isEmpty() || nextProIdx < arrival.length){
			if (!queue.isEmpty()){
				Process cur = queue.poll();
				//continue summing up the waitingTime
				waitTime += curTime - cur.arrivalTime;
				curTime += Math.min(cur.executeTime, q);
				//if arrival time of next process is smaller than current time, the next process
				//should be pushed into the queue
				for (int i = nextProIdx; i < arrival.length; i++){
					if(arrival[i] <= curTime){
						queue.offer(new Process(arrival[i],run[i]));
						nextProIdx = i+1;
					}
					else{
						break;
					}
				}
				//push the interrupted process into the tail of the queue
				if (cur.executeTime > q){
					queue.offer(new Process(curTime, cur.executeTime-q));
				}
			}
			else{//push element in arrival time array and corresponding run time array into queue
					queue.offer(new Process(arrival[nextProIdx],run[nextProIdx]));
					//update the current time point
					curTime = arrival[nextProIdx++];
				}
			
		}
		return (float)waitTime / arrival.length ;
	}
	
	
	public static int[][] rotate(int[][] matrix, int flag) {
        if (matrix == null || matrix.length == 0 ||matrix[0].length == 0) {
             return null;
        }
        int rows = matrix.length;
        int cols = matrix[0].length;
        int[][] right = new int[cols][rows];
        int[][] left = new int[cols][rows];
        if (flag == 1) {
        	for (int i = 0; i < rows; i++) {
        		for (int j = 0; j < cols; j++) {
                   right[j][rows - 1 - i] = matrix[i][j];
        		}
        	}
        	return right;
        }else if (flag == 0) {
        	for (int i = 0; i < rows; i++) {
        		for (int j = 0; j < cols; j++) {
        			left[cols - 1 -j][i] = matrix[i][j];
        		}
        	}
        	return left;
        }
        return null;
	}

	public static void printMatrix(int[][] test) {
		for (int i = 0; i < test.length; i++) {
			for (int j = 0; j < test[i].length; j++) {
				System.out.print(" " + test[i][j]);
			}
			System.out.println();
		}
	}
	
	public static class TreeNode{
		int val;
		TreeNode left;
		TreeNode right;
		TreeNode(int x){
			val = x;
		}
	}
	
	public static int minPath(TreeNode root){
		if (root == null) return 0;
		if (root.left!=null && root.right == null)
			return minPath(root.left) + root.val;
		if (root.right != null && root.left == null)
			return minPath(root.right) + root.val;
		return Math.min(minPath(root.left), minPath(root.right)) + root.val;
	}
	
	
	public static int minPath2(TreeNode root){
		if (root==null) return 0;
		if (root.left == null && root.right == null) return root.val;
		
		int minSum = Integer.MAX_VALUE;
		Stack<TreeNode> pathNode = new Stack<TreeNode>();
		Stack<Integer> pathSum = new Stack<Integer>();
		pathNode.push(root);
		pathSum.push(root.val);
		
		while (!pathNode.isEmpty()){
			TreeNode cur = pathNode.pop();
			int curSum = pathSum.pop();
			if (cur.left == null && cur.right == null){
				if (minSum > curSum){
					minSum = curSum;
				}
			}
			if (cur.right != null){
				pathNode.push(cur.right);
				pathSum.push(cur.right.val + curSum);
			}
			if (cur.left != null){
				pathNode.push(cur.left);
				pathSum.push(cur.left.val + curSum);
			}
		}
		return minSum;
	}
	
	
	public static JNode insert(JNode head, int x){
		if (head == null){
			head = new JNode(x);
			head.next = head;
			return head;
		}
		JNode cur = head;
		JNode pre = null;
		do{
			pre = cur;
			cur = cur.next;
			if (x <= cur.val && x >= pre.val) break;
			if (pre.val > cur.val && (x > pre.val || x < cur.val)) break;
		} while (cur != head);
		JNode newNode = new JNode(x);
		newNode.next = cur;
		pre.next = newNode;
		return newNode;
	}
	
	
	public static float SJFaverage(int[] request, int[] duration){
		if (request == null || duration == null || request.length != duration.length) return 0;
		PriorityQueue<Process> heap = new PriorityQueue<Process>(new Comparator<Process>(){
			public int compare(Process p1,  Process p2){
				if (p1.executeTime == p2.executeTime) return p1.arrivalTime - p2.arrivalTime;
				return p1.executeTime - p2.executeTime;
			}
		});
		int index = 0;
		int curTime = 0;
		int waitTime = 0;
		int len = request.length;
		while(! heap.isEmpty() || index < len){
			if (! heap.isEmpty()){
				Process cur = heap.poll();
				waitTime += curTime - cur.arrivalTime;
				curTime += cur.executeTime;
				while (index < len && curTime >= request[index])
					heap.offer(new Process(request[index], duration[index++]));
			}else{
				heap.offer(new Process(request[index], duration[index]));
				curTime = request[index++];
			}
		}
		return (float)waitTime / len;
	}
	
	
	public static int countMiss(int[] arr, int size){
		if (arr == null || arr.length == 0) return 0;
		List<Integer> cache = new ArrayList<Integer>();
		int count = 0;
		for (int i = 0; i < arr.length; i++){
			if (cache.contains(arr[i])){
				cache.remove(cache.indexOf(arr[i]));
				cache.add(arr[i]);
			}else{
				cache.add(arr[i]);
				count++;
			}
			if (cache.size() > size){
				cache.remove(0);
			}
		}
		return count;
	}
	
	public static int[] dayChange(int[] arr, int days){
		if (arr == null || arr.length <= 1 || days <= 0) return arr;
		int len = arr.length;
		//preNum represents previous day's list
		int[] preNum = new int[len];
		preNum = arr;
		for (int i = 0; i < days; i++){
			int[] curNum = new int[len];
			curNum[0] = preNum[1];
			curNum[len-1] = preNum[len-2];
			for (int j = 1; j < len - 1; j++){
				curNum[j] = preNum[j-1] ^ preNum[j+1];
			}
			preNum = curNum;
		}
		return preNum;
	}
	public static int maze(int[][] grid){ //only allow go right or down, not fully tested
		int m = grid.length;
		int n = grid[0].length;
		int[][] dp = new int[m][n];
		dp[0][0] = 1;
		for (int i = 1; i < m; i++){
			if (grid[i][0] == 9) return dp[i-1][0];
			dp[i][0] = (grid[i][0] == 1 ? 1 : 0);
			if (dp[i][0] == 0) break;
		}
		for (int j = 1; j < n; j++){
			if (grid[0][j] == 9) return dp[0][j-1];
			dp[0][j] = (grid[0][j] == 1 ? 1 : 0);
			if (dp[0][j] == 0) break;
		}
		//printMatrix(dp);
		int i = 1;
		int j = 1;
		for(i = 1; i < m; i++){
			for(j = 1; j < n; j++){
				if (grid[i][j] == 0){
					dp[i][j] = 0;
				}
				else if (grid[i][j] != 9){
					dp[i][j] = ((dp[i-1][j] == 1 || dp[i][j-1] == 1) ? 1 : 0);
				}else{
					int ret = ((dp[i-1][j] == 1 || dp[i][j-1] == 1) ? 1 : 0);
					return ret;
				}
			}
		}
		return 0;
	}
	
	public static class point {
        int x;
        int y;
        point(int x, int y) {
        	this.x = x;
        	this.y = y; }
	}
	

	
	public static int maze2(int[][] grid){//use a queue
		if (grid == null) return 0;
		int m = grid.length;
		int n = grid[0].length;
		int[][] visited = new int[m][n];
		Queue<point> queue = new LinkedList<point>();
		int[] dx = {0, 0, 1, -1};
		int[] dy = {1, -1, 0, 0};
		
		queue.offer(new point(0,0));
		while (!queue.isEmpty()){
			point tmp = queue.poll();
			visited[tmp.x][tmp.y] = 1;
			if(grid[tmp.x][tmp.y]==9) return 1;
			else if(grid[tmp.x][tmp.y]==0) continue;
			else{
				for(int i = 0; i<4; i++){
					if(tmp.x + dx[i] >= 0 && tmp.x + dx[i] < m && tmp.y + dy[i] >= 0 && tmp.y + dy[i] < n){
						if(visited[tmp.x + dx[i]][tmp.y + dy[i]]==0){
							queue.offer(new point(tmp.x+dx[i],tmp.y+dy[i]));
						}
					}
				}
			}
		}
		return 0;
	}
	
	public static int maze3(int[][] grid)  {//use dfs
        int rows = grid.length;
        int cols = grid[0].length;
        int[][] visited = new int[rows][cols];
        if(dfs(grid, 0, 0, visited))
        	return 1;
        return 0;
	}
	public static boolean dfs(int[][] grid, int i, int j, int[][] visited){
		if(i >= 0 && i < grid.length && j >= 0 && j < grid[0].length && grid[i][j]==9){
			visited[i][j] = 1;
			return true;
		}
		if (isSafe(grid, i ,j) == true && visited[i][j] == 0){
			visited[i][j] = 1;
			if (dfs(grid, i-1, j, visited) == true)
				return true;
			if (dfs(grid, i+1, j, visited) == true)
				return true;
			if (dfs(grid, i, j+1, visited) == true)
				return true;
			if (dfs(grid, i, j-1, visited) == true)
				return true;
			visited[i][j] = 0;
			return false;
			}
		return false;
	}
    
	public static boolean isSafe(int[][]grid, int i, int j){
		if (i >= 0 && i < grid.length && j >= 0 && j < grid[0].length && grid[i][j] == 1){
			return true;
		}
		return false;
	}
	
	//public static boolean dfshelper(String[][] nums, int i, int[] visited){
		
	//}
	
	static String[] simpleWords(String[] words) {
        HashSet<String> set = new HashSet<String>(words.length);
        for (String s : words){
        	set.add(s);
        }
        ArrayList<String> simple = new ArrayList<String>();
        for (String s : words){
        	if (!isCompound(s,set)){
        		simple.add(s);
        	}
        }
        String[] simpleArr = new String[simple.size()];
        simpleArr = simple.toArray(simpleArr);
        return simpleArr;
    }
    static boolean isCompound(String word, HashSet<String> set) {
        //boolean wordInSet = false;
        //if (set.contains(word)){
        	//wordInSet = true;
        	set.remove(word);
        //}
        if (word.length()==0) return false;
        
        boolean[] dp = new boolean[word.length()];
        for (int i = 0; i < word.length(); i++){
        	dp[i] = false;
        }
        for (int i = 0; i < word.length(); i++){
        	helper(set, word.substring(0,i+1), dp);
        }
        //if (wordInSet){
        	set.add(word);
        //}
        return dp[word.length()-1];
        
    }
    static void helper(HashSet<String> set, String s, boolean[] dp) {
    	if (set.contains(s)){
    		dp[s.length()-1] = true;
    		return;
    	}
    	else{
    		for (int i = 0; i < s.length(); i++){
    			if (dp[i]){
    				if (set.contains(s.substring(i+1,s.length()))){
    					dp[s.length()-1] = true;
    					return;
    				}
    			}
    		}
    	}
        
    }
    static boolean isValid(int x, int y, Character tmp, Character[][] grid){
    	for (int i=0;i<9;i++){
    		if (tmp.equals(grid[i][y])) return false;
    	}
    	for (int j = 0; j< 9; j++){
    		if (tmp.equals(grid[x][j])) return false;
    	}
    	for (int i = 0; i < 3; i++){
    		for (int j = 0; j < 3; j++){
    			if (tmp.equals(grid[(x/3)*3+i][(y/3)*3+j])) return false;
    		}
    	}
    	return true;
    }
    static int sudoku(String input){
    	Character[][] grid = new Character[9][9];
		for (int i = 0; i < 9; i++){
			for (int j = 0; j < 9; j++){
				grid[i][j] = input.charAt(i*9+j);
			}
		}
		for (int i = 0; i < 9; i++){
			for(int j=0;j<9;j++){
				if (isValid(i,j,grid[i][j],grid)) continue;
				else return 0;
			}
		}
		return 1;
    }
    public static int minPalindrome(String s){
    	//http://www.geeksforgeeks.org/dynamic-programming-set-28-minimum-insertions-to-form-a-palindrome/
    	int n = s.length();
    	int[][] dp = new int[n][n];
    	//fill table according to gap number
    	for (int gap = 1; gap < n; gap++){
    		for (int low = 0, high = gap; high < n; low++,high++){
    			dp[low][high] = s.charAt(low)==s.charAt(high) ? dp[low+1][high-1] : Math.min(dp[low+1][high], Math.min(dp[low][high-1], dp[low+1][high-1]))+1;
    			//dp[low][high] = s.charAt(low)==s.charAt(high) ? dp[low+1][high-1] : Math.min(dp[low+1][high], dp[low][high-1])+1;
    		}
    	}
    	return dp[0][n-1];
    }
    
    public static boolean dfsRock(int i, int v, Character[] s){
    	if (i > s.length - 1) return false;
    	if (i == s.length-1 || s[i].equals('O')) return true;
    	if (v <= 0) return false;
    	
    	if (s[i].equals('F')){
    		return dfsRock(i+1,1,s);
    	}
    	else if (s[i].equals('R')){
    		return dfsRock(i+(v-1),v-1, s) || dfsRock(i+v,v,s) || dfsRock(i+(v+1),v+1,s);
    	}
    	else return false;
    }
    public static ArrayList<ArrayList<Integer>> zigzag(int[][] matrix){
    	if (matrix == null) return new ArrayList<ArrayList<Integer>>();
    	ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
    	int m = matrix.length;
    	int n = matrix[0].length;
    	int i = 0;
    	int j = 0;
    	for (int k = 0; k < m+n-1; k++){
    		ArrayList<Integer> tmp = new ArrayList<Integer>();
    		if (k < n){
    			i = 0;
    			j = k;
    		}else{
    			j = n-1;
    			i = k-j;
    		}
    		while (i < m && j >= 0){
    			tmp.add(matrix[i][j]);
    			i++;
    			j--;
    		}
    		result.add(tmp);
    	}
    	return result;
    }
    
    public static boolean hasAnagram(String s, String t){
    	if ((s == null || s.length() == 0) && (t== null || t.length() == 0)) return true;
    	if ( s == null || s.length() == 0) return false;
    	if (t == null || t.length() == 0) return true;
    	HashMap<Character, Integer> map = new HashMap<>();
    	for (Character c : t.toCharArray()){
    		map.put(c, map.getOrDefault(c, 0)+1);
    	}
    	int missing = t.length();
    	int left = 0;
    	int tLength = t.length();
    	for (int right = 0; right < s.length(); right++){
    		char cur = s.charAt(right);
    		if(map.containsKey(cur)){
    			map.put(cur, map.get(cur)-1);
    			missing--;
    		}
    		if(right >= tLength){
    			char curLeft = s.charAt(left);
    			if(map.containsKey(curLeft)){
    				map.put(curLeft, map.get(curLeft)+1);
    				missing++;
    			}
    			left++;
    		}
    		if (missing == 0) break;
    	}
    	return missing==0;
    }
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		System.out.println(Math.pow(2, 3));
		String s = "aferews";
		String t = "aefr";
		System.out.println(hasAnagram(s,t));
		
		int[] a = {1,2,3,4,5,6,7};
		int[] b = Arrays.copyOfRange(a, 2, 5);
		for(int i = 0;i < b.length;i++){
		System.out.println(b[i]);}
		String[] x = {"rexy","xz","abcd","ef","z"};
		System.out.println(x[0].indexOf(x[1]));
		Arrays.sort(x);
		for(int i = 0; i < x.length;i++){
			System.out.println(x[i]);
		}
		String[] words = {"chat", "ever", "snapchat", "snap", "salesperson", "per", "person", "sales", "son", "whatsoever", "what", "","so"};
		String[] res = simpleWords(words);
		for(String sq : res){
			System.out.println(sq);
		}
		//System.out.println(minPalindrome(s));
		//Character[] r = {'F', 'R', 'R', 'R', 'W', 'W', 'R', 'W', 'R','W', 'O'};
		/*Character[] r = {'F', 'R', 'W', 'R', 'W', 'W','R','W','O'};
		System.out.println(r.length);
		System.out.println(dfsRock(0,1,r));*/
		/*int[][] matrix = {{1,2,3,4},{5,6,7,8},{9,10,11,12}};
		ArrayList<ArrayList<Integer>> result = zigzag(matrix);
		Iterator<ArrayList<Integer>> iter1 = result.iterator();
		while(iter1.hasNext()){
			System.out.println(iter1.next());
		}
		System.out.println(result);
		ArrayList<Integer> t = new ArrayList<Integer>();
		t.add(1);
		t.add(2);
		t.add(3);
		ArrayList<ArrayList<Integer>> s = new ArrayList<ArrayList<Integer>>();
		s.add(t);
		s.add(t);
		System.out.println(s);*/
		
		/*StringBuilder n = new StringBuilder();
		n.append(" ");
		n.append("#");
		
		System.out.println(n);
		
		String[] words = {"chat", "ever", "snapchat", "snap", "salesperson", "per", "person", "sales", "son", "whatsoever", "what", "","so"};
		String[] res = simpleWords(words);
		for(String s : res){
			System.out.println(s);
		}
		
		String input = "163174258178325649254689731821437596496852317735961824589713462317246985642598173";
		System.out.print(sudoku(input));*/
		
		
		/*int[][] arr = {{1,0,9,0},{1,1,1,0},{1,0,1,1}};
		int[][] arr2 = {{1,1,0,0},{1,0,9,0},{1,0,1,1}};
		printMatrix(arr2);
		System.out.println(maze3(arr2));*/
		
		/*Scanner in = new Scanner(System.in);
		int row = in.nextInt();
		int col = in.nextInt();
		
		String[][] nums = new String[row][col];
		
		for(int i = 0; i < row; i++){
			for (int j = 0; j< col; j++){
				nums[i][j] = in.next();
			}
		}

		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				System.out.print(" " + nums[i][j]);
			}
			System.out.println();
		}
		List<Integer> list = new ArrayList<>();
		list.add(3);
		list.add(5);
		list.add(7);
		for (int te : list){
			System.out.println(te);
		}*/
		/*JNode head = new JNode(0);
		JNode dummy = new JNode(0);
		dummy = head;
				
		for (int i = 0; i < nums.length-1;i++){
			head.val = nums[i];
			head.next = new JNode(0);
			head=head.next;
		}
		head.val = nums[nums.length-1];
		head.next = dummy;
				
		JNode loop = dummy;
		int count = 0;
		while (count < nums.length){
			System.out.print(loop.val);
			loop = loop.next;
			count++;
		}
		System.out.println();
		//reverseHalf(dummy.next);
		JNode ans = insert(dummy,5);
		JNode tmp = ans;
		int cnt = 0;
		while (cnt < nums.length + 1){
			System.out.print(tmp.val);
			tmp = tmp.next;
			cnt++;
		}
		System.out.println();*/
		
		/*ArrayList<Character> ch = new ArrayList<Character>();
		ch.add('a'); 
		ch.add('b');
		ch.add('c');System.out.println(ch);
		//ch.remove(0);System.out.println(ch);
		System.out.println(ch.contains('b'));
		ArrayList<Character> cpy = (ArrayList)ch.clone();
		System.out.println(cpy);
		System.out.println(cpy.get(0));
		cpy.set(1, 'f'); System.out.println(cpy);System.out.println(ch);*/
		/*Map<Integer, String> map = new TreeMap<>();
		map.put(5, "a5");
		map.put(2, "a2");
		map.put(1, "a1");
		
		Iterator<String> iter = map.values().iterator();
		while (iter.hasNext()){
			
			System.out.println(iter.next());
		}
		
		for(Entry<Integer, String> entry : map.entrySet()){
			System.out.println(entry.getKey());
		}
		
		
		int i = 1<<2;
		System.out.println(i);*/
		/*final List<Integer> list = new ArrayList<Integer>();
		Collections.addAll(list, 1,5,2,3,7,3,8,9);
		final Integer pos = Integer.valueOf(3);
		list.remove(pos);
		System.out.println(list);*/
		/*Map<Integer, String> hashMap = new HashMap<Integer, String>(5);
		hashMap.put(1, "apple");
		hashMap.put(2, null);
		hashMap.put(new Integer(3), "peach");
		hashMap.put(3, "orange");
		hashMap.put(4, "peach");
		for(String v : hashMap.values()){
			if("orange".equals(v)){
				hashMap.put(5,"banana");
			}
		}
		System.out.println(hashMap.get(3));*/
		
	
		
	}

}
