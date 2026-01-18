
import re
def clean_string(string):
	"""Deletes irrelevant characters from string.

	input: 
	- String (str)

	return: 
	- string (str): the same string but without non alphabetical characters (like '!@#$%^&*()1234567890_-<>?' )
	"""
	regex = re.compile('[^a-zA-Z]')
	clean_string = regex.sub('', string)
	return clean_string


def pal_length_v2(string:str):
	"""
	Find longest palindromic substring length using expand around center method.
	- Time Complexity: O(n^2)
	- Space Complexity: O(1)

	input:
	- string (str): Input string to find palindrome in
	
	Returns:
	- int: Length of longest palindromic substring
	"""
	max_pal=''
	for i in range(len(string)):
		max_pal=max(max_pal,expand(string,i,i), expand(string,i,i+1), key=len)
	
	return len(max_pal)
            
def expand(string,i,j):
	"""
    Expand around center to find palindrome.
    
    inputs:
    - string (str): Input string
    - i (int): Left center index
    - j (int): Right center index
    
    Returns:
    - str: Longest palindromic substring around given center
    """
	while i>=0 and j<len(string) and string[i]==string[j]:
		i-=1
		j+=1
	return string[i+1:j]


def pal_length_manachers(string:str): 
	"""
    Find longest palindromic substring length using Manacher's algorithm.
    read more on this algoritem in: https://www.geeksforgeeks.org/manachers-algorithm-linear-time-longest-palindromic-substring-part-1/
    - Time Complexity: O(n)
    - pace Complexity: O(n)
    
    inputs:
    - string (str): Input string to find palindrome in
    
    Returns:
    - int: Length of longest palindromic substring
    """
	N = len(string) 
	if N == 0: 
		return 0
	if N ==1: return 1
	N = 2*N+1 # Position count 
	L = [0] * N 
	L[0],L[1] = 0,1
	mid = 1	 # centerPosition 
	R = 2	 # centerRightPosition 
	i = 0 # currentRightPosition 
	iMirror, maxLPSLength = 0,0 # currentLeftPosition 

	diff = -1
	for i in range(2,N): 
	
		# get currentLeftPosition iMirror for currentRightPosition i 
		iMirror = 2*mid-i 
		L[i] = 0
		diff = R - i 
		# If currentRightPosition i is within centerRightPosition R 
		if diff > 0: 
			L[i] = min(L[iMirror], diff) 

		# try to expand palindrome centered at currentRightPosition i.
		try: #TODO: what expation it can rise
			while ((i+L[i]) < N and (i-L[i]) > 0) and (((i+L[i]+1) % 2 == 0) or (string[(i+L[i]+1)//2] == string[(i-L[i]-1)//2])): 
				L[i]+=1
		except Exception as e: 
			pass

		if L[i] > maxLPSLength:	 # Track maxLPSLength 
			maxLPSLength = L[i] 
			# maxLPSCenterPosition = i 

		# If palindrome centered at currentRightPosition i  - we need to expend centerRightPosition R, 
		# and then adjust centerPosition mid based on expanded palindrome. 
		if i + L[i] > R: 
			mid = i 
			R = i + L[i] 

	return maxLPSLength 
