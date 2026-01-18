from pathlib import Path
import sys #TODO - packge?
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)
import dotenv
import os
dotenv.load_dotenv(os.path.join(project_root, '.env'))


from src.palindrome import clean_string, pal_length_v2, pal_length_manachers, expand
#test the clean_string function
def test_clean_string():
    assert clean_string("hello123!") == "hello"
    assert clean_string("@#$%^&*") == ""
    assert clean_string("ABCabc") == "ABCabc"
    assert clean_string("1234567890") == ""
    assert clean_string("") == ""
    assert clean_string("AB!C ab><c") == "ABCabc"
#test the expand help-function
def test_expand():
    assert expand("racecar", 3, 3) == "racecar"  # Odd-length palindrome
    assert expand("abba", 1, 2) == "abba"  # Even-length palindrome
    assert expand("abcd", 1, 2) == ""  # No palindrome
    assert expand("a", 0, 0) == "a"  # Single character
    assert expand("aa", 0, 1) == "aa"  # Small even-length palindrome
# test test_pal_length_v2 
def test_pal_length_v2():
    assert pal_length_v2("babad") == 3  # Two palindromes
    assert pal_length_v2("abba") == 4  # Entire string is a palindrome
    assert pal_length_v2("abcdef") == 1  # Single character palindrome
    assert pal_length_v2("a") == 1  # Single character
    assert pal_length_v2("") == 0  # Empty string
    assert pal_length_v2("cbbd") == 2 #The example from the assignment

# test test_pal_length_manachers 
def test_pal_length_manachers():
    assert pal_length_manachers("babad") == 3  # Two palindromes
    assert pal_length_manachers("abba") == 4  # Entire string is a palindrome
    assert pal_length_manachers("abcdef") == 1  # Single character palindrome
    assert pal_length_manachers("a") == 1  # Single character
    assert pal_length_manachers("") == 0  # Empty string
    assert pal_length_manachers("cbbd") == 2 #The example from the assignment

