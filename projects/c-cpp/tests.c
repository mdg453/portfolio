#include "tests.h"
#include <string.h>
#include <stdio.h>
#define K_1 3

// See full documentation in header file
int test_encode_non_cyclic_lower_case_positive_k ()
{
  char in[] = "abc";
  char out[] = "def";
  encode (in, K_1);
  return strcmp (in, out) != 0;
}

// See full documentation in header file
int test_encode_cyclic_lower_case_special_char_positive_k ()
{
  char in[] = "xyz" ;
  char out[] = "abc";
  encode (in, K_1);
  return strcmp (in, out) != 0;
}

// See full documentation in header file
int test_encode_non_cyclic_lower_case_special_char_negative_k ()
{
    char in[] = "xyz" ;
    char out[] = "uvw";
    encode (in, -K_1);
    return strcmp (in, out) != 0;
}

// See full documentation in header file
int test_encode_cyclic_lower_case_negative_k ()
{
    char in[] = "abc" ;
    char out[] = "xyz";
    encode (in, -K_1);
    return strcmp (in, out) != 0;
}

// See full documentation in header file
int test_encode_cyclic_upper_case_positive_k ()
{
    char in[] = "XYZ";
    char out[] = "ABC";
    encode (in, K_1);
    return strcmp (in, out) != 0;
}

// See full documentation in header file
int test_decode_non_cyclic_lower_case_positive_k ()
{
  char in[] = "def";
  char out[] = "abc";
  decode (in, K_1);
  return strcmp (in, out) != 0;
}

// See full documentation in header file
int test_decode_cyclic_lower_case_special_char_positive_k ()
{
    char in[] = "abc";
    char out[] = "xyz";
    decode (in, K_1);
    return strcmp (in, out) != 0;
}

// See full documentation in header file
int test_decode_non_cyclic_lower_case_special_char_negative_k ()
{
    char in[] = "abc";
    char out[] = "def";
    decode (in, -K_1);
    return strcmp (in, out) != 0;
}

// See full documentation in header file
int test_decode_cyclic_lower_case_negative_k ()
{
    char in[] = "xyz";
    char out[] = "abc";
    decode (in, -K_1);
    return strcmp (in, out) != 0;
}

// See full documentation in header file
int test_decode_cyclic_upper_case_positive_k ()
{
    char in[] = "XYZ";
    char out[] = "ABC";
    decode (in, -K_1);
    return strcmp (in, out) != 0;
}
