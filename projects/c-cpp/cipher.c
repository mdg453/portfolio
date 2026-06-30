#include "cipher.h"
#include <string.h>
#include <stdio.h>
#define ALPHNUM 26
#define BUTTOM_LOWER 96
#define TOP_LOWER 123
#define BOTTOM_UPPER 64
#define TOP_UPPER 91

/// IN THIS FILE, IMPLEMENT EVERY FUNCTION THAT'S DECLARED IN cipher.h.

// See full documentation in header file
void encode (char s[], int k)
{
    int i = 0 ;
    while (s[i] != '\0'){
        int a = s[i];
        if (a > BUTTOM_LOWER && a < TOP_LOWER){
            a = a + k ;
            while (a >= TOP_LOWER){
                a = a -ALPHNUM ;
            }
            while (a <= BUTTOM_LOWER) {
                a = a + ALPHNUM;
            }
        }
        else if (a < TOP_UPPER && a > BOTTOM_UPPER) {
            a = a + k;
            while (a >= TOP_UPPER) {
                a = a - ALPHNUM;
            }
            while (a <= BOTTOM_UPPER) {
                a = a + ALPHNUM;
            }
        }
    char b = a;
    s[i] = b ;
    i++ ;
    }
}

// See full documentation in header file
void decode (char s[], int k)
{
    int i = 0 ;
    while (s[i] != '\0'){
        int a = s[i];
        if (a > BUTTOM_LOWER && a < TOP_LOWER){
            a = a - k ;
            while (a >= TOP_LOWER){
                a = a -ALPHNUM ;
            }
            while (a <= BUTTOM_LOWER) {
                a = a + ALPHNUM;
            }
        }
        else if (a < TOP_UPPER && a > BOTTOM_UPPER) {
            a = a - k;
            while (a >= TOP_UPPER) {
                a = a - ALPHNUM;
            }
            while (a <= BOTTOM_UPPER) {
                a = a + ALPHNUM;
            }
        }
        char b = a;
        s[i] = b ;
        i++ ;
    }
}
