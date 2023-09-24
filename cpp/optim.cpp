
#include <iostream>
#include <utility>

union A {
    struct {
        unsigned a:32;
        unsigned b:32;
    };
    unsigned long long c;
};

int main()
{
    int x = 0;
    unsigned a = clock();

    //if (a & 12) std::cout << "";
    // if (bool(a & 8) || bool(a & 4)) std::cout << "";
    //if (bool((a >> 3) & 1) || bool((a >> 2) & 1)) std::cout << "";

    std::cout << sizeof(A);

    A s;
    s.a = clock();
    s.b = clock();
    s.b ^= 365;
    x = bool(s.a & 1) || bool(s.b & 1);

    s.c = clock();
    x = bool(s.c & ((1ULL << 32) | 1ULL));

    return x;
}

