#include <iostream>
#include <map>
#include <utility>

using namespace std;

struct A {
    int x;

    void set_a(unsigned a) { x |= a << 0; }
    void set_b(unsigned b) { x |= b << 1; }
    void set_c(unsigned c) { x |= c << 3; }
    void set_d(unsigned d) { x |= d << 7; }
    void set_e(unsigned e) { x |= e << 15; }

    unsigned a() const { return (x >> 0)  & 1; }
    unsigned b() const { return (x >> 1)  & 3; }
    unsigned c() const { return (x >> 3)  & 15; }
    unsigned d() const { return (x >> 7)  & 255; }
    unsigned e() const { return (x >> 15) & 65535; }
};

union B {
    int x;

    struct {
        unsigned a: 1;
        unsigned b: 2;
        unsigned c: 4;
        unsigned d: 8;
        unsigned e: 16;
        unsigned pad: 1;
    };
};

/*int processA (int n) {
    int k = 0;
    for (int i = 0; i < n; i++) {
        A x;
        x.set_a(rand() & 1);
        x.set_b(2);
        x.set_c(rand() & 5);
        x.set_d(111);
        x.set_e(rand() & 65535);
        k ^= x.x;
    }
    return k;
}

int processB (int n) {
    int k = 0;
    for (int i = 0; i < n; i++) {
        B x;
        x.x = rand();
        //x.a = rand() & 1;
        //x.b = 2;
        //x.c = rand() & 5;
        //x.d = 111;
        //x.e = rand() & 65535;
        if (x.d || x.b) k ^= x.x;
    }
    return k;
}*/

int main() {
    /*int n = 1000000;
    cerr << sizeof(A) << " " << sizeof(B) << endl;
    
    clock_t s = clock();
    //unsigned x = processA(n);
    unsigned x = processB(n);
    clock_t e = clock();
    cerr << x << endl;
    cerr << (e - s) << endl;
*/
    B b;
    b.x = rand();
    signed c = b.c;
    cerr << (c * 2) << endl;

    return 0;
}