#include <iostream>
#include <bitset>

using namespace std;

template <size_t N>
void out() {
    cerr << "sizeof(bitset<" << N << ">)" << " = " << sizeof(bitset<N>) << endl;
}

int main() {
    out<1>();
    out<2>();
    out<4>();
    out<8>();
    out<16>();
    out<32>();
    out<64>();
    out<96>();
    out<128>();
    return 0;
}