#include <iostream>
#include <vector>
#include <random>
#include <utility>
#include <algorithm>

using namespace std;

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}
template <class T>
ostream& operator <<(ostream& ost, const vector<T>& src) {
    ost << "{";
    size_t cnt = 0;
    for (T v : src) {
        ost << v;
        if (++cnt < src.size()) ost << ", ";
    }
    ost << "}";
    return ost;
}

int main()
{
    P2 p2(50, 25, 9);
    mt19937 mt(0);
    uniform_real_distribution<double> rd(0, 100);
    
    for (int i = 0; i < 100; i++)
    {
        double r = rd(mt);
        cerr << r << " ->" << p2.insert(r) << endl;
        p2.show();
    }
}
