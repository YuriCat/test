#include <iostream>

using namespace std;

struct X
{
    X() {
        cerr << "X constructor" << endl;
    }
    virtual int value() const {
        return 0;
    }
};

struct Y : public X
{
    Y() {
        cerr << "Y construcrtor" << endl;
    }
    virtual int value() const {
        return 1;
    }
};



int main() {
    X* x = new X();
    X* y = new Y();

    cerr << x->value() << endl;
    cerr << y->value() << endl;
}