#include <iostream>

using namespace std;

struct X
{
    X() {
        cerr << "X constructor" << endl;
    }
    X(int k) {
       cerr << "X constructor with argument " << k << endl;
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
    Y(int k): X(k) {
        cerr << "Y constructor with augument " << k << endl;
    }
    virtual int value() const {
        return 1;
    }
};



int main() {
    X* x = new X();
    X* y = new Y();
    X* xx = new X(1);
    X* yy = new Y(2);

    cerr << x->value() << endl;
    cerr << y->value() << endl;

    Y ay;
    X* px = &ay;
    cerr << px->value() << endl;
}
