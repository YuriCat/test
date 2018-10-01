#include <iostream>
#include <sstream>
#include <cassert>
#include <vector>
#include <set>
#include <string>
#include <random>

#ifdef PY
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif

using namespace std;

struct Test
{
    vector<int> a;
    Test()
    {
        a.resize(10);
        for (int i = 0; i < 10; i++) a[i] = i; 
    }
    vector<int> get() const
    {
        return a; 
    }
    Test copy() const
    {
        Test test;
        test.a = a;
        return test;
    }
    void up()
    {
        for (int i = 0; i < 10; i++) a[i]++; 
    }
};


#ifdef PY

namespace py = pybind11;
PYBIND11_MODULE(pbtest, m) {
    m.doc() = "pybind test";

    py::class_<Test>(m, "Test")
    .def(pybind11::init<>(), "constructor")
    .def("get", &Test::get, "get value")
    .def("up", &Test::up, "add")
    .def("copy", &Test::copy, "copy");
};

#else

int main()
{}

#endif
