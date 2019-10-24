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
    // abstract class
    virtual ~Test() {}
    virtual int test() const = 0;
};

struct Test1 : public Test
{
    vector<int> a;
    Test1()
    {
        a.resize(10);
        for (int i = 0; i < 10; i++) a[i] = i;
    }
    vector<int> get() const
    {
        return a;
    }
    Test1 copy() const
    {
        Test1 test;
        test.a = a;
        return test;
    }
    void up()
    {
        for (int i = 0; i < 10; i++) a[i]++;
    }
    int test() const override
    {
        return 1;
    }
};

// 継承 https://pybind11.readthedocs.io/en/stable/advanced/classes.html

struct PyTest : public Test
{
    using Test::Test;
    int test() const override
    {
        PYBIND11_OVERLOAD_PURE(int, Test, test);
    }
};

struct Test2 : public Test
{
    int test() const override
    {
        return 2;
    }
};

#ifdef PY

namespace py = pybind11;
PYBIND11_MODULE(pbtest, m) {
    m.doc() = "pybind test";

    py::class_<Test, PyTest>(m, "Test")
    .def(pybind11::init<>(), "constructor")
    .def("test", &Test::test, "test");

    py::class_<Test1>(m, "Test1")
    .def("get", &Test1::get, "get value")
    .def("up", &Test1::up, "add")
    .def("copy", &Test1::copy, "copy")
    .def("test", &Test1::test, "test");

    py::class_<Test2>(m, "Test2")
    .def("test", &Test2::test, "test");
};

#else

int main()
{}

#endif
