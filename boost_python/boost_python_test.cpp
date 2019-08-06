#include <iostream>
#include <string>
#include <fstream>
#include <streambuf>
#include <boost/python.hpp>
#include <string>


int main() {
    Py_Initialize();

    auto main_ns = boost::python::import("__main__").attr("__dict__");

    try {
        std::ifstream ifs("python_code.py");
        std::string script((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());
        boost::python::exec(script.c_str(), main_ns);
        boost::python::object func = main_ns["func"];
        boost::python::object result = func(1.0);
        std::cout << boost::python::extract<double>(result) << std::endl;
    }
    catch (boost::python::error_already_set) {
        PyErr_Print();
    }
}
