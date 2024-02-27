#include <iostream>
using namespace std;
class Base {
public:
    virtual void publicVirtualFunction() {
        // 虚函数的公有实现
         cout << "publicVirtualFunction in base " << endl;
    }

protected:
    virtual void protectedVirtualFunction() {
        // 虚函数的受保护实现
        cout << "protectedVirtualFunction in base " << endl;
    }

private:
    virtual void privateVirtualFunction() {
        // 虚函数的私有实现
    }
};

class Derived : public Base {
public:
    void publicVirtualFunction() override {
        // 派生类覆盖了基类的公有虚函数
         cout << "publicVirtualFunction in derived " << endl;
    }

protected:
    void protectedVirtualFunction() override {
        // 派生类覆盖了基类的受保护虚函数
        cout << "protectedVirtualFunction in derived " << endl;
    }

private:
    void privateVirtualFunction() override {
        // 派生类无法访问基类的私有虚函数
    }
};

int main() {
    Base* basePtr = new Derived;
    basePtr->publicVirtualFunction();  // 通过基类指针调用公有虚函数
    basePtr->protectedVirtualFunction();

    return 0;
}