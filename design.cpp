/// author: kingil
/// theme: interview prepare
/// time: 2019-03

///
///简单工厂
///工厂类 Factory
///抽象产品 Product
///具体产品ProductA
class Product
{
    Product();
    virtual ~Product();
    virtual void use() = 0;
}
Product::Product(){}
Product::~Product(){}
///////
class ProA:Product
{
    ProA();
    virtual ~ProA();
    virtual void use();
}
ProA::ProA(){}
ProA::~ProA(){}
ProA::use()
{
    std::cout<<"use A..."<<endl;
}

class ProB:Product
{
    ProB();
    virtual ~ProB();
    virtual void use();
}
ProB::ProB(){}
ProB::~ProB(){}
ProB::use()
{
    std::cout<<"use B..."<<endl;
}

///////
class Factory
{
    Factory();
    virtual ~Factory();
    static Product * createProduct(string name);
}
Factory::Factory(){}
Factory::~Factory(){}
Product * Factory::createProduct(string name)
{
    if(name == 'A')
        return new ProA();
    if(name == 'B')
        return new ProB();
    else return null;
}

///单例：
class singleton
{
    private static singleton _instance = null;
    private singleton();
    ~singleton();
    public static singleton getinstance()
    {
        if(_instance == null)
            _instance = new singleton();
        return _instance;
    }
}
///中介模式
///抽象中介、具体中介，抽象角色、具体角色

///代理模式
///subject
///realsubject
///proxy 代理角色
class subject
{
    subject();
    ~subject();
    virtual void request();
}
subject::subject(){}
subject::~subject(){}
subject::request(){}
///
class proxy:subject
{
public:
    proxy();
    ~proxy();
    request();
private:
    void prerequest();
    void afterrequest();
    realsubject * _realsubject;
}
proxy::proxy()
{
    _realsubject = new realsubject();
}
proxy::~proxy()
{
    delete _realsubject;
}
proxy::prerequest()
{
    std::cout<<"this is pre request..."<<endl;
}
proxy::afterrequest()
{
    std::cout<<"this is after request..."<<endl;
}
proxy::request()
{
    prerequest();
    _realsubject->realrequest();
    afterrequest();
}
///
class realsubject()
{
    real subject();
    ~realsubject();
    realrequest();
}
realsubject::realrequest(){}
realsubject::~realsubject(){}
realsubject::realrequest()
{
    std::cout<<"this is real request function..."<<endl;
}

int main ()
{
    proxy myproxy();
    myproxy.request();
    return 0;
}


