/// author: kingil
/// theme: interview prepare
/// time: 2019-03
#include <iostream>
#include <string>
#include <unordered_map>
#include <stdio.h>
#include <map>
using namespace std;

//两个类互相引用的时候，必须要申明。

//简单工厂模式---------------------------------------
//工厂类 SimpleFactory
//抽象产品 SimpleProduct
//具体产品 SimpleProductA、SimpleProductB
//class SimpleProduct
class SimpleProduct
{
public:
	SimpleProduct();
	virtual ~SimpleProduct();
	virtual void use();
};
SimpleProduct::SimpleProduct() {}
SimpleProduct::~SimpleProduct() {}
void SimpleProduct::use() {}
//class simpleproductA
class simpleproductA :public SimpleProduct
{
public:
	simpleproductA();
	virtual ~simpleproductA();
	virtual void use();
};
simpleproductA::simpleproductA(){}
simpleproductA::~simpleproductA() {}
void simpleproductA::use()
{
	cout << "use AAA" << endl;
}
class simpleproductB :public SimpleProduct
{
public:
	simpleproductB();
	virtual ~simpleproductB();
	virtual void use();
};
simpleproductB::simpleproductB() {}
simpleproductB::~simpleproductB() {}
void simpleproductB::use()
{
	cout << "use BBB" << endl;
}
//class SimpleFactory
class SimpleFactory
{
public:
	SimpleFactory();
	virtual ~ SimpleFactory();
	static SimpleProduct * createProduct(string type);
};
SimpleFactory::SimpleFactory() {}
SimpleFactory::~SimpleFactory() {}
SimpleProduct * SimpleFactory::createProduct(string type)
{
	if (type == "A")
	{
		return new simpleproductA();
	}
	else if (type == "B")
	{
		return new simpleproductB();
	}
	else 
		return NULL;
}

//抽象工厂模式---------------------------------------------
//抽象工厂 Factory
//具体工厂 ConFactory1(A1B1)/ConFactory2(A2B2)
//抽象产品 ProductA/ProductB
//具体产品 A1/A2/B1/B2
class Factory;
class ProductA;
class ProductB;
//class ProductA
class ProductA
{
public:
	ProductA();
	virtual ~ProductA();
	virtual void use();
};
ProductA::ProductA() {};
ProductA::~ProductA() {};
void ProductA::use() {};
//class conProduct
class ProductA1:public ProductA
{
public:
	ProductA1();
	virtual ~ProductA1();
	virtual void use();
};
ProductA1::ProductA1() {};
ProductA1::~ProductA1() {};
void ProductA1::use()
{
	cout << "use AAAA11111" << endl;
}
class ProductA2:public ProductA
{
public:
	ProductA2();
	virtual ~ProductA2();
	virtual void use();
};
ProductA2::ProductA2() {};
ProductA2::~ProductA2() {};
void ProductA2::use()
{
	cout << "use AAAA22222" << endl;
}
//class ProductB
class ProductB
{
public:
	ProductB();
	virtual ~ProductB();
	virtual void use();
};
ProductB::ProductB() {};
ProductB::~ProductB() {};
void ProductB::use() {};
//class conProduct
class ProductB1 :public ProductB
{
public:
	ProductB1();
	virtual ~ProductB1();
	virtual void use();
};
ProductB1::ProductB1() {};
ProductB1::~ProductB1() {};
void ProductB1::use()
{
	cout << "useBBBBB11111" << endl;
}
class ProductB2 :public ProductB
{
public:
	ProductB2();
	virtual ~ProductB2();
	virtual void use();
};
ProductB2::ProductB2() {};
ProductB2::~ProductB2() {};
void ProductB2::use()
{
	cout << "use BBBB22222" << endl;
}
//class Facory
class Factory
{
public:
	Factory();
	virtual ~Factory();
	virtual ProductA * createA()=0;
	virtual ProductB * createB()=0;
};
Factory::Factory() {}
Factory::~Factory() {}
//class ConFactory1
class ConFactory1 :public Factory
{
public:
	ConFactory1();
	virtual ~ConFactory1();
	virtual ProductA * createA();
	virtual ProductB * createB();

};
ConFactory1::ConFactory1() {};
ConFactory1::~ConFactory1() {};
ProductA * ConFactory1::createA()
{
	return new ProductA1();
}
ProductB * ConFactory1::createB()
{
	return new ProductB1();
}
//class ConFactory2
class ConFactory2 :public Factory
{
public:
	ConFactory2();
	virtual ~ConFactory2();
	virtual ProductA * createA();
	virtual ProductB * createB();

};
ConFactory2::ConFactory2() {};
ConFactory2::~ConFactory2() {};
ProductA * ConFactory2::createA()
{
	return new ProductA2();
}
ProductB * ConFactory2::createB()
{
	return new ProductB2();
}

//中介模式-----------------------------------------------
//抽象成员 member
//具体成员 conmember
//抽象中介 med
//具体中介 conmed

//member
class med;
class member
{
public:
	member();
	virtual ~member();
	void setmed(med * themed);
	virtual void sendmsg(int towho, string msg);
	virtual void receivemsg(string msg);
protected:
	med * _themed;
};
//实现member
member::member() {}
member::~member() {}
void member::setmed(med * themed)
{
	_themed = themed;
}
void member::sendmsg(int towho, string msg)
{

}
void member::receivemsg(string msg)
{
	cout << "receive msg is ..." << msg << endl;
}
//med
class med
{
public:
	med();
	virtual ~med();
	virtual void operation(int who, std::string msg);
	virtual void reg(int who, member* themember);
};
//实现med
med::med() {}
med::~med() {}
void med::operation(int who, std::string msg) {}
void med::reg(int who, member* themember) {}
//conmed
class conmed :public med
{
public:
	conmed();
	virtual ~conmed();
	virtual void operation(int who, std::string str);
	virtual void reg(int who, member* themember);
private:
	map<int, member*> _membermap;
};
//实现conmed
conmed::conmed() {}
conmed::~conmed() {}
void conmed::operation(int who, std::string msg)
{
	if (_membermap.find(who) == _membermap.end())
	{
		cout << "not found this member" << endl;
		return;
	}
	member * resultwho = _membermap.find(who)->second;
	resultwho->receivemsg(msg);
}
void conmed::reg(int who, member* themember)
{
	if (_membermap.find(who) == _membermap.end())
	{
		_membermap.insert(make_pair(who, themember));
		themember->setmed(this);
	}
}
//conmember
class conmemberA :public member
{
public:
	conmemberA();
	virtual ~conmemberA();
	virtual void sendmsg(int towho, string msg);
	virtual void receivemsg(string msg);
};
class conmemberB :public member
{
public:
	conmemberB();
	virtual ~conmemberB();
	virtual void sendmsg(int towho, string msg);
	virtual void receivemsg(string msg);
};
//实现conmemberA
conmemberA::conmemberA() {}
conmemberA::~conmemberA() {}
void conmemberA::sendmsg(int towho, string msg)
{
	cout << "sned msg from a to ..." << towho << endl;
	_themed->operation(towho, msg);
}
void conmemberA::receivemsg(string msg)
{
	cout << "A receive msg:.." << msg << endl;
}
conmemberB::conmemberB() {}
conmemberB::~conmemberB() {}
void conmemberB::sendmsg(int towho, string msg)
{
	cout << "sned msg from a to ..." << towho << endl;
	_themed->operation(towho, msg);
}
void conmemberB::receivemsg(string msg)
{
	cout << "B receive msg:.." << msg << endl;
}
///观察者模式

int main()
{
	//中介模式
	//cout << "-----" << endl;
	//conmemberA * pa = new conmemberA();
	//conmemberB * pb = new conmemberB();
	//conmed * manager = new conmed();
	//manager->reg(1,pa);
	//manager->reg(2,pb);
	//pa->sendmsg(2,"i am a...");
	//pb->sendmsg(1,"i am b...");
	//delete pa, pb, manager;
	//抽象工厂模式
/*	cout << "-----" << endl;
	Factory * f1 = new ConFactory1();
	ProductA * pa = f1->createA();
	ProductB * pb = f1->createB();
	pa->use();
	pb->use();
	Factory * f2 = new ConFactory2();
	ProductA * qa = f2->createA();
	ProductB * qb = f2->createB();
	qa->use();
	qb->use();
	delete f1, f2, pa, pb, qa, qb*/;
	//简单工厂模式
	//cout << "-----" << endl;
	//SimpleProduct * prod = SimpleFactory::createProduct("A");//申明为静态，只有一个工厂生产不同产品
	//prod->use();
	//delete prod;

	getchar();
	return 0;
}