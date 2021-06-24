#include <iostream>
#include <string>
#include <map>

typedef int(*BrewFunction)();//函数指针类型变量
typedef std::map<std::string, BrewFunction> BrewMap;
BrewMap g_brew_map;//全局变量


// 定义Resgister 类更新 g_brew_map 字典
#define RegisterBrewFunction(func) \
namespace {\
class __Registerer_##func{ \
public: /* NOLINT */ \
	__Registerer_##func() {\
	g_brew_map[#func] = &func; \
	} \
}; \
__Registerer_##func g_registerer_##func; \
}


//返回函数指针
static BrewFunction GetBrewFunction(const std::string& name) {
	if (g_brew_map.count(name)) {//查找元素
		return g_brew_map[name];
	}
	else {
	std::cout << "Available caffe actions:";
		for (BrewMap::iterator it = g_brew_map.begin();
			it != g_brew_map.end(); ++it) {
			std::cout << "\t" << it->first;
		}
		std::cout << "Unknown action: " << name;
		return NULL;  // not reachable, just to suppress old compiler warnings.
	}
}

int train(){
	//add code here
	std::cout << "Training Net Func" << std::endl;
	return 0;
}
RegisterBrewFunction(train);

int test(){
	//add code here
	std::cout << "Testing Net Func" << std::endl;
	return 0;
}
RegisterBrewFunction(test);

int main(int argc, char** argv)
{
	argv[1] = "train";
	//argv[1]是输入的函数名字，根据这个名词，执行不同的函数。
	int ret = GetBrewFunction(std::string(argv[1]))(); //()表示函数调用，函数形参为空。
	return 0;
}
