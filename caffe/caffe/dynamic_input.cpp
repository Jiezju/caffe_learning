#include <iostream>
#include <string>
#include <map>

typedef int(*BrewFunction)();//����ָ�����ͱ���
typedef std::map<std::string, BrewFunction> BrewMap;
BrewMap g_brew_map;//ȫ�ֱ���


// ����Resgister ����� g_brew_map �ֵ�
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


//���غ���ָ��
static BrewFunction GetBrewFunction(const std::string& name) {
	if (g_brew_map.count(name)) {//����Ԫ��
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
	//argv[1]������ĺ������֣�����������ʣ�ִ�в�ͬ�ĺ�����
	int ret = GetBrewFunction(std::string(argv[1]))(); //()��ʾ�������ã������β�Ϊ�ա�
	return 0;
}
