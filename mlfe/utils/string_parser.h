#pragma once
#include <vector>
#include <string>

namespace mlfe{

class string_parser{
public:
	string_parser(std::string str);

	string_parser& split(std::string delim);

	string_parser& remove(std::string delim);

	size_t size() const;

	std::string item(int idx) const;

private:
	std::string __str;
	std::string __resent_delim;
	std::vector<std::string> __found_items;
};

} // end namespace mlfe
