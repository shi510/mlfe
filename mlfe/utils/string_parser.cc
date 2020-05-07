#include "mlfe/utils/string_parser.h"

namespace mlfe{

string_parser::string_parser(std::string str) : __str(str){ }

string_parser& string_parser::split(std::string delim)
{
	size_t prev = 0;
	auto cur = __str.find(delim);
	while (cur != std::string::npos)
	{
		auto sub = __str.substr(prev, cur - prev);
		__found_items.push_back(sub);
		prev = cur + 1;
		cur = __str.find(',',prev); 
	}
	__found_items.push_back(__str.substr(prev, cur - prev));
	return *this;
}

string_parser& string_parser::remove(std::string delim)
{
	size_t pos = 0;
	while ((pos = __str.find(delim)) != std::string::npos)
	{
		__found_items.push_back(__str.erase(pos, delim.size()));
	}
	return *this;
}

size_t string_parser::size() const
{
	return __found_items.size();
}

std::string string_parser::item(int idx) const
{
	if(__found_items.empty())
	{
		return "";
	}
	else if (idx >= (int)__found_items.size())
	{
		throw std::string("Out of range at found items in string_parser, ") +
			"you accessed index " +
			std::to_string(idx) +
			" but the items size is " +
			std::to_string(__found_items.size()) + ".";
	}
	else if(idx < 0)
	{
		return *std::prev(__found_items.end());
	}
	return __found_items[idx];
}


} // end namespace mlfe;