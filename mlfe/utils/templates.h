#pragma once

namespace mlfe{
namespace util{

template <typename _Base>
struct template_unpacker {
	template <typename _First, typename ..._Last>
	template_unpacker(_First f, _Last... l)
	{
		params.push_back(std::make_unique<_First>(f));
		iter<_Last...>(l...);
	}

	template <typename _First, typename ..._Last>
	void iter(_First f, _Last... l)
	{
		params.push_back(std::make_unique<_First>(f));
		iter<_Last...>(l...);
	}

	template <typename _Last>
	void iter(_Last l)
	{
		params.push_back(std::make_unique<_Last>(l));
	}
	std::vector<std::unique_ptr<_Base>> params;
};

} // end namespace util
} // end namespace mlfe