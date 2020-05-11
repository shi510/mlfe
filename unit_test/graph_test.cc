#include <gtest/gtest.h>
#include <mlfe/core/graph.h>

namespace graph_test{
using namespace mlfe;

/*
topology1:
    n1---n4---n5
         /   /
    n2--/   /
           /
    n3----/
 */
TEST(graph_test, topology1)
{
	std::string results;
	node n1, n2, n3, n4, n5;
	auto check_for = [&results](node& n, std::string answer)
	{
		results.clear();
		n.run();
		EXPECT_EQ(results, answer);
	};
	n1.set_task(make_task([&results](){results += "1";}));
	n2.set_task(make_task([&results](){results += "2";}));
	n3.set_task(make_task([&results](){results += "3";}));
	n4.set_task(make_task([&results](){results += "4";}));
	n5.set_task(make_task([&results](){results += "5";}));

	n4.add_input(n1);
	n4.add_input(n2);
	n5.add_input(n4);
	n5.add_input(n3);

	check_for(n5, "12435");
	check_for(n4, "124");
	check_for(n3, "3");
	check_for(n2, "2");
	check_for(n1, "1");
}

/*
topology2:
        n1---n5---n7
             /   /
        n2--/   /
               /
    n0--n3----/
          \
          n6---n8
          /
        n4
 */
TEST(graph_test, topology2)
{
	std::string results;
	node n0, n1, n2, n3, n4, n5, n6, n7, n8;
	auto check_for = [&results](node& n, std::string answer)
	{
		results.clear();
		n.run();
		EXPECT_EQ(results, answer);
	};
	n0.set_task(make_task([&results](){results += "0";}));
	n1.set_task(make_task([&results](){results += "1";}));
	n2.set_task(make_task([&results](){results += "2";}));
	n3.set_task(make_task([&results](){results += "3";}));
	n4.set_task(make_task([&results](){results += "4";}));
	n5.set_task(make_task([&results](){results += "5";}));
	n6.set_task(make_task([&results](){results += "6";}));
	n7.set_task(make_task([&results](){results += "7";}));
	n8.set_task(make_task([&results](){results += "8";}));

	n3.add_input(n0);

	n5.add_input(n1);
	n5.add_input(n2);

	n6.add_input(n3);
	n6.add_input(n4);

	n7.add_input(n5);
	n7.add_input(n3);

	n8.add_input(n6);

	check_for(n7, "125037");
	check_for(n8, "03468");
	check_for(n6, "0346");
	check_for(n5, "125");
	check_for(n4, "4");
	check_for(n3, "03");
	check_for(n2, "2");
	check_for(n1, "1");
	check_for(n0, "0");
}

/*
topology3:
    n1---n2---n3
    \_________/
 */
TEST(graph_test, topology3)
{
	std::string results;
	node n1, n2, n3;
	auto check_for = [&results](node& n, std::string answer)
	{
		results.clear();
		n.run();
		EXPECT_EQ(results, answer);
	};
	n1.set_task(make_task([&results](){results += "1";}));
	n2.set_task(make_task([&results](){results += "2";}));
	n3.set_task(make_task([&results](){results += "3";}));
	n2.add_input(n1);

	n3.add_input(n2);
	n3.add_input(n1);

	check_for(n3, "1213");
	check_for(n2, "12");
	check_for(n1, "1");
}

} // end namespace graph_test