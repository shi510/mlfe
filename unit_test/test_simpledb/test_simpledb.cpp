#include <iostream>
#include <mlfe/device_context/cpu_context.cpp>
#include <mlfe/utils/db/simple_db.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace mlfe;

TEST(SimpleDB_IO_Test, VerifyCPUResults) {
    shared_ptr<DataBase> sdb = make_shared<simpledb::SimpleDB>();
    try{
        sdb->Open("testdb.simpledb");
        EXPECT_EQ(sdb->IsOpen(), true);
        sdb->Put("key1", "My key is a key1.");
        sdb->Put("key3", "My key is a key3.");
        sdb->Put("key5", "My key is a key5.");
        sdb->Close();
    }
    catch(std::string &e){
        std::cout<<"Error occurs : "<<e<<std::endl;
        EXPECT_TRUE(false);
        return;
    }
    try{
        sdb->Open("testdb.simpledb");
        EXPECT_EQ(sdb->IsOpen(), true);
        sdb->Put("key0", "My key is a key0.");
        sdb->Put("key2", "My key is a key2.");
        sdb->Close();
    }
    catch(std::string &e){
        std::cout<<"Error occurs : "<<e<<std::endl;
        EXPECT_TRUE(false);
        return;
    }
    
    try{
        sdb->Open("testdb.simpledb");
        EXPECT_EQ(sdb->IsOpen(), true);
        sdb->Put("key0", "My key is a key0.");
    }
    catch(std::string &e){
        std::cout<<"Error must occur : ";
        std::cout<<e<<std::endl;
        sdb->Close();
    }
    std::string val;
    sdb->Open("testdb.simpledb");
    EXPECT_EQ(sdb->IsOpen(), true);
    sdb->Get("key1", val);
    EXPECT_EQ(val.compare("My key is a key1."), 0);
    sdb->Get("key3", val);
    EXPECT_EQ(val.compare("My key is a key3."), 0);
    sdb->Get("key5", val);
    EXPECT_EQ(val.compare("My key is a key5."), 0);
    sdb->Get("key0", val);
    EXPECT_EQ(val.compare("My key is a key0."), 0);
    sdb->Get("key2", val);
    EXPECT_EQ(val.compare("My key is a key2."), 0);
    sdb->Close();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
