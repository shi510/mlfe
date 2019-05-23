#ifndef __MLFE_OP_H__
#define __MLFE_OP_H__
namespace mlfe{
namespace detail{

class op{
public:
    op() = default;

    virtual ~op() = default;

    virtual void run() = 0;
};

} // end namespace detail
} // end namespace mlfe

#endif // end #ifndef __MLFE_OP_H__