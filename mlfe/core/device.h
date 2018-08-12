#ifndef __DEVICE_HPP__
#define __DEVICE_HPP__
#include "../utils/types.h"
#include <memory>

namespace mlfe{

class Device final{
template <class Base, class Der>
using IsBase = std::is_base_of<Base, Der>;
template <class Base, class Der>
using EnableWithBase = std::enable_if<IsBase<Base, Der>::value>;
private:
    struct DeviceType{};
public:
    template <class Dev>
    class Select;

    struct CUDA final : DeviceType{
        static bool enable_cudnn;
        static const char *string;
        static const char *string_cudnn;
    };

    struct CPU final : DeviceType{
        static const char *string;
    };

    template <class Dev>
    Device(Select<Dev> s);

    Device() = default;

    type::uint8::T *Data() const;

    template <class T>
    T *Data() const;

    type::uint32::T Size() const;

    void Allocate(type::uint32::T size);

    std::string Name() const;

    template <class From, class To,
        typename = typename EnableWithBase<DeviceType, From>::type,
        typename = typename EnableWithBase<DeviceType, To>::type
    >
    static void Copy(const Device from, Device to){
        CopyInternal<From, To>(from, to);
    }

private:
    template <class From, class To>
    static void CopyInternal(const Device from, Device to);

    class SelectedDevice;
    std::shared_ptr<SelectedDevice> sd;
};

class Device::SelectedDevice{
public:
    virtual void Allocate(type::uint32::T size) = 0;

    virtual type::uint8::T *Data() const = 0;

    virtual type::uint32::T Size() const = 0;

    virtual std::string Name() const = 0;
};

template <class Dev>
class Device::Select final : public Device::SelectedDevice{
public:
    Select();

    Select(type::uint8::T *ptr, type::uint32::T size);

    void Allocate(type::uint32::T size) override;

    type::uint8::T *Data() const override;

    type::uint32::T Size() const override;

    std::string Name() const override;

private:
    struct DeviceData;
    std::shared_ptr<DeviceData> dd;
};

template <class Dev>
Device::Device(Select<Dev> s){
    sd = std::make_shared<Select<Dev>>(s);
}

template <class T>
T *Device::Data() const{
    return reinterpret_cast<T *>(sd->Data());
}

template <> void
Device::CopyInternal<Device::CPU, Device::CPU>(const Device from, Device to);

template <> void
Device::CopyInternal<Device::CPU, Device::CUDA>(const Device from, Device to);

template <> void
Device::CopyInternal<Device::CUDA, Device::CPU>(const Device from, Device to);

template <> void
Device::CopyInternal<Device::CUDA, Device::CUDA>(const Device from, Device to);

} // end namespace mlfe
#endif // end ifndef __DEVICE_HPP__
