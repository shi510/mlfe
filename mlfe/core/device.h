#ifndef __DEVICE_HPP__
#define __DEVICE_HPP__
#include "../utils/types.h"
#include <memory>

namespace mlfe{

class DeviceMemory;

class Device final{
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

    DeviceMemory CreateDeviceMemory() const;

    std::string Name() const;

    template <class From, class To>
    static void Copy(const DeviceMemory from, DeviceMemory to);

private:
    template <class From, class To>
    static void CopyInternal(const DeviceMemory from, DeviceMemory to);

    class SelectedDevice;
    std::shared_ptr<SelectedDevice> sd;
};

class Device::SelectedDevice{
public:
    virtual DeviceMemory CreateDeviceMemory() const = 0;

    virtual std::string Name() const = 0;
};

template <class Dev>
class Device::Select final : public Device::SelectedDevice{
public:
    DeviceMemory CreateDeviceMemory() const override;

    std::string Name() const override;
};

class DeviceMemory{
public:
    DeviceMemory();

    type::uint8::T *Data() const;

    template <class T>
    T *Data() const;

    void Allocate(type::uint32::T size);

    void Allocate(type::uint8::T *ptr, type::uint32::T size);

    type::uint32::T Size() const;

    class Allocator;

protected:
    DeviceMemory(std::shared_ptr<Allocator> alloc);

private:
    friend class Device::Select<Device::CPU>;
    friend class Device::Select<Device::CUDA>;

    std::shared_ptr<Allocator> alloc;
};

class DeviceMemory::Allocator{
public:
    virtual ~Allocator();

    virtual type::uint8::T *Data() const = 0;

    virtual void Allocate(type::uint32::T size) = 0;

    virtual void Allocate(type::uint8::T *ptr, type::uint32::T size) = 0;

    virtual type::uint32::T Size() const = 0;
};

template <class Dev>
Device::Device(Select<Dev> s){
    sd = std::make_shared<Select<Dev>>(s);
}

template <class From, class To>
void Device::Copy(const DeviceMemory from, DeviceMemory to){
    CopyInternal<From, To>(from, to);
}

template <class T>
T *DeviceMemory::Data() const{
    return reinterpret_cast<T *>(alloc->Data());
}

template <> void
Device::CopyInternal<Device::CPU, Device::CPU>(const DeviceMemory from, DeviceMemory to);

template <> void
Device::CopyInternal<Device::CPU, Device::CUDA>(const DeviceMemory from, DeviceMemory to);

template <> void
Device::CopyInternal<Device::CUDA, Device::CPU>(const DeviceMemory from, DeviceMemory to);

template <> void
Device::CopyInternal<Device::CUDA, Device::CUDA>(const DeviceMemory from, DeviceMemory to);

} // end namespace mlfe
#endif // end ifndef __DEVICE_HPP__
