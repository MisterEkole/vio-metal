#pragma once

#include <string>

namespace vio {

// Forward declaration of the internal Obj-C wrapper
class MetalContext {
public:
    static bool isAvailable();

    MetalContext();
    ~MetalContext();

    void* getDevice() const;
    void* getCommandQueue() const;
    void* newSharedBuffer(size_t length);
    void* loadLibrary(const std::string& path);
    void* getPipeline(const std::string& name, void* library);

private:
    // PIMPL pattern to hide Objective-C types
    struct ObjCImpl;
    ObjCImpl* impl;
};

} 