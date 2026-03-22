#pragma once
#include <string>
#include <vector>

namespace vio {

class MetalContext {
public:
    MetalContext();
    ~MetalContext();

    void* getDevice() const;
    void* getCommandQueue() const;
    
    void setLastBuffer(void* buf);
    void waitForLastBuffer();
    

    void* newSharedBuffer(size_t length);
    void* loadLibrary(const std::string& path);
    void* getPipeline(const std::string& name, void* library);

private:
    void* device_;
    void* command_queue_;
    void* last_buffer_;
};

} // namespace vio