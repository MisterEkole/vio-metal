#import "MetalContext.h"
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include <iostream>

namespace vio {

// Helper to cast from void* back to Metal types
#define AS_DEVICE ((__bridge id<MTLDevice>)device_)
#define AS_QUEUE  ((__bridge id<MTLCommandQueue>)command_queue_)

MetalContext::MetalContext() : device_(nullptr), command_queue_(nullptr), last_buffer_(nullptr) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "[MetalContext] Failed to create Metal Device\n";
        return;
    }
    device_ = (__bridge_retained void*)device;
    command_queue_ = (__bridge_retained void*)[device newCommandQueue];
}

MetalContext::~MetalContext() {
    if (device_) { CFRelease(device_); }
    if (command_queue_) { CFRelease(command_queue_); }
}

void* MetalContext::getDevice() const { return device_; }
void* MetalContext::getCommandQueue() const { return command_queue_; }

void MetalContext::setLastBuffer(void* buf) { last_buffer_ = buf; }

void MetalContext::waitForLastBuffer() {
    if (!last_buffer_) return;
    id<MTLCommandBuffer> buf = (__bridge id<MTLCommandBuffer>)last_buffer_;
    [buf waitUntilCompleted];
    last_buffer_ = nullptr;
}

void* MetalContext::newSharedBuffer(size_t length) {
    id<MTLBuffer> buffer = [AS_DEVICE newBufferWithLength:length options:MTLResourceStorageModeShared];
    return (__bridge_retained void*)buffer;
}

void* MetalContext::loadLibrary(const std::string& path) {
    NSError* error = nil;
    id<MTLLibrary> library = [AS_DEVICE newLibraryWithFile:[NSString stringWithUTF8String:path.c_str()] error:&error];
    if (!library) {
        std::cerr << "[MetalContext] Library Error: " << [[error localizedDescription] UTF8String] << "\n";
        return nullptr;
    }
    return (__bridge_retained void*)library;
}

void* MetalContext::getPipeline(const std::string& name, void* library) {
    id<MTLLibrary> lib = (__bridge id<MTLLibrary>)library;
    id<MTLFunction> func = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
    if (!func) return nullptr;

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [AS_DEVICE newComputePipelineStateWithFunction:func error:&error];
    if (!pipeline) {
        std::cerr << "[MetalContext] Pipeline Error: " << [[error localizedDescription] UTF8String] << "\n";
        return nullptr;
    }
    return (__bridge_retained void*)pipeline;
}

} // namespace vio