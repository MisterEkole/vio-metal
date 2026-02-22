#import "MetalContext.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>

namespace vio {

struct MetalContext::ObjCImpl {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    // We use strong references here so ARC keeps them alive
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelineCache;
};

bool MetalContext::isAvailable() {
    @autoreleasepool {
        return MTLCreateSystemDefaultDevice() != nil;
    }
}

MetalContext::MetalContext() {
    impl = new ObjCImpl();
    impl->device = MTLCreateSystemDefaultDevice();
    
    if (!impl->device) {
        std::cerr << "[MetalContext] No Metal device found.\n";
        return;
    }

    impl->commandQueue = [impl->device newCommandQueue];
    impl->pipelineCache = [NSMutableDictionary new];
    
    std::cout << "[MetalContext] M-Series GPU Active: " 
              << [[impl->device name] UTF8String] << "\n";
}

MetalContext::~MetalContext() {
    // Because we are using ARC (-fobjc-arc in CMake), 
    // setting these to nil triggers the release.
    if (impl) {
        impl->device = nil;
        impl->commandQueue = nil;
        impl->pipelineCache = nil;
        delete impl;
    }
}

void* MetalContext::getDevice() const { 
    return (__bridge void*)impl->device; 
}

void* MetalContext::getCommandQueue() const { 
    return (__bridge void*)impl->commandQueue; 
}

void* MetalContext::newSharedBuffer(size_t length) {
    id<MTLBuffer> buffer = [impl->device newBufferWithLength:length 
                                                    options:MTLResourceStorageModeShared];
    // Use __bridge to pass the pointer without transferring ownership.
    // The C++ side doesn't "own" this, the Context's ARC does.
    return (__bridge void*)buffer;
}

void* MetalContext::loadLibrary(const std::string& path) {
    @autoreleasepool {
        NSError* error = nil;
        NSString* nsPath = [NSString stringWithUTF8String:path.c_str()];
        NSURL* url = [NSURL fileURLWithPath:nsPath];
        id<MTLLibrary> library = [impl->device newLibraryWithURL:url error:&error];
        
        if (error) {
            std::cerr << "[MetalContext] Library Error: " << [[error localizedDescription] UTF8String] << "\n";
            return nullptr;
        }
        // This transfers a +1 reference count to the C++ side.
        return (__bridge_retained void*)library;
    }
}

void* MetalContext::getPipeline(const std::string& name, void* library) {
    NSString* nsName = [NSString stringWithUTF8String:name.c_str()];
    id<MTLLibrary> lib = (__bridge id<MTLLibrary>)library;
    
    // Check cache
    if (impl->pipelineCache[nsName]) {
        return (__bridge void*)impl->pipelineCache[nsName];
    }

    id<MTLFunction> func = [lib newFunctionWithName:nsName];
    if (!func) {
        std::cerr << "[MetalContext] Failed to find kernel: " << name << "\n";
        return nullptr;
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [impl->device newComputePipelineStateWithFunction:func error:&error];
    
    if (pipeline) {
        [impl->pipelineCache setObject:pipeline forKey:nsName];
        return (__bridge void*)pipeline;
    } else {
        std::cerr << "[MetalContext] Pipeline Error: " << [[error localizedDescription] UTF8String] << "\n";
        return nullptr;
    }
}

} 