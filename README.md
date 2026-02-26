<INSERT_FULL_README_CONTENT_HERE_OF_README.md>

## Architecture & Data Flow

1. **Input Source**
    
    Stereo Images
    
2. **Metal GPU Pipeline Flow (vio-metal-gpu)**

    Stereo Images → Metal Undistort → Metal FAST + Harris NMS → Metal ORB → Metal Stereo Matcher
    
3. **Metal KLT Tracker** (now moved to CPU)