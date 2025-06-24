# Morton Stream Compaction 🚀

This project implements a baseline CPU + GPU Stream Compaction system for filtering 2D wind field data.

## ✅ Features (v1.0)
- Morton Z-order encoding for locality
- CPU and GPU sorting (std::sort, Thrust CPU/GPU)
- CPU stream compaction
- Thrust-based stream compaction (CPU/GPU)

## 🛠️ Build
```bash
make
./build/main
