# FAQ

**Q: What input size should I use?**
A: 224x224 is the standard default. Use 128 for edge devices, 256+ for higher accuracy.

**Q: How do I choose a model size?**
A: m-0.5x for edge/mobile, m for general use, m-1.5x for best accuracy.

**Q: Training is slow on CPU?**
A: Use a smaller model (m-0.5x), reduce input size, or use a GPU.

**Q: CUDA out of memory?**
A: Reduce batch size, enable `amp=True`, try `activation_checkpointing=True`.

**Q: Low accuracy?**
A: Ensure sufficient data (100+ images/class), use augmentation, increase epochs.
