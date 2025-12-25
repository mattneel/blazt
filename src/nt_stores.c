// Minimal non-temporal (streaming) memcpy helper.
//
// Zig's inline asm (0.16.0-dev.1657) currently rejects movnt* mnemonics, so we
// emit the instruction via C inline asm and link it into the Zig module.
//
// This file intentionally avoids libc headers.

typedef __SIZE_TYPE__ size_t;

// Copy `n` bytes from `src` to `dst` using non-temporal stores where supported.
//
// Contract:
// - This is intended for **non-overlapping** ranges.
// - Callers should ensure `dst` is 16-byte aligned for best results.
void blazt_nt_memcpy(unsigned char *dst, const unsigned char *src, size_t n) {
    size_t i = 0;

#if defined(__x86_64__) || defined(__i386__)
    // Stream-store 16B chunks.
    for (; i + 16 <= n; i += 16) {
        __asm__ __volatile__(
            "movups (%[src]), %%xmm0\n\t"
            "movntps %%xmm0, (%[dst])\n\t"
            :
            : [src] "r"(src + i),
              [dst] "r"(dst + i)
            : "xmm0", "memory");
    }
#endif

    // Tail copy (or full copy on non-x86).
    for (; i < n; i += 1) {
        dst[i] = src[i];
    }

#if defined(__x86_64__) || defined(__i386__)
    // Ensure streaming stores are globally visible before returning.
    __asm__ __volatile__("sfence" ::: "memory");
#endif
}

// Set `n` bytes at `dst` to 0 using non-temporal stores where supported.
//
// Contract:
// - Intended for large, write-only-ish regions where cache pollution matters.
// - Callers should ensure `dst` is 16-byte aligned for best results.
void blazt_nt_memset_zero(unsigned char *dst, size_t n) {
    size_t i = 0;

#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pxor %%xmm0, %%xmm0" ::: "xmm0");
    for (; i + 16 <= n; i += 16) {
        __asm__ __volatile__(
            "movntps %%xmm0, (%[dst])\n\t"
            :
            : [dst] "r"(dst + i)
            : "memory");
    }
#endif

    for (; i < n; i += 1) {
        dst[i] = 0;
    }

#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("sfence" ::: "memory");
#endif
}


