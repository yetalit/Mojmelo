#include <sys/sysctl.h>

int cachel1() {
    int l1_cache_size;
    size_t len = sizeof(int);

    // Get L1 Cache Size
    if (sysctlbyname("hw.perflevel0.l1dcachesize", &l1_cache_size, &len, NULL, 0) == 0) {
        if (l1_cache_size <= 1) {
			if (sysctlbyname("hw.l1dcachesize", &l1_cache_size, &len, NULL, 0) == 0) {
				if (l1_cache_size <= 1) {
					return 65536;
				}
				return l1_cache_size;
			} else {
				return 65536;
			}
		}
		return l1_cache_size;
    } else {
        if (sysctlbyname("hw.l1dcachesize", &l1_cache_size, &len, NULL, 0) == 0) {
			if (l1_cache_size <= 1) {
				return 65536;
			}
			return l1_cache_size;
		} else {
			return 65536;
		}
    }
}

int cachel2() {
    int l2_cache_size;
    size_t len = sizeof(int);

    // Get L2 Cache Size
    if (sysctlbyname("hw.perflevel0.l2dcachesize", &l2_cache_size, &len, NULL, 0) == 0) {
        if (l2_cache_size <= 1) {
			if (sysctlbyname("hw.l2dcachesize", &l2_cache_size, &len, NULL, 0) == 0) {
				if (l2_cache_size <= 1) {
					return 4194304;
				}
				return l2_cache_size;
			} else {
				return 4194304;
			}
		}
		return l2_cache_size;
    } else {
        if (sysctlbyname("hw.l2dcachesize", &l2_cache_size, &len, NULL, 0) == 0) {
			if (l2_cache_size <= 1) {
				return 4194304;
			}
			return l2_cache_size;
		} else {
			return 4194304;
		}
    }
}
