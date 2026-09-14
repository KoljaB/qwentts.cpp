#pragma once
// backend.h: GGML backend initialization
//
// All modules use the same pattern: load all backends, pick best GPU,
// keep CPU as fallback. Each backend_init call returns a fresh backend
// pair with its own device context and memory pool, so independent
// qt_contexts never share allocator state and can run concurrently.
// Sharing within one pipeline (talker, predictor, codec) is done by
// passing the same BackendPair to each module.

#include "ggml-backend.h"
#include "qt-error.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#if defined(__linux__)
#include <sched.h>
#endif

struct BackendPair {
    ggml_backend_t backend;
    ggml_backend_t cpu_backend;
    bool           has_gpu;
    struct ggml_threadpool * cpu_threadpool = nullptr;
    void (*cpu_threadpool_free)(struct ggml_threadpool *) = nullptr;
    void (*cpu_threadpool_attach)(ggml_backend_t, struct ggml_threadpool *) = nullptr;
};

// Physical core count heuristic (logical / 2 for HT/SMT).
// Used for GGML CPU thread count: GEMM shares SIMD units across hyperthreads,
// so one thread per physical core is optimal.
//
// QWENTTS_CPU_THREADS is intentionally process-scoped and read only while
// constructing a backend. It is a bounded diagnostic/tuning override, not a
// public ABI field: malformed, empty, zero, negative, and out-of-range values
// fall back to the physical-core heuristic.
static int backend_cpu_n_threads(void) {
    int n = (int) std::thread::hardware_concurrency() / 2;
    if (n <= 0) {
        n = 1;
    }

    const char * raw = std::getenv("QWENTTS_CPU_THREADS");
    if (!raw || !*raw) {
        return n;
    }

    char * end = nullptr;
    errno      = 0;
    long requested = std::strtol(raw, &end, 10);
    if (errno == 0 && end != raw && *end == '\0' && requested >= 1 && requested <= 256) {
        return (int) requested;
    }

    qt_log(QT_LOG_WARN, "[Load] ignoring QWENTTS_CPU_THREADS=%s (expected integer 1..256); using %d", raw, n);
    return n;
}

// Standalone CPU backend via Registry API (DL-safe, no ggml-cpu.h needed).
// Sets thread count via proc address since ggml_backend_cpu_device_init_backend
// ignores its params string and always defaults to GGML_DEFAULT_N_THREADS (4).
// Returns NULL on failure.
static ggml_backend_t cpu_backend_new(int n_threads, bool require_thread_setter = false) {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t     cpu     = NULL;
    if (cpu_dev) {
        cpu = ggml_backend_dev_init(cpu_dev, NULL);
    }
    if (!cpu) {
        cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    }
    if (!cpu) {
        return NULL;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(cpu);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : NULL;
    bool thread_configured = false;
    if (reg) {
        auto set_fn =
            (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_fn) {
            set_fn(cpu, n_threads);
            thread_configured = true;
        }
    }
    if (require_thread_setter && !thread_configured) {
        qt_log(QT_LOG_ERROR, "[Load] CPU backend cannot prove requested thread count; ggml_backend_set_n_threads unavailable");
        ggml_backend_free(cpu);
        return NULL;
    }
    return cpu;
}

// Opt-in placement inside the caller's existing CPU affinity only. A pool
// barrier must not make runnable workers compete for the same CPU after idle.
static bool backend_cpu_strict_affinity() {
    const char * value = std::getenv("QWENTTS_CPU_STRICT_AFFINITY");
    return value && std::strcmp(value, "1") == 0;
}

// GGML pins its calling worker on resume. Do not leak that placement into
// the host's event loop or into a later independently created context.
struct BackendCpuAffinityGuard {
#if defined(__linux__)
    cpu_set_t previous;
    bool restore = false;
    BackendCpuAffinityGuard() {
        restore = backend_cpu_strict_affinity() && sched_getaffinity(0, sizeof(previous), &previous) == 0;
    }
    ~BackendCpuAffinityGuard() {
        if (restore) sched_setaffinity(0, sizeof(previous), &previous);
    }
#endif
};

// Only the owning qt_context releases this pool; pipeline copies borrow it.
static bool backend_cpu_pool_init(BackendPair & bp, int n_threads) {
    // Keep one pool for this context. Talker, predictor and codec execute
    // serially on its shared CPU backend; separate contexts own separate pools.
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(bp.cpu_backend));
    auto create = reinterpret_cast<struct ggml_threadpool * (*)(struct ggml_threadpool_params *)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new"));
    auto destroy = reinterpret_cast<void (*)(struct ggml_threadpool *)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free"));
    auto attach = reinterpret_cast<void (*)(ggml_backend_t, struct ggml_threadpool *)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool"));
    if (!create || !destroy || !attach) {
        qt_log(QT_LOG_ERROR, "[Load] CPU backend lacks persistent thread-pool support");
        return false;
    }
    auto pool_params = ggml_threadpool_params_default(n_threads);
    if (backend_cpu_strict_affinity()) {
#if defined(__linux__)
        cpu_set_t allowed;
        if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return false;
        int available = 0;
        for (int cpu = 0; cpu < GGML_MAX_N_THREADS && cpu < CPU_SETSIZE; ++cpu) {
            pool_params.cpumask[cpu] = CPU_ISSET(cpu, &allowed);
            available += pool_params.cpumask[cpu] ? 1 : 0;
        }
        if (available < n_threads) {
            qt_log(QT_LOG_ERROR, "[Load] Strict CPU affinity needs %d allowed CPUs, found %d", n_threads, available);
            return false;
        }
        pool_params.strict_cpu = true;
        pool_params.paused = true;
        qt_log(QT_LOG_INFO, "[Load] Strict CPU worker affinity enabled inside %d allowed CPUs", available);
#else
        qt_log(QT_LOG_ERROR, "[Load] Strict CPU affinity requires Linux");
        return false;
#endif
    }

    bp.cpu_threadpool = create(&pool_params);
    if (!bp.cpu_threadpool) {
        return false;
    }
    bp.cpu_threadpool_free = destroy;
    bp.cpu_threadpool_attach = attach;
    attach(bp.cpu_backend, bp.cpu_threadpool);    return true;
}

// The registry setter parks an old pool when detaching it. Reattach the same
// paused pool so the next graph wakes it through GGML's ordinary kickoff.
static void backend_cpu_pool_park(const BackendPair & bp) {
    if (bp.cpu_threadpool) {
        bp.cpu_threadpool_attach(bp.cpu_backend, nullptr);
        bp.cpu_threadpool_attach(bp.cpu_backend, bp.cpu_threadpool);
    }
}

// Declare after the context execution lock, so parking precedes its release.
struct BackendCpuIdleGuard {
    const BackendPair & bp;
    BackendCpuAffinityGuard affinity;
    ~BackendCpuIdleGuard() { backend_cpu_pool_park(bp); }
};

// Initialize backends: load all available (CUDA, Metal, Vulkan...),
// pick the best one, keep CPU as fallback.
// label: log prefix, e.g. "DiT", "VAE", "LM"
// Each call returns a fresh backend pair with its own memory pool.
// Returns a BackendPair with .backend == NULL when initialisation fails;
// the caller must check this before passing it to any pipeline_*_load.
// Collapse exact consecutive duplicate ggml log lines and report the total
// count when the run ends (tames the CUDA graph capture "reused" flood).
static void qt_ggml_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    static char last[256] = { 0 };
    static int  count     = 0;

    if (count > 0 && strcmp(text, last) == 0) {
        count++;
        return;
    }

    if (count > 1) {
        fprintf(stderr, "[Dedup] Previous line repeated %d times total\n", count);
    }

    fputs(text, stderr);
    strncpy(last, text, sizeof(last) - 1);
    last[sizeof(last) - 1] = 0;
    count                  = 1;
    fflush(stderr);
}

static void backend_load_all_once(void) {
    // Magic static: log callback install and dynamic backend loading
    // happen exactly once, safe under concurrent qt_init calls.
    static const bool loaded = [] {
        ggml_log_set(qt_ggml_log, nullptr);
        ggml_backend_load_all();
        return true;
    }();
    (void) loaded;
}

static BackendPair backend_init(const char * label) {
    backend_load_all_once();

    BackendPair bp = {};

    // GGML_BACKEND env var: force a specific device instead of auto-best.
    // Device names: CUDA0, Vulkan0, CPU, BLAS (see ggml_backend_dev_name).
    const char * force_backend = std::getenv("GGML_BACKEND");
    if (force_backend) {
        bp.backend = ggml_backend_init_by_name(force_backend, nullptr);
        if (!bp.backend) {
            // Assemble the device list inline so the log callback gets one
            // self-contained line instead of three. The available list can
            // grow with each backend that registers, so a std::string here
            // keeps the formatting allocation-free for the common case.
            std::string msg = "[Load] GGML_BACKEND=";
            msg += force_backend;
            msg += " not found. Available:";
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                msg += ' ';
                msg += ggml_backend_dev_name(ggml_backend_dev_get(i));
            }
            qt_log(QT_LOG_ERROR, "%s", msg.c_str());
            return BackendPair{};
        }
    } else {
        bp.backend = ggml_backend_init_best();
    }
    if (!bp.backend) {
        qt_log(QT_LOG_ERROR, "[Load] no backend available");
        return BackendPair{};
    }
    bool best_is_cpu = (strcmp(ggml_backend_name(bp.backend), "CPU") == 0);
    int  n_threads   = backend_cpu_n_threads();
    if (best_is_cpu) {
        ggml_backend_free(bp.backend);
        bp.backend     = cpu_backend_new(n_threads);
        bp.cpu_backend = bp.backend;
    } else {
        bp.cpu_backend = cpu_backend_new(n_threads);
    }
    if (!bp.cpu_backend) {
        qt_log(QT_LOG_ERROR, "[Load] failed to init CPU backend");
        if (bp.backend && bp.backend != bp.cpu_backend) {
            ggml_backend_free(bp.backend);
        }
        return BackendPair{};
    }
    if (best_is_cpu && !backend_cpu_pool_init(bp, n_threads)) {
        ggml_backend_free(bp.backend);
        return BackendPair{};
    }
    bp.has_gpu = !best_is_cpu;
    qt_log(QT_LOG_INFO, "[Load] %s backend: %s (CPU threads: %d)", label, ggml_backend_name(bp.backend), n_threads);
    return bp;
}

// CPU-only initialization for the additive qt_init_cpu ABI. It deliberately
// bypasses GGML_BACKEND and best-device selection, so it never reads or
// mutates process environment and cannot accidentally select a GPU backend.
static BackendPair backend_init_cpu(const char * label, int n_threads) {
    backend_load_all_once();

    BackendPair bp = {};
    bp.backend = cpu_backend_new(n_threads, true);
    bp.cpu_backend = bp.backend;
    bp.has_gpu = false;
    if (!bp.backend) {
        qt_log(QT_LOG_ERROR, "[Load] failed to init CPU backend");
        return BackendPair{};
    }

    if (!backend_cpu_pool_init(bp, n_threads)) {
        ggml_backend_free(bp.backend);
        return BackendPair{};
    }

    qt_log(QT_LOG_INFO, "[Load] %s backend: CPU (CPU threads: %d)", label, n_threads);
    return bp;
}

// Free a backend pair returned by backend_init.
static void backend_release(BackendPair bp) {
    auto backend = bp.backend;
    auto cpu_backend = bp.cpu_backend;
    if (backend && backend != cpu_backend) {
        ggml_backend_free(backend);
    }
    if (cpu_backend) {
        ggml_backend_free(cpu_backend);
    }
    if (bp.cpu_threadpool) {
        bp.cpu_threadpool_free(bp.cpu_threadpool);
    }
}

// Create a scheduler from a backend pair.
// max_nodes: graph size hint (4096 for small models, 8192 for large)
// When a GPU is present, use its host buffer type for the CPU backend.
// Pinned memory lets the scheduler keep more ops on GPU instead of
// falling back to CPU with plain malloc.
static ggml_backend_sched_t backend_sched_new(BackendPair bp, int max_nodes) {
    ggml_backend_t             backends[2] = { bp.backend, bp.cpu_backend };
    ggml_backend_buffer_type_t bufts[2]    = { NULL, NULL };
    int                        n           = (bp.backend == bp.cpu_backend) ? 1 : 2;

    bufts[0] = ggml_backend_get_default_buffer_type(bp.backend);
    if (n == 2) {
        ggml_backend_dev_t         gpu_dev   = ggml_backend_get_device(bp.backend);
        ggml_backend_buffer_type_t host_buft = gpu_dev ? ggml_backend_dev_host_buffer_type(gpu_dev) : NULL;
        bufts[1] = host_buft ? host_buft : ggml_backend_get_default_buffer_type(bp.cpu_backend);
    }

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, n, max_nodes, false, true);
    if (!sched) {
        qt_log(QT_LOG_ERROR, "[Load] failed to create scheduler");
        return nullptr;
    }
    return sched;
}
