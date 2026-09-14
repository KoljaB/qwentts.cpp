// Real CPU backend test: independent ownership, idle parking and wakeup.
#include "backend.h"
#include <vector>
static void require(bool ok) { if (!ok) { std::fprintf(stderr, "CPU pool regression failed\n"); std::exit(1); } }
static void compute(const BackendPair & bp, float value) {
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    auto ctx = ggml_init(params);
    require(ctx != nullptr);
    auto a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    auto b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 8);
    auto out = ggml_mul_mat(ctx, a, b);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, bp.backend);
    require(buffer != nullptr);
    std::vector<float> av(64 * 64, 1.0f), bv(64 * 8, value), result(64 * 8);
    ggml_backend_tensor_set(a, av.data(), 0, av.size() * sizeof(float));
    ggml_backend_tensor_set(b, bv.data(), 0, bv.size() * sizeof(float));
    require(ggml_backend_graph_compute(bp.backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    for (float x : result) require(x == 64 * value);
    backend_cpu_pool_park(bp);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}
int main() {
#ifdef _WIN32
    _putenv_s("QWENTTS_CPU_THREADS", "2");
    _putenv_s("GGML_BACKEND", "CPU");
#else
    setenv("QWENTTS_CPU_THREADS", "2", 1);
    setenv("GGML_BACKEND", "CPU", 1);
#endif
    for (int i = 0; i < 3; ++i) {
        auto first = backend_init_cpu("test forced CPU", 2);
        auto second = backend_init("test selected CPU");
        require(first.cpu_threadpool && second.cpu_threadpool);
        require(first.cpu_threadpool != second.cpu_threadpool);
        compute(first, 2); compute(second, 3); compute(first, 4);
        backend_release(first);
        compute(second, 5);
        backend_release(second);
    }
    std::puts("CPU_POOL_TEST_PASS: independent pools, parked wakeup, surviving context, repeated teardown");
}
