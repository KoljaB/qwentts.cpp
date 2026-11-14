// ICL prefix cache test on a real Base model: the audio with the cache on
// (first request and reuse) equals the audio with the cache off, sample for
// sample, for several utterances and two voices; the counters and the entry
// limit behave as documented.
//
// Usage: test-icl-prefix-cache <talker.gguf> <codec.gguf> [cpu_threads]
// The reference "voices" are two synthetic tones: the test is about the
// arithmetic of the prefill, not about how the result sounds.
#include "qwen.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "ICL_PREFIX_CACHE_TEST_FAIL: %s (%s)\n", what, qt_last_error());
        std::exit(1);
    }
}

static std::vector<float> tone(float hz, float seconds) {
    std::vector<float> pcm((size_t) (seconds * 24000.0f));
    for (size_t i = 0; i < pcm.size(); i++) {
        const float t = (float) i / 24000.0f;
        pcm[i]        = 0.3f * std::sin(6.2831853f * hz * t) * (0.6f + 0.4f * std::sin(6.2831853f * 3.0f * t));
    }
    return pcm;
}

static std::vector<float> speak(qt_context * q, const qt_voice_ref & voice, const char * ref_text, const char * text) {
    qt_tts_params p;
    qt_tts_default_params(&p);
    p.text           = text;
    p.lang           = "english";
    p.seed           = 1234;
    p.max_new_tokens = 24;
    p.ref_spk_emb    = voice.ref_spk_emb;
    p.ref_spk_dim    = voice.ref_spk_dim;
    p.ref_codes      = voice.ref_codes;
    p.ref_T          = voice.ref_T;
    p.ref_text       = ref_text;
    qt_audio out     = {};
    require(qt_synthesize(q, &p, &out) == QT_STATUS_OK && out.n_samples > 0, "qt_synthesize");
    std::vector<float> pcm(out.samples, out.samples + out.n_samples);
    qt_audio_free(&out);
    return pcm;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <talker.gguf> <codec.gguf> [cpu_threads]\n", argv[0]);
        return 2;
    }
    qt_init_params ip;
    qt_init_default_params(&ip);
    ip.talker_path = argv[1];
    ip.codec_path  = argv[2];
    qt_context * q = argc > 3 ? qt_init_cpu(&ip, std::atoi(argv[3])) : qt_init(&ip);
    require(q != NULL, "qt_init");

    require(qt_set_icl_prefix_cache(NULL, 4) == QT_STATUS_INVALID_PARAMS, "NULL handle rejected");
    require(qt_set_icl_prefix_cache(q, -1) == QT_STATUS_INVALID_PARAMS, "negative size rejected");
    require(qt_set_icl_prefix_cache(q, 257) == QT_STATUS_INVALID_PARAMS, "oversize rejected");

    const std::vector<float> wav_a = tone(180.0f, 3.0f), wav_b = tone(240.0f, 4.0f);
    qt_voice_ref             a = {}, b = {};
    require(qt_extract_voice_ref(q, wav_a.data(), (int) wav_a.size(), &a) == QT_STATUS_OK, "voice a");
    require(qt_extract_voice_ref(q, wav_b.data(), (int) wav_b.size(), &b) == QT_STATUS_OK, "voice b");
    const char * ref_a   = "The first reference sentence.";
    const char * ref_b   = "Another voice says something else.";
    const char * texts[] = { "Yes.", "That sounds good, let us do it.", "Wait a moment, I will have a look." };

    // Cache off: the reference results. Nothing is counted or stored.
    std::vector<std::vector<float>> plain_a, plain_b;
    for (const char * text : texts) {
        plain_a.push_back(speak(q, a, ref_a, text));
        plain_b.push_back(speak(q, b, ref_b, text));
    }
    uint64_t hits = 9, misses = 9;
    require(qt_icl_prefix_cache_stats(q, &hits, &misses) == 0 && hits == 0 && misses == 0, "off: no entries");

    // Cache on: first request per voice stores, the others reuse.
    require(qt_set_icl_prefix_cache(q, 4) == QT_STATUS_OK, "cache on");
    for (int round = 0; round < 2; round++) {
        for (size_t i = 0; i < 3; i++) {
            require(speak(q, a, ref_a, texts[i]) == plain_a[i], "voice a: same audio as without the cache");
            require(speak(q, b, ref_b, texts[i]) == plain_b[i], "voice b: same audio as without the cache");
        }
    }
    require(qt_icl_prefix_cache_stats(q, &hits, &misses) == 2 && misses == 2 && hits == 10, "two voices, ten reuses");

    // One entry: the two voices evict each other, the audio stays the same.
    require(qt_set_icl_prefix_cache(q, 1) == QT_STATUS_OK, "cache size 1");
    require(qt_icl_prefix_cache_stats(q, NULL, NULL) == 1, "shrunk to one entry");
    require(speak(q, a, ref_a, texts[1]) == plain_a[1], "size 1: voice a");
    require(speak(q, b, ref_b, texts[1]) == plain_b[1], "size 1: voice b");
    require(speak(q, b, ref_b, texts[2]) == plain_b[2], "size 1: voice b again");
    require(qt_icl_prefix_cache_stats(q, &hits, &misses) == 1, "still one entry");

    // Off again: entries are gone, results unchanged.
    require(qt_set_icl_prefix_cache(q, 0) == QT_STATUS_OK, "cache off");
    require(qt_icl_prefix_cache_stats(q, NULL, NULL) == 0, "off: entries freed");
    require(speak(q, a, ref_a, texts[0]) == plain_a[0], "off: voice a");

    qt_voice_ref_free(&a);
    qt_voice_ref_free(&b);
    qt_free(q);
    std::puts("ICL_PREFIX_CACHE_TEST_PASS: cached audio equals uncached audio, counters and entry limit hold");
    return 0;
}
