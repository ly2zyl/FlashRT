// FlashRT llama.cpp engine backed directly by Houmo's HLIELLama libllama.
//
// This provider keeps GGML/llama.cpp types behind frt_llama_cpp_engine_v1 and
// therefore exposes the same stable frt_model_runtime_v1 boundary as the
// existing Jetson-PI provider.  Model execution remains owned by the vendor
// libllama build, which selects the HoumoNPU backend and drives M50/XH2.

#include "flashrt/providers/llama_cpp/jetson_pi_engine.h"

#if defined(FLASHRT_CPP_WITH_HOUMO_LLAMA)

#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Engine {
    llama_model* model = nullptr;
    llama_context* context = nullptr;
    llama_context_params context_params{};
    llama_sampler* sampler = nullptr;
    const llama_vocab* vocab = nullptr;

    std::string prompt;
    std::vector<llama_token> input_tokens;
    std::string output;
    std::string last_error;
    uint32_t max_tokens = 0;
    uint32_t generated_tokens = 0;
    llama_token next_token = 0;
    int32_t is_eog = 0;
    bool input_ready = false;
    bool tokens_input = false;
    bool prefilled = false;
    bool next_token_ready = false;
    bool logits_ready = false;
    bool context_pristine = true;
    std::atomic<long> refs{1};
};

static std::once_flag g_backend_once;
static thread_local std::string g_create_error;

void set_error(Engine* engine, const std::string& message) {
    if (engine) engine->last_error = message;
}

std::string format_user_prompt(Engine* engine, const std::string& prompt) {
    const char* tmpl = llama_model_chat_template(engine->model, nullptr);
    if (!tmpl || !tmpl[0]) return prompt;

    const llama_chat_message message{"user", prompt.c_str()};
    int32_t needed = llama_chat_apply_template(
        tmpl, &message, 1, true, nullptr, 0);
    if (needed <= 0) return prompt;

    std::vector<char> buffer(static_cast<size_t>(needed) + 1, '\0');
    const int32_t written = llama_chat_apply_template(
        tmpl, &message, 1, true, buffer.data(),
        static_cast<int32_t>(buffer.size()));
    if (written <= 0) return prompt;
    return std::string(buffer.data(), static_cast<size_t>(written));
}

bool tokenize(Engine* engine, const std::string& text,
              std::vector<llama_token>* tokens) {
    int32_t count = llama_tokenize(
        engine->vocab, text.data(), static_cast<int32_t>(text.size()),
        nullptr, 0, true, true);
    if (count == std::numeric_limits<int32_t>::min()) {
        set_error(engine, "prompt is too large to tokenize");
        return false;
    }
    if (count < 0) count = -count;
    if (count <= 0) {
        set_error(engine, "prompt tokenized to an empty sequence");
        return false;
    }
    tokens->resize(static_cast<size_t>(count));
    const int32_t actual = llama_tokenize(
        engine->vocab, text.data(), static_cast<int32_t>(text.size()),
        tokens->data(), count, true, true);
    if (actual <= 0) {
        set_error(engine, "llama_tokenize failed");
        return false;
    }
    tokens->resize(static_cast<size_t>(actual));
    return true;
}

bool append_piece(Engine* engine, llama_token token) {
    char local[256];
    int32_t size = llama_token_to_piece(
        engine->vocab, token, local, sizeof(local), 0, false);
    if (size >= 0) {
        engine->output.append(local, static_cast<size_t>(size));
        return true;
    }
    if (size == std::numeric_limits<int32_t>::min()) {
        set_error(engine, "token piece length overflow");
        return false;
    }
    const int32_t needed = -size;
    std::vector<char> buffer(static_cast<size_t>(needed));
    size = llama_token_to_piece(
        engine->vocab, token, buffer.data(), needed, 0, false);
    if (size < 0) {
        set_error(engine, "llama_token_to_piece failed");
        return false;
    }
    engine->output.append(buffer.data(), static_cast<size_t>(size));
    return true;
}

void retain(void* self) {
    static_cast<Engine*>(self)->refs.fetch_add(1, std::memory_order_relaxed);
}

void release(void* self) {
    auto* engine = static_cast<Engine*>(self);
    if (engine->refs.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
    if (engine->sampler) llama_sampler_free(engine->sampler);
    if (engine->context) llama_free(engine->context);
    if (engine->model) llama_model_free(engine->model);
    delete engine;
}

int set_input(void* self, uint32_t port, const void* data, uint64_t bytes,
              int /*stream*/) {
    auto* engine = static_cast<Engine*>(self);
    if (!engine) return -1;
    engine->last_error.clear();
    engine->input_ready = false;
    engine->tokens_input = false;
    engine->prefilled = false;
    engine->next_token_ready = false;
    engine->logits_ready = false;
    engine->generated_tokens = 0;
    engine->prompt.clear();
    engine->input_tokens.clear();
    engine->output.clear();
    if (port != FRT_LLAMA_CPP_LLM_PORT_PROMPT &&
        port != FRT_LLAMA_CPP_LLM_PORT_TOKENS) {
        set_error(engine, "Houmo Llama engine accepts prompt or token input");
        return -2;
    }
    if (!data || bytes == 0 ||
        bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        set_error(engine, "prompt must be a non-empty byte string");
        return -1;
    }
    if (port == FRT_LLAMA_CPP_LLM_PORT_PROMPT) {
        engine->prompt.assign(static_cast<const char*>(data),
                              static_cast<size_t>(bytes));
    } else {
        if (bytes % sizeof(int32_t) != 0) {
            set_error(engine, "tokens payload must be a non-empty int32 array");
            return -1;
        }
        const auto* begin = static_cast<const int32_t*>(data);
        engine->input_tokens.assign(begin, begin + bytes / sizeof(int32_t));
        engine->tokens_input = true;
    }
    engine->input_ready = true;
    return 0;
}

int reset_session(Engine* engine) {
    // Houmo's HMM backend keeps an internal KV offset that is not reset by
    // llama_memory_clear()/llama_memory_seq_rm(). Recreate only the context
    // to guarantee session isolation while keeping model weights resident.
    if (!engine->context_pristine) {
        llama_free(engine->context);
        engine->context =
            llama_init_from_model(engine->model, engine->context_params);
        if (!engine->context) {
            set_error(engine, "failed to recreate Houmo Llama context");
            return -8;
        }
    }
    engine->context_pristine = true;
    llama_sampler_reset(engine->sampler);
    engine->output.clear();
    engine->generated_tokens = 0;
    engine->prefilled = false;
    engine->next_token_ready = false;
    engine->logits_ready = false;
    return 0;
}

bool prepare_tokens(Engine* engine, std::vector<llama_token>* tokens) {
    if (engine->tokens_input) {
        *tokens = engine->input_tokens;
        return !tokens->empty();
    }
    const std::string formatted = format_user_prompt(engine, engine->prompt);
    return tokenize(engine, formatted, tokens);
}

int run_prefill(Engine* engine) {
    if (!engine->input_ready) {
        set_error(engine, "prefill requires prompt or token input");
        return -1;
    }

    if (reset_session(engine) != 0) return -8;
    std::vector<llama_token> tokens;
    if (!prepare_tokens(engine, &tokens)) {
        if (engine->last_error.empty()) {
            set_error(engine, "token input must be non-empty");
        }
        return -2;
    }
    if (tokens.size() >= llama_n_ctx(engine->context)) {
        set_error(engine, "tokenized prompt exceeds the configured context");
        return -2;
    }

    llama_batch batch = llama_batch_get_one(
        tokens.data(), static_cast<int32_t>(tokens.size()));
    int32_t rc = llama_decode(engine->context, batch);
    if (rc != 0) {
        set_error(engine, "llama_decode failed during prompt prefill (rc=" +
                          std::to_string(rc) + ")");
        return -8;
    }
    engine->prefilled = true;
    engine->context_pristine = false;
    engine->logits_ready = llama_get_logits_ith(engine->context, -1) != nullptr;
    if (!engine->logits_ready) {
        set_error(engine, "prefill completed without next-token logits");
        return -8;
    }
    return 0;
}

int run_decode(Engine* engine) {
    engine->next_token_ready = false;
    if (!engine->prefilled) {
        set_error(engine, "decode requires a successful prefill");
        return -1;
    }
    if (engine->generated_tokens >= engine->max_tokens) {
        set_error(engine, "decode exceeds configured max_tokens");
        return -1;
    }
    if (!engine->logits_ready) {
        set_error(engine, "decode requires next-token logits");
        return -7;
    }

    engine->logits_ready = false;
    engine->next_token =
        llama_sampler_sample(engine->sampler, engine->context, -1);
    engine->is_eog = llama_vocab_is_eog(engine->vocab, engine->next_token) ? 1 : 0;
    engine->next_token_ready = true;
    ++engine->generated_tokens;
    if (engine->is_eog) return 0;

    if (!append_piece(engine, engine->next_token)) {
        engine->next_token_ready = false;
        return -8;
    }
    llama_token next = engine->next_token;
    llama_batch batch = llama_batch_get_one(&next, 1);
    const int32_t rc = llama_decode(engine->context, batch);
    if (rc != 0) {
        engine->next_token_ready = false;
        set_error(engine, "llama_decode failed during generation (rc=" +
                          std::to_string(rc) + ")");
        return -8;
    }
    engine->logits_ready = llama_get_logits_ith(engine->context, -1) != nullptr;
    if (!engine->logits_ready) {
        engine->next_token_ready = false;
        set_error(engine, "decode completed without next-token logits");
        return -8;
    }
    return 0;
}

int run_infer(void* self) {
    auto* engine = static_cast<Engine*>(self);
    if (!engine) return -1;
    engine->last_error.clear();
    if (run_prefill(engine) != 0) return -8;

    for (uint32_t i = 0; i < engine->max_tokens; ++i) {
        if (run_decode(engine) != 0) return -8;
        if (engine->is_eog) break;
    }
    engine->next_token_ready = false;
    return 0;
}

int run_stage(void* self, uint32_t stage) {
    auto* engine = static_cast<Engine*>(self);
    if (!engine) return -1;
    engine->last_error.clear();
    if (stage == FRT_LLAMA_CPP_LLM_STAGE_INDEX_INFER) return run_infer(self);
    if (stage == FRT_LLAMA_CPP_LLM_STAGE_INDEX_RESET) {
        return reset_session(engine);
    }
    if (stage == FRT_LLAMA_CPP_LLM_STAGE_INDEX_PREFILL) {
        return run_prefill(engine);
    }
    if (stage == FRT_LLAMA_CPP_LLM_STAGE_INDEX_DECODE) {
        return run_decode(engine);
    }
    set_error(engine, "unknown Houmo Llama stage");
    return -1;
}

int get_output(void* self, uint32_t port, void* out, uint64_t capacity,
               uint64_t* written, int /*stream*/) {
    auto* engine = static_cast<Engine*>(self);
    if (!engine || !written) return -1;
    if (port == FRT_LLAMA_CPP_LLM_PORT_NEXT_TOKEN) {
        *written = sizeof(engine->next_token);
        if (!engine->next_token_ready) {
            set_error(engine, "next token not ready");
            return -7;
        }
        if (!out || capacity < sizeof(engine->next_token)) return -5;
        std::memcpy(out, &engine->next_token, sizeof(engine->next_token));
        return 0;
    }
    if (port == FRT_LLAMA_CPP_LLM_PORT_IS_EOG) {
        *written = sizeof(engine->is_eog);
        if (!engine->next_token_ready) {
            set_error(engine, "is_eog not ready");
            return -7;
        }
        if (!out || capacity < sizeof(engine->is_eog)) return -5;
        std::memcpy(out, &engine->is_eog, sizeof(engine->is_eog));
        return 0;
    }
    if (port == FRT_LLAMA_CPP_LLM_PORT_LOGITS) {
        const uint64_t needed = static_cast<uint64_t>(
            llama_vocab_n_tokens(engine->vocab)) * sizeof(float);
        *written = needed;
        if (!engine->logits_ready) {
            set_error(engine, "logits not ready");
            return -7;
        }
        if (!out || capacity < needed) return -5;
        const float* logits = llama_get_logits_ith(engine->context, -1);
        if (!logits) {
            set_error(engine, "llama_get_logits_ith returned null");
            return -8;
        }
        std::memcpy(out, logits, static_cast<size_t>(needed));
        return 0;
    }
    if (port != FRT_LLAMA_CPP_LLM_PORT_TEXT) {
        set_error(engine, "unknown Houmo Llama output port");
        return -2;
    }
    const uint64_t needed = static_cast<uint64_t>(engine->output.size());
    *written = needed;
    if (!out || capacity < needed) return -5;
    if (needed) std::memcpy(out, engine->output.data(), engine->output.size());
    return 0;
}

const char* last_error(void* self) {
    auto* engine = static_cast<Engine*>(self);
    if (!engine) return "null Houmo Llama engine";
    return engine->last_error.empty() ? "no error" : engine->last_error.c_str();
}

int create_llm(void*, const frt_llama_cpp_llm_config* config,
               frt_llama_cpp_engine_v1* out) {
    g_create_error.clear();
    if (!config || !out || !config->model_path || !config->backend) return -1;
    std::memset(out, 0, sizeof(*out));
    if (std::strcmp(config->backend, "houmo") != 0 &&
        std::strcmp(config->backend, "m50") != 0) {
        g_create_error = "Houmo Llama backend must be 'houmo' or 'm50'";
        return -2;
    }
    if (!std::filesystem::is_regular_file(config->model_path)) {
        g_create_error = "GGUF model path is not a regular file";
        return -2;
    }
    if (config->max_tokens == 0) {
        g_create_error = "max_tokens must be greater than zero";
        return -2;
    }

    std::call_once(g_backend_once, [] { llama_backend_init(); });
    auto* engine = new (std::nothrow) Engine();
    if (!engine) return -5;
    engine->max_tokens = config->max_tokens;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1;  // full offload to the HoumoNPU backend
    engine->model = llama_model_load_from_file(config->model_path, model_params);
    if (!engine->model) {
        g_create_error = "llama_model_load_from_file failed";
        release(engine);
        return -8;
    }
    engine->vocab = llama_model_get_vocab(engine->model);
    if (!engine->vocab) {
        g_create_error = "loaded GGUF has no vocabulary";
        release(engine);
        return -8;
    }

    engine->context_params = llama_context_default_params();
    engine->context_params.n_ctx = config->n_ctx ? config->n_ctx : 2048;
    engine->context_params.n_batch =
        std::min<uint32_t>(engine->context_params.n_ctx, 2048);
    engine->context_params.n_ubatch =
        std::min<uint32_t>(engine->context_params.n_batch, 512);
    if (config->n_threads > 0) {
        engine->context_params.n_threads = config->n_threads;
        engine->context_params.n_threads_batch = config->n_threads;
    }
    engine->context =
        llama_init_from_model(engine->model, engine->context_params);
    if (!engine->context) {
        g_create_error = "llama_init_from_model failed";
        release(engine);
        return -8;
    }

    llama_sampler_chain_params sampler_params =
        llama_sampler_chain_default_params();
    engine->sampler = llama_sampler_chain_init(sampler_params);
    if (!engine->sampler) {
        g_create_error = "llama_sampler_chain_init failed";
        release(engine);
        return -5;
    }
    if (config->temp <= 0.0f) {
        llama_sampler_chain_add(engine->sampler, llama_sampler_init_greedy());
    } else {
        if (config->top_k != 0) {
            llama_sampler_chain_add(
                engine->sampler, llama_sampler_init_top_k(config->top_k));
        }
        if (config->top_p > 0.0f) {
            llama_sampler_chain_add(
                engine->sampler, llama_sampler_init_top_p(config->top_p, 1));
        }
        llama_sampler_chain_add(
            engine->sampler, llama_sampler_init_temp(config->temp));
        llama_sampler_chain_add(
            engine->sampler, llama_sampler_init_dist(config->seed));
    }

    out->struct_size = FRT_LLAMA_CPP_ENGINE_V1_RUN_STAGE_SIZE;
    out->self = engine;
    out->retain = retain;
    out->release = release;
    out->set_input = set_input;
    out->run_infer = run_infer;
    out->get_output = get_output;
    out->last_error = last_error;
    out->run_stage = run_stage;
    return 0;
}

const char* factory_last_error(void*) {
    return g_create_error.empty() ? "no error" : g_create_error.c_str();
}

}  // namespace

extern "C" const frt_llama_cpp_engine_factory_v1*
frt_llama_cpp_default_engine_factory(void) {
    static const frt_llama_cpp_engine_factory_v1 factory = {
        sizeof(frt_llama_cpp_engine_factory_v1),
        0,
        nullptr,
        nullptr,
        create_llm,
        nullptr,
        factory_last_error,
    };
    return &factory;
}

#endif  // FLASHRT_CPP_WITH_HOUMO_LLAMA
