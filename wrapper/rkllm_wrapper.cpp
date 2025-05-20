#include "rkllm_wrapper.h"
#include "rkllm.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

static LLMHandle  llmHandle = nullptr;
static std::mutex mtx;
static std::condition_variable cv;
static bool       generation_finished = false;

struct InferenceData {
    std::string           output;
    RkllmStreamCallback   cb   = nullptr;
    void                 *ud   = nullptr;
};

static inline void emitToken(RkllmStreamCallback cb,
                             void               *ud,
                             const char         *txt)
{
    if (cb && txt && *txt) cb(txt, ud);
}

static void unifiedCallback(RKLLMResult *result,
                            void        *userdata,
                            LLMCallState state)
{
    auto *data = static_cast<InferenceData *>(userdata);

    switch (state) {
        case RKLLM_RUN_FINISH:
            emitToken(data->cb, data->ud, "[[EOS]]");
            {
                std::lock_guard<std::mutex> lk(mtx);
                generation_finished = true;
            }
            cv.notify_one();
            break;

        case RKLLM_RUN_ERROR:
            std::fprintf(stderr, "rkllm_wrapper: LLM run error\n");
            {
                std::lock_guard<std::mutex> lk(mtx);
                generation_finished = true;
            }
            cv.notify_one();
            break;

        default:
            {
                std::lock_guard<std::mutex> lk(mtx);
                data->output += result->text;
            }
            emitToken(data->cb, data->ud, result->text);
            break;
    }
}

int rkllmwrapper_init(const char *model_path, const RkllmOptions *opts)
{
    if (!opts) return -1;

    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path      = model_path;
    param.is_async        = false;
    param.max_new_tokens  = opts->max_new_tokens;
    param.max_context_len = opts->max_context_len;

    if (opts->top_k            > 0)   param.top_k            = opts->top_k;
    if (opts->top_p            > 0.f) param.top_p            = opts->top_p;
    if (opts->temperature      > 0.f) param.temperature      = opts->temperature;
    if (opts->repeat_penalty   > 0.f) param.repeat_penalty   = opts->repeat_penalty;
    if (opts->frequency_penalty> 0.f) param.frequency_penalty= opts->frequency_penalty;
    if (opts->presence_penalty > 0.f) param.presence_penalty = opts->presence_penalty;
    if (opts->mirostat         >= 0)  param.mirostat         = opts->mirostat;
    if (opts->mirostat_tau     > 0.f) param.mirostat_tau     = opts->mirostat_tau;
    if (opts->mirostat_eta     > 0.f) param.mirostat_eta     = opts->mirostat_eta;
    if (opts->n_keep           > 0)   param.n_keep           = opts->n_keep;

    param.skip_special_token = opts->skip_special_token ? true : false;

    if (opts->num_cpus > 0) {
        int N = opts->num_cpus;
        uint32_t mask = (N >= 32 ? 0xFFFFFFFFu : ((1u << N) - 1));
        param.extend_param.enabled_cpus_mask = mask;
        param.extend_param.enabled_cpus_num  = N;
    }

    int ret = rkllm_init(&llmHandle, &param, unifiedCallback);
    if (ret)
        std::fprintf(stderr, "rkllmwrapper_init: rkllm_init failed (%d)\n", ret);
    return ret;
}

int rkllm_run_ex(const void           *input,
                 int                   input_mode,
                 char                 *output,
                 int                   output_size,
                 size_t                token_count,
                 RkllmStreamCallback   callback,
                 void                 *user_data)
{
    if (!llmHandle) return -1;

    RKLLMInput llmInput{};
    int32_t   *cTokens = nullptr;

    if (input_mode == RKLLM_INPUT_PROMPT) {
        llmInput.input_type   = RKLLM_INPUT_PROMPT;
        llmInput.prompt_input = static_cast<const char*>(input);

    } else if (input_mode == RKLLM_INPUT_TOKEN) {
        cTokens = static_cast<int32_t *>(
                     std::malloc(token_count * sizeof(int32_t)));
        if (!cTokens) {
            std::fprintf(stderr, "rkllm_wrapper: token malloc failed\n");
            return -1;
        }
        std::memcpy(cTokens, input, token_count * sizeof(int32_t));
        llmInput.input_type = RKLLM_INPUT_TOKEN;
        llmInput.token_input = { cTokens, token_count };

    } else {
        return -1;
    }

    auto *data       = new InferenceData{};
    data->cb         = callback;
    data->ud         = user_data;

    {
        std::lock_guard<std::mutex> lk(mtx);
        generation_finished = false;
    }

    RKLLMInferParam inferParams{};
    inferParams.mode = RKLLM_INFER_GENERATE;

    int ret = rkllm_run(llmHandle, &llmInput, &inferParams, data);
    if (ret) {
        delete data;
        if (cTokens) std::free(cTokens);
        return ret;
    }

    {
        std::unique_lock<std::mutex> ulk(mtx);
        cv.wait(ulk, [] { return generation_finished; });
    }

    if (output && output_size > 0) {
        if (data->output.size() >= static_cast<size_t>(output_size)) {
            delete data;
            if (cTokens) std::free(cTokens);
            return -2;   /* buffer too small */
        }
        std::strcpy(output, data->output.c_str());
    }

    delete data;
    if (cTokens) std::free(cTokens);
    return 0;
}

int rkllmwrapper_is_running(void)
{
    if (!llmHandle) return -1;
    return rkllm_is_running(llmHandle);
}

int rkllmwrapper_abort(void)
{
    if (!llmHandle) return -1;

    int ret = rkllm_abort(llmHandle);

    {
        std::lock_guard<std::mutex> lk(mtx);
        generation_finished = true;
    }
    cv.notify_one();
    return ret;
}

void rkllm_destroy_simple(void)
{
    if (llmHandle) {
        rkllm_destroy(llmHandle);
        llmHandle = nullptr;
    }
}
