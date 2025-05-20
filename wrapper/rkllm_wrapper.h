#ifndef RKLLM_WRAPPER_H
#define RKLLM_WRAPPER_H

#if defined(__cplusplus)
  #include <cstddef>
  #include <cstdint>
#else
  #include <stddef.h>
  #include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t max_new_tokens;
    int32_t max_context_len;
    int32_t top_k;
    float   top_p;
    float   temperature;
    float   repeat_penalty;
    float   frequency_penalty;
    float   presence_penalty;
    int32_t mirostat;
    float   mirostat_tau;
    float   mirostat_eta;
    int32_t n_keep;
    int     skip_special_token;
    int     num_cpus;
} RkllmOptions;

typedef void (*RkllmStreamCallback)(const char *token, void *user_data);

int  rkllmwrapper_init(const char *model_path, const RkllmOptions   *opts);

int  rkllm_run_ex(const void            *input,
                  int                    input_mode,
                  char                  *output,
                  int                    output_size,
                  size_t                 token_count,
                  RkllmStreamCallback    callback,
                  void                  *user_data);

int  rkllmwrapper_is_running(void);
int  rkllmwrapper_abort     (void);
void rkllm_destroy_simple   (void);

#ifdef __cplusplus
}
#endif
#endif