package legacy

/*
#cgo LDFLAGS: -L/usr/lib/ -lrkllm_wrapper -lrkllmrt -lstdc++
#cgo CXXFLAGS: -std=c++11
#include "rkllm_wrapper.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"unsafe"
)

const (
	RKLLM_INPUT_PROMPT     = 0
	RKLLM_INPUT_TOKEN      = 1
	RKLLM_INPUT_EMBED      = 2
	RKLLM_INPUT_MULTIMODAL = 3
)

func Init(modelPath string, maxNewTokens, maxContextLen int) error {
	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))
	ret := C.rkllm_init_simple(cModelPath, C.int(maxNewTokens), C.int(maxContextLen))
	if ret != 0 {
		return errors.New("failed to initialise RKLLM")
	}
	return nil
}

func RunInferenceWithFifo(prompt, fifo string, inputMode int) (string, error) {
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))
	cFifo := C.CString(fifo)
	defer C.free(unsafe.Pointer(cFifo))

	const bufSize = 8192
	outBuffer := C.malloc(C.size_t(bufSize))
	defer C.free(outBuffer)

	ret := C.rkllm_run_simple_with_fifo(unsafe.Pointer(cPrompt), C.int(inputMode), cFifo, (*C.char)(outBuffer), C.int(bufSize), 0)
	if ret != 0 {
		return "", errors.New("LLM inference error")
	}

	return C.GoString((*C.char)(outBuffer)), nil
}

func RunInferenceWithFifoTokens(tokens []int32, fifo string) (string, error) {
	if len(tokens) == 0 {
		return "", errors.New("tokens slice is empty")
	}
	cFifo := C.CString(fifo)
	defer C.free(unsafe.Pointer(cFifo))

	const bufSize = 8192
	outBuffer := C.malloc(C.size_t(bufSize))
	defer C.free(outBuffer)

	cTokens := (*C.int32_t)(unsafe.Pointer(&tokens[0]))
	tokenCount := C.size_t(len(tokens))

	ret := C.rkllm_run_simple_with_fifo(unsafe.Pointer(cTokens), C.int(RKLLM_INPUT_TOKEN), cFifo, (*C.char)(outBuffer), C.int(bufSize), tokenCount)
	if ret != 0 {
		return "", errors.New("LLM inference error")
	}

	return C.GoString((*C.char)(outBuffer)), nil
}

func Destroy() {
	C.rkllm_destroy_simple()
}
