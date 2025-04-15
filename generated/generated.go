// THE AUTOGENERATED LICENSE. ALL THE RIGHTS ARE RESERVED BY ROBOTS.

// WARNING: This file has automatically been generated on Tue, 15 Apr 2025 19:54:03 UTC.
// Code generated by https://git.io/c-for-go. DO NOT EDIT.

package generated

/*
#cgo LDFLAGS: -L/usr/lib -lrkllm_wrapper
#include "/usr/include/rkllm_wrapper.h"
#include <stdlib.h>
#include "cgo_helpers.h"
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// Rkllm_init_simple function as declared in include/rkllm_wrapper.h:10
func Rkllm_init_simple(model_path string, max_new_tokens int32, max_context_len int32) int32 {
	model_path = safeString(model_path)
	cmodel_path, cmodel_pathAllocMap := unpackPCharString(model_path)
	cmax_new_tokens, cmax_new_tokensAllocMap := (C.int)(max_new_tokens), cgoAllocsUnknown
	cmax_context_len, cmax_context_lenAllocMap := (C.int)(max_context_len), cgoAllocsUnknown
	__ret := C.rkllm_init_simple(cmodel_path, cmax_new_tokens, cmax_context_len)
	runtime.KeepAlive(cmax_context_lenAllocMap)
	runtime.KeepAlive(cmax_new_tokensAllocMap)
	runtime.KeepAlive(model_path)
	runtime.KeepAlive(cmodel_pathAllocMap)
	__v := (int32)(__ret)
	return __v
}

// Rkllm_run_simple function as declared in include/rkllm_wrapper.h:12
func Rkllm_run_simple(prompt string, input_mode int32, output []byte, output_size int32) int32 {
	prompt = safeString(prompt)
	cprompt, cpromptAllocMap := unpackPCharString(prompt)
	cinput_mode, cinput_modeAllocMap := (C.int)(input_mode), cgoAllocsUnknown
	coutput, coutputAllocMap := (*C.char)(unsafe.Pointer((*sliceHeader)(unsafe.Pointer(&output)).Data)), cgoAllocsUnknown
	coutput_size, coutput_sizeAllocMap := (C.int)(output_size), cgoAllocsUnknown
	__ret := C.rkllm_run_simple(cprompt, cinput_mode, coutput, coutput_size)
	runtime.KeepAlive(coutput_sizeAllocMap)
	runtime.KeepAlive(coutputAllocMap)
	runtime.KeepAlive(cinput_modeAllocMap)
	runtime.KeepAlive(prompt)
	runtime.KeepAlive(cpromptAllocMap)
	__v := (int32)(__ret)
	return __v
}

// Rkllm_run_simple_with_fifo function as declared in include/rkllm_wrapper.h:14
func Rkllm_run_simple_with_fifo(input unsafe.Pointer, input_mode int32, fifo string, output []byte, output_size int32, token_count uint64) int32 {
	cinput, cinputAllocMap := input, cgoAllocsUnknown
	cinput_mode, cinput_modeAllocMap := (C.int)(input_mode), cgoAllocsUnknown
	fifo = safeString(fifo)
	cfifo, cfifoAllocMap := unpackPCharString(fifo)
	coutput, coutputAllocMap := (*C.char)(unsafe.Pointer((*sliceHeader)(unsafe.Pointer(&output)).Data)), cgoAllocsUnknown
	coutput_size, coutput_sizeAllocMap := (C.int)(output_size), cgoAllocsUnknown
	ctoken_count, ctoken_countAllocMap := (C.size_t)(token_count), cgoAllocsUnknown
	__ret := C.rkllm_run_simple_with_fifo(cinput, cinput_mode, cfifo, coutput, coutput_size, ctoken_count)
	runtime.KeepAlive(ctoken_countAllocMap)
	runtime.KeepAlive(coutput_sizeAllocMap)
	runtime.KeepAlive(coutputAllocMap)
	runtime.KeepAlive(fifo)
	runtime.KeepAlive(cfifoAllocMap)
	runtime.KeepAlive(cinput_modeAllocMap)
	runtime.KeepAlive(cinputAllocMap)
	__v := (int32)(__ret)
	return __v
}

// Rkllm_destroy_simple function as declared in include/rkllm_wrapper.h:16
func Rkllm_destroy_simple() {
	C.rkllm_destroy_simple()
}
