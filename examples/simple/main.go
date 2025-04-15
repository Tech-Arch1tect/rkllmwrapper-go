package main

import (
	"bufio"
	"bytes"
	"fmt"
	"log"
	"os"
	"strings"
	"syscall"
	"time"
	"unsafe"

	"github.com/tech-arch1tect/rkllmwrapper-go/generated"
)

const RKLLM_INPUT_PROMPT = 0

func main() {
	fifoPath := "/tmp/llm_output_123.fifo"

	if _, err := os.Stat(fifoPath); os.IsNotExist(err) {
		if err := syscall.Mkfifo(fifoPath, 0666); err != nil {
			log.Fatalf("Failed to create FIFO: %v", err)
		}
	}

	file, err := os.OpenFile(fifoPath, os.O_RDWR, os.ModeNamedPipe)
	if err != nil {
		log.Fatalf("Failed to open FIFO: %v", err)
	}
	defer file.Close()

	var fifoClosed bool

	go func() {
		reader := bufio.NewReader(file)
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				fifoClosed = true
				break
			}
			trimmed := strings.TrimSpace(line)
			if trimmed == "[[EOS]]" {
				log.Println("Received EOS marker, ending stream")
				fifoClosed = true
				break
			}
			fmt.Printf("Received chunk: %s", line)
		}
	}()

	modelPath := os.Getenv("RKLLM_MODEL_PATH")
	if modelPath == "" {
		log.Fatalf("RKLLM_MODEL_PATH environment variable is not set")
	}

	if ret := generated.Rkllm_init_simple(modelPath, 4096, 4096); ret != 0 {
		log.Fatalf("Failed to initialise RKLLM, return code: %d", ret)
	}
	defer generated.Rkllm_destroy_simple()

	prompt := "Hello, How are you?"
	cPrompt := []byte(prompt + "\x00")
	bufSize := int32(8192)
	outputBuffer := make([]byte, bufSize)

	ret := generated.Rkllm_run_ex(
		unsafe.Pointer(&cPrompt[0]),
		RKLLM_INPUT_PROMPT,
		outputBuffer,
		bufSize,
		0,
		fifoPath,
	)
	if ret != 0 {
		log.Fatalf("Inference error: return code %d", ret)
	}

	finalOutput := string(bytes.Trim(outputBuffer, "\x00"))

	fmt.Println("LLM Final Output:")
	fmt.Println(finalOutput)

	for !fifoClosed {
		time.Sleep(1 * time.Second)
	}
}
