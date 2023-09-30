package main

import (
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"os"

	"github.com/alexflint/go-arg"
	openai "github.com/sashabaranov/go-openai"
	"github.com/tmc/langchaingo/jsonschema"
	"github.com/tmc/langchaingo/textsplitter"
)

func main() {
	cfg, err := newConfig(os.Args[1:])
	if err != nil {
		fmt.Fprintf(os.Stderr, "unable to parse config: %v\n", err)
		os.Exit(1)
	}

	err = run(context.Background(), cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "application returned an error: %v\n", err)
		os.Exit(1)
	}
}

//go:embed assets
var assets embed.FS

func run(ctx context.Context, cfg *config) error {
	doc, err := assets.ReadFile("assets/rfc8193.txt")
	if err != nil {
		return err
	}

	splitter := textsplitter.NewTokenSplitter(textsplitter.WithChunkSize(500), textsplitter.WithChunkOverlap(50))
	chunks, err := splitter.SplitText(string(doc))
	if err != nil {
		return err
	}

	markdownChunks, err := convertChunksToMarkdown(ctx, cfg, chunks)
	if err != nil {
		return err
	}

	for i, chunk := range markdownChunks {
		fmt.Fprintf(os.Stderr, "Markdown chunk #%d:\n------\n\n%s\n\n------\n", i, chunk)
	}

	return nil
}

type config struct {
	AzureOpenAIKey      string `arg:"--azure-openai-key,env:AZURE_OPENAI_KEY,required" help:"The Azure OpenAI Key"`
	AzureOpenAIEndpoint string `arg:"--azure-openai-endpoint,env:AZURE_OPENAI_ENDPOINT,required" help:"The Azure OpenAI Endpoint"`
}

func newConfig(args []string) (*config, error) {
	cfg := &config{}
	parser, err := arg.NewParser(arg.Config{
		Program:   "text-chunking-gpt",
		IgnoreEnv: false,
	}, cfg)
	if err != nil {
		return nil, err
	}

	err = parser.Parse(args)
	if err != nil {
		return nil, err
	}

	return cfg, nil
}

type input struct {
	Iteration          int      `json:"iteration"`
	RetryLastIteration bool     `json:"retry_last_iteration"`
	MaxIndex           int      `json:"max_index"`
	StartIndex         int      `json:"start_index"`
	EndIndex           int      `json:"end_index"`
	Chunks             []string `json:"chunks"`
}

type output struct {
	Finished       bool   `json:"finished"`
	Store          bool   `json:"store"`
	Markdown       string `json:"markdown"`
	NextStartIndex int    `json:"next_start_index"`
	NextEndIndex   int    `json:"next_end_index"`
}

func convertChunksToMarkdown(ctx context.Context, cfg *config, chunks []string) ([]string, error) {
	openaiConfig := openai.DefaultAzureConfig(cfg.AzureOpenAIKey, cfg.AzureOpenAIEndpoint)
	openaiConfig.APIVersion = "2023-07-01-preview"
	client := openai.NewClientWithConfig(openaiConfig)

	startIndex := 0
	endIndex := 2
	iterationCounter := 0
	retryLastIteration := false
	retryLastIterationCounter := 0
	markdownChunks := []string{}

	for {
		inputChunks := []string{}
		for i := startIndex; i <= endIndex; i++ {
			inputChunks = append(inputChunks, chunks[i])
		}

		inputData := input{
			Iteration:          iterationCounter,
			RetryLastIteration: retryLastIteration,
			MaxIndex:           len(chunks) - 1,
			StartIndex:         startIndex,
			EndIndex:           endIndex,
			Chunks:             inputChunks,
		}
		retryLastIteration = false

		inputJSON, err := json.Marshal(inputData)
		if err != nil {
			return nil, fmt.Errorf("unable to marshal input data: %v", err)
		}

		systemPrompt := `You are an AI receiving chunks of a document in the following format:
{
	"iteration": {
		"type": "integer",
		"description": "The current iteration of the external loop."
	},
	"retry_last_iteration": {
		"type": "boolean",
		"description": "False by default. When received, it means the external loop was unable to unmarshal the JSON in your previous function call. When received, you MUST ensure that the JSON is valid, especially that the markdown text doesn't make it invalid."
	},
	"max_index": {
		"type": "integer",
		"description": "The max index of the chunks. The output function can never have an end_index larger than this.",
	},
	"start_index": {
		"type": "integer",
		"description": "The index of the first chunk in the property chunks."
	},
	"end_index": {
		"type": "integer",
		"description": "The index of the last chunk in the property chunks."
	},
	"chunks": {
		"type": "array",
		"items": {
			"type": "string",
		},
		"description": "The chunks of the document in order from start_index to end_index."
	}
}

You MUST convert the chunks into markdown. The output MUST be contextual part of the text and if you don't have enough text, you must make sure the output function has the property store set to false.
If the property store is set to false in the output function, you must continue to use the same start_index but request a higher end_index than before.

Do your best to clean up headers, footers and other text that doesn't add value to the actual context of the new chunk your generate.

The input will be provided by the user. You MUST only use the function output. You should try to keep the range between next_start_index and next_end_index to 3 chunks, but a maximum of up to 6 chunks.`

		messagesInput := []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: systemPrompt,
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: string(inputJSON),
			},
		}

		functionsInput := []openai.FunctionDefinition{
			{
				Name:        "output",
				Description: "This function drives an external loop that feeds the AI chunks of data based on the responses.",
				Parameters: jsonschema.Definition{
					Type: jsonschema.Object,
					Properties: map[string]jsonschema.Definition{
						"finished": {
							Type:        jsonschema.Boolean,
							Description: "False by default. Set to true to exit the external loop and store all the markdown chunks.",
						},
						"store": {
							Type:        jsonschema.Boolean,
							Description: "True by default. Set to false to indicate that the data provided in the markdown property is empty and you need more chunks to create a contextual makrdown text. If set to true, next_start_index MUST stay the same as the input start_index, while next_end_index MUST be larger than the end_index provided in the input.",
						},
						"markdown": {
							Type:        jsonschema.String,
							Description: "The markdown chunk that should be stored. Leave empty if the property store is set to false. MUST be valid inside a json string.",
						},
						"next_start_index": {
							Type:        jsonschema.Number,
							Description: "The number returned here is what the external loop should send the next time as the start_index. MUST be smaller or equal to the next_end_index. Set to -1 when the property finished is set to true.",
						},
						"next_end_index": {
							Type:        jsonschema.Number,
							Description: "The number returned here is what the external loop should send the next time as the end_index. MUST be larger or equal to the next_start_index. Set to -1 when the property finished is set to true.",
						},
					},
					Required: []string{"finished", "store", "markdown", "next_start_index", "next_end_index"},
				},
			},
		}

		chatCompletionRequest := openai.ChatCompletionRequest{
			Model:     openai.GPT4,
			Messages:  messagesInput,
			Functions: functionsInput,
		}

		res, err := client.CreateChatCompletion(ctx, chatCompletionRequest)
		if err != nil {
			return nil, fmt.Errorf("chat completion failed: %v", err)
		}

		if len(res.Choices) != 1 {
			return nil, fmt.Errorf("res.Choices should be 1 but received: %d", len(res.Choices))
		}

		if res.Choices[0].Message.FunctionCall.Name != "output" {
			return nil, fmt.Errorf("functionCall name not output but: %s", res.Choices[0].Message.FunctionCall.Name)
		}

		fmt.Fprintf(os.Stderr, "Received arguments:\n------\n%s\n------\n", res.Choices[0].Message.FunctionCall.Arguments)

		outputData := output{}
		err = json.Unmarshal([]byte(res.Choices[0].Message.FunctionCall.Arguments), &outputData)
		if err != nil {
			if retryLastIterationCounter >= 1 {
				return nil, fmt.Errorf("unable to unmarshal output data, retry count %d: %v", retryLastIterationCounter, err)
			}
			retryLastIteration = true
			retryLastIterationCounter++
			continue
		}
		retryLastIterationCounter = 0

		if outputData.Finished {
			break
		}

		if outputData.NextEndIndex > inputData.MaxIndex {
			return nil, fmt.Errorf("next end index %d received from output is larger than max index %d", outputData.NextEndIndex, inputData.MaxIndex)
		}

		if outputData.NextStartIndex > outputData.NextEndIndex {
			return nil, fmt.Errorf("next start index %d is larger than next end index %d", outputData.NextStartIndex, outputData.NextEndIndex)
		}

		if outputData.Store {
			markdownChunks = append(markdownChunks, outputData.Markdown)
		}

		startIndex = outputData.NextStartIndex
		endIndex = outputData.NextEndIndex
		iterationCounter++
	}

	return markdownChunks, nil
}
