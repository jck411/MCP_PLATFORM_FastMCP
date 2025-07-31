# OpenAI Orchestrator - Thinking Blocks Support

This document describes the thinking blocks (reasoning) feature added to the OpenAI orchestrator for use with OpenRouter and reasoning-capable models.

## What are Thinking Blocks?

Thinking blocks (also called reasoning tokens) allow AI models to perform internal reasoning before generating their final response. This helps models:
- Think through complex problems step-by-step
- Provide more accurate and well-reasoned answers
- Show their reasoning process (when enabled in logs)

## How It Works

The OpenAI orchestrator automatically detects when reasoning should be enabled based on:

1. **Provider Detection**: Automatically enables for OpenRouter (`openrouter.ai` in base URL)
2. **Model Detection**: Enables for reasoning-capable models (o1, o3, o4, deepseek, claude, qwen, r1)

## Configuration

### Basic Configuration (config.yaml)

```yaml
llm:
  active: "openrouter"
  providers:
    openrouter:
      base_url: "https://openrouter.ai/api/v1"
      model: "deepseek/deepseek-r1"  # A reasoning-capable model
      temperature: 0.7
      max_tokens: 4096
      reasoning:
        effort: "medium"  # "low", "medium", "high"
        # OR
        # max_tokens: 2000  # Explicit token limit
        # exclude: false   # Set to true to use reasoning but not show in logs
```

### Reasoning Configuration Options

- **effort**: `"low"`, `"medium"`, `"high"` - Controls reasoning intensity
- **max_tokens**: Number (e.g., 2000) - Explicit token limit for reasoning
- **exclude**: Boolean - Use reasoning internally but don't include in response/logs

## Logging

When reasoning is enabled, you'll see thinking content in the logs:

```
2025-01-31 12:34:56 - INFO - 🧠 Reasoning/Thinking (Streaming response): The user is asking me to compare 9.11 and 9.9. I need to think about decimal comparison carefully...
```

### Enable Reasoning Logs

Ensure this is set in your config.yaml:

```yaml
chat:
  service:
    logging:
      llm_replies: true  # Must be true to see reasoning logs
      llm_reply_truncate_length: 500  # Adjust as needed
```

## Supported Models

The orchestrator automatically enables reasoning for models containing these keywords:
- `o1`, `o3`, `o4` (OpenAI reasoning models)
- `deepseek` (DeepSeek reasoning models)
- `claude` (Claude models with reasoning)
- `qwen`, `r1` (Other reasoning models)

## Usage Examples

### Using with OpenRouter

1. Set your API key:
   ```bash
   export OPENROUTER_API_KEY="your-key-here"
   ```

2. Update config.yaml:
   ```yaml
   llm:
     active: "openrouter"
     providers:
       openrouter:
         model: "deepseek/deepseek-r1"
         reasoning:
           effort: "high"
   ```

3. The orchestrator will automatically add reasoning to requests when appropriate.

### Compatibility

- **OpenAI Direct**: No reasoning added (maintains compatibility)
- **OpenRouter**: Reasoning automatically enabled for supported models
- **Other Providers**: Only enabled if model name contains reasoning keywords

## Implementation Details

The orchestrator:
1. Detects if reasoning should be enabled via `_supports_reasoning()`
2. Builds reasoning config via `_get_reasoning_config()`
3. Adds `reasoning` parameter to API requests
4. Logs reasoning content from responses (streaming and non-streaming)
5. Preserves backward compatibility with standard OpenAI usage

## Benefits

- **Better Accuracy**: Models can think through problems more carefully
- **Transparency**: See the model's reasoning process in logs
- **No Breaking Changes**: Existing OpenAI usage continues to work
- **Automatic Detection**: No manual configuration required for basic usage

## Troubleshooting

1. **No reasoning in logs**: Ensure `llm_replies: true` in logging config
2. **Model doesn't support reasoning**: Check if your model is reasoning-capable
3. **OpenRouter not detected**: Verify base_url contains "openrouter.ai"
4. **API errors**: Some models may not support the reasoning parameter

## Future Enhancements

Potential future additions:
- Custom reasoning prompts
- Reasoning-specific metrics
- Fine-grained reasoning control per conversation
- Integration with reasoning-specific model features
