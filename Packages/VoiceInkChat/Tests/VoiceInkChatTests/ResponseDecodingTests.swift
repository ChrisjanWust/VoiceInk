import Foundation
import Testing
@testable import VoiceInkChat

/// Canned responses shaped like the real APIs' output. Each checks that only the final answer
/// reaches `text`, and that truncation, usage and errors come through.
@Suite struct ResponseDecodingTests {
    private func data(_ json: String) -> Data { Data(json.utf8) }

    // MARK: OpenAI-compatible

    @Test func openAIStringContentWithUsage() throws {
        let result = try OpenAICompatibleChatClient.decodeResult(from: data("""
            {"id": "chatcmpl-1", "object": "chat.completion", "model": "gpt-6-luna-2026-10-01",
             "choices": [{"index": 0, "message": {"role": "assistant", "content": "  The meeting moved to Thursday.\\n",
                          "refusal": null, "annotations": []}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 42, "completion_tokens": 9, "total_tokens": 51,
                       "completion_tokens_details": {"reasoning_tokens": 0, "accepted_prediction_tokens": 0}},
             "service_tier": "default"}
            """))
        #expect(result.text == "The meeting moved to Thursday.")
        #expect(result.finishReason == .stop)
        #expect(!result.isTruncated)
        #expect(result.usage == ChatUsage(inputTokens: 42, outputTokens: 9, reasoningTokens: 0))
    }

    @Test func reasoningContentAndReasoningFieldsAreLeftOut() throws {
        // Together/Fireworks/DeepSeek put reasoning in `reasoning_content`; Groq and Cerebras
        // in `reasoning`.
        let result = try OpenAICompatibleChatClient.decodeResult(from: data("""
            {"model": "deepseek-ai/DeepSeek-V4",
             "choices": [{"message": {"role": "assistant", "content": "Moved to Thursday.",
                          "reasoning_content": "The user wants filler words removed...",
                          "reasoning": "Let me think step by step."}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 50,
                       "completion_tokens_details": {"reasoning_tokens": 41}}}
            """))
        #expect(result.text == "Moved to Thursday.")
        #expect(result.usage?.reasoningTokens == 41)
    }

    @Test func inlineThinkSectionsAreStripped() throws {
        let result = try OpenAICompatibleChatClient.decodeResult(from: data("""
            {"choices": [{"message": {"content": "<think>\\nThe user said um.\\n</think>\\n\\nThe meeting moved to Thursday."},
                          "finish_reason": "stop"}]}
            """))
        #expect(result.text == "The meeting moved to Thursday.")

        // A chat template that opens the tag in the prompt leaves only the closing tag.
        let templated = try OpenAICompatibleChatClient.decodeResult(from: data("""
            {"choices": [{"message": {"content": "Okay, remove fillers.</think>The meeting moved to Thursday."},
                          "finish_reason": "stop"}]}
            """))
        #expect(templated.text == "The meeting moved to Thursday.")
    }

    @Test func mistralChunkedContentKeepsOnlyText() throws {
        let result = try OpenAICompatibleChatClient.decodeResult(from: data("""
            {"id": "a1", "object": "chat.completion", "model": "magistral-medium-latest",
             "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "tool_calls": null,
               "content": [
                 {"type": "thinking", "thinking": [{"type": "text", "text": "Remove um and uh."}]},
                 {"type": "text", "text": "The meeting is "},
                 {"type": "text", "text": "moved to Thursday."}]}}],
             "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50}}
            """))
        #expect(result.text == "The meeting is moved to Thursday.")
    }

    /// Mistral Large 4 and GLM 5.3 always reason, so every reply is a chunk list, even at the
    /// lowest effort. Shaped as Mistral's reasoning docs show it (no Mistral key to record one):
    /// a thinking chunk with nested text chunks, sometimes more than one, then the answer.
    @Test func alwaysReasoningMistralModelsKeepOnlyTheAnswer() throws {
        for model in ["mistral-large-4", "zai-glm-5-3"] {
            let result = try OpenAICompatibleChatClient.decodeResult(from: data("""
                {"id": "b2", "object": "chat.completion", "model": "\(model)",
                 "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant",
                   "content": [
                     {"type": "thinking", "thinking": [{"type": "text", "text": "The user says um."},
                                                      {"type": "text", "text": " Drop fillers."}]},
                     {"type": "thinking", "thinking": [{"type": "text", "text": "Check the day."}]},
                     {"type": "text", "text": "The meeting moved "},
                     {"type": "text", "text": "to Thursday."}]}}],
                 "usage": {"prompt_tokens": 40, "completion_tokens": 90, "total_tokens": 130}}
                """))
            #expect(result.text == "The meeting moved to Thursday.")
            #expect(result.finishReason == .stop)
            #expect(!result.text.contains("Drop fillers"))
        }
    }

    @Test func truncationIsReportedEvenWithNoText() throws {
        let cutOff = try OpenAICompatibleChatClient.decodeResult(from: data("""
            {"choices": [{"message": {"content": "The meeting is moved to"}, "finish_reason": "length"}]}
            """))
        #expect(cutOff.isTruncated)
        #expect(cutOff.text == "The meeting is moved to")

        // A reasoning model that spent the whole budget thinking: null content, length.
        let empty = try OpenAICompatibleChatClient.decodeResult(from: data("""
            {"choices": [{"message": {"content": null, "reasoning_content": "Hmm..."}, "finish_reason": "length"}]}
            """))
        #expect(empty.isTruncated)
        #expect(empty.text == "")

        let mistral = try OpenAICompatibleChatClient.decodeResult(from: data("""
            {"choices": [{"message": {"content": "Partial"}, "finish_reason": "model_length"}]}
            """))
        #expect(mistral.finishReason == .length)

        let unclosed = try OpenAICompatibleChatClient.decodeResult(from: data("""
            {"choices": [{"message": {"content": "<think>I need to think about"}, "finish_reason": "length"}]}
            """))
        #expect(unclosed.text == "")
        #expect(unclosed.isTruncated)
    }

    @Test func emptyAnswerWithoutAReasonIsNoResult() {
        #expect(throws: ChatError.noResultReturned) {
            try OpenAICompatibleChatClient.decodeResult(from: data("""
                {"choices": [{"message": {"content": "   "}, "finish_reason": "stop"}]}
                """))
        }
        #expect(throws: ChatError.noResultReturned) {
            try OpenAICompatibleChatClient.decodeResult(from: data(#"{"choices": []}"#))
        }
    }

    @Test func contentFilterIsReported() throws {
        let result = try OpenAICompatibleChatClient.decodeResult(from: data("""
            {"choices": [{"message": {"content": null}, "finish_reason": "content_filter"}]}
            """))
        #expect(result.finishReason == .contentFilter)
    }

    @Test func malformedBodyIsDecodingError() {
        #expect {
            try OpenAICompatibleChatClient.decodeResult(from: data("<html>Bad gateway</html>"))
        } throws: { error in
            if case ChatError.decodingError = error { return true }
            return false
        }
    }

    // MARK: Anthropic

    @Test func anthropicSkipsThinkingBlocks() throws {
        let result = try AnthropicChatClient.decodeResult(from: data("""
            {"id": "msg_01", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
             "content": [
               {"type": "thinking", "thinking": "The user wants a cleanup.", "signature": "EqQB..."},
               {"type": "redacted_thinking", "data": "EmwKAhgB..."},
               {"type": "text", "text": "The meeting moved "},
               {"type": "text", "text": "to Thursday."}],
             "stop_reason": "end_turn", "stop_sequence": null,
             "usage": {"input_tokens": 120, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                       "output_tokens": 60, "service_tier": "standard"}}
            """))
        #expect(result.text == "The meeting moved to Thursday.")
        #expect(result.finishReason == .stop)
        #expect(result.usage == ChatUsage(inputTokens: 120, outputTokens: 60))
    }

    @Test func anthropicMaxTokensIsTruncation() throws {
        let result = try AnthropicChatClient.decodeResult(from: data("""
            {"content": [{"type": "thinking", "thinking": "Long thought", "signature": "x"}],
             "stop_reason": "max_tokens", "usage": {"input_tokens": 1, "output_tokens": 8192}}
            """))
        #expect(result.isTruncated)
        #expect(result.rawFinishReason == "max_tokens")
        #expect(AnthropicChatClient.mapStopReason("model_context_window_exceeded") == .length)
        #expect(AnthropicChatClient.mapStopReason("refusal") == .contentFilter)
    }

    // MARK: Gemini

    @Test func geminiReadsTheFinalModelOutputAndSkipsThoughts() throws {
        let result = try GeminiChatClient.decodeResult(from: data("""
            {"id": "v1_abc", "status": "completed", "model": "gemini-3.6-flash", "role": "model",
             "created": "2026-11-18T10:00:00Z",
             "steps": [
               {"type": "user_input", "content": [{"type": "text", "text": "um so"}]},
               {"type": "thought", "signature": "CpEB...", "summary": [{"type": "text", "text": "Cleaning up."}]},
               {"type": "model_output", "content": [{"type": "text", "text": "Draft"}]},
               {"type": "model_output", "content": [
                  {"type": "image", "mime_type": "image/png", "data": "iVBOR"},
                  {"type": "text", "text": "The meeting moved "},
                  {"type": "text", "text": "to Thursday."}]}],
             "usage": {"total_input_tokens": 30, "total_output_tokens": 8, "total_thought_tokens": 120,
                       "total_tokens": 158}}
            """))
        #expect(result.text == "The meeting moved to Thursday.")
        #expect(result.finishReason == .stop)
        #expect(result.usage == ChatUsage(inputTokens: 30, outputTokens: 8, reasoningTokens: 120))
    }

    @Test func geminiIncompleteIsTruncation() throws {
        let result = try GeminiChatClient.decodeResult(from: data("""
            {"status": "incomplete", "steps": [{"type": "thought", "summary": []}]}
            """))
        #expect(result.isTruncated)
        #expect(throws: ChatError.noResultReturned) {
            try GeminiChatClient.decodeResult(from: data(#"{"status": "completed", "steps": []}"#))
        }
    }

    // MARK: OpenRouter

    @Test func openRouterKeepsRoutingMetadata() throws {
        let result = try OpenRouterChatClient.decodeResult(from: data("""
            {"id": "gen-1", "provider": "Groq", "model": "openai/gpt-oss-120b",
             "choices": [{"finish_reason": "stop", "native_finish_reason": "stop",
                          "message": {"role": "assistant", "content": "Moved to Thursday.", "reasoning": null}}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 4, "completion_tokens_details": {"reasoning_tokens": 0}},
             "openrouter_metadata": {"attempt": 2, "attempts": [{"provider": "Groq", "status": 200}]}}
            """))
        #expect(result.text == "Moved to Thursday.")
        #expect(result.provider == "Groq")
        // AIEnhancementService logs the routing attempt.
        #expect(result.metadata?["attempt"]?.intValue == 2)
    }

    @Test func openRouterMistralChunksAndTruncation() throws {
        let result = try OpenRouterChatClient.decodeResult(from: data("""
            {"provider": "Mistral", "choices": [{"finish_reason": "length", "message": {"content": [
              {"type": "thinking", "thinking": [{"type": "text", "text": "hmm"}]},
              {"type": "text", "text": "Moved to"}]}}]}
            """))
        #expect(result.text == "Moved to")
        #expect(result.isTruncated)
    }

    // MARK: Ollama

    @Test func ollamaDropsThinking() throws {
        let result = try OllamaChatClient.decodeResult(from: data("""
            {"model": "qwen3:8b", "created_at": "2026-11-18T10:00:00Z",
             "message": {"role": "assistant", "content": "The meeting moved to Thursday.",
                         "thinking": "The user said um..."},
             "done": true, "done_reason": "stop", "total_duration": 1000, "prompt_eval_count": 26, "eval_count": 300}
            """))
        #expect(result.text == "The meeting moved to Thursday.")
        #expect(result.usage == ChatUsage(inputTokens: 26, outputTokens: 300))
        let cutOff = try OllamaChatClient.decodeResult(from: data("""
            {"message": {"role": "assistant", "content": "<think>still going"}, "done": true, "done_reason": "length"}
            """))
        #expect(cutOff.isTruncated)
        #expect(cutOff.text == "")
    }

    // MARK: Bedrock

    @Test func bedrockSkipsReasoningContent() throws {
        let result = try BedrockChatClient.decodeResult(from: data("""
            {"output": {"message": {"role": "assistant", "content": [
                {"reasoningContent": {"reasoningText": {"text": "Thinking...", "signature": "sig"}}},
                {"text": "The meeting moved  \\nto Thursday.  "}]}},
             "stopReason": "end_turn",
             "usage": {"inputTokens": 30, "outputTokens": 12, "totalTokens": 42},
             "metrics": {"latencyMs": 450}}
            """))
        #expect(result.text == "The meeting moved\nto Thursday.")
        #expect(result.usage == ChatUsage(inputTokens: 30, outputTokens: 12))
    }

    @Test func bedrockStopReasons() throws {
        let truncated = try BedrockChatClient.decodeResult(from: data("""
            {"output": {"message": {"content": [{"text": "Partial"}]}}, "stopReason": "max_tokens"}
            """))
        #expect(truncated.isTruncated)
        let blocked = try BedrockChatClient.decodeResult(from: data("""
            {"output": {"message": {"content": []}}, "stopReason": "guardrail_intervened"}
            """))
        #expect(blocked.finishReason == .contentFilter)

        // A guardrail's replacement text comes back as text, so the finish reason must be checked.
        let replaced = try BedrockChatClient.decodeResult(from: data("""
            {"output": {"message": {"content": [{"text": "Sorry, I can't help with that."}]}},
             "stopReason": "guardrail_intervened"}
            """))
        #expect(replaced.text == "Sorry, I can't help with that.")
        #expect(replaced.finishReason == .contentFilter)
        #expect(!replaced.isTruncated)
    }

    @Test func bedrockStripsInlineThinkSections() throws {
        let result = try BedrockChatClient.decodeResult(from: data("""
            {"output": {"message": {"content": [{"text": "<think>Remove the fillers.</think>\\n\\nMoved to Thursday."}]}},
             "stopReason": "end_turn"}
            """))
        #expect(result.text == "Moved to Thursday.")
    }

    // MARK: Error bodies

    @Test func providerErrorBodies() {
        let cases: [(String, String?, String, String?)] = [
            (#"{"error": {"message": "Incorrect API key provided.", "type": "invalid_request_error", "code": "invalid_api_key"}}"#,
             nil, "Incorrect API key provided.", "invalid_request_error"),
            (#"{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}"#,
             nil, "Overloaded", "overloaded_error"),
            (#"{"error": {"code": 400, "message": "thinking_level is not supported", "status": "INVALID_ARGUMENT"}}"#,
             nil, "thinking_level is not supported", "INVALID_ARGUMENT"),
            (#"{"error": "model 'llama9' not found"}"#, nil, "model 'llama9' not found", nil),
            (#"{"message": "The security token included in the request is invalid."}"#,
             "UnrecognizedClientException:http://internal.amazon.com/coral/com.amazon.coral.service/",
             "The security token included in the request is invalid.", "UnrecognizedClientException"),
            (#"{"Message": "Bad model", "__type": "com.amazonaws.bedrock#ValidationException"}"#,
             nil, "Bad model", "ValidationException"),
            (#"{"object": "error", "message": "Invalid model: x", "type": "invalid_model", "code": "1500"}"#,
             nil, "Invalid model: x", "invalid_model"),
            (#"{"detail": "Not Found"}"#, nil, "Not Found", nil),
            ("upstream connect error", nil, "upstream connect error", nil),
            ("", nil, "No error details", nil),
        ]
        for (body, header, message, type) in cases {
            let parsed = ProviderErrorBody.parse(Data(body.utf8), errorTypeHeader: header)
            #expect(parsed.message == message, "\(body)")
            #expect(parsed.errorType == type, "\(body)")
        }
    }
}
