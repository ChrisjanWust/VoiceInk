import Foundation
import Testing
@testable import VoiceInkChat

/// Golden request bodies and headers for every client. A Default request (no reasoning,
/// temperature or extras) is compared against the body LLMkit sends today, so switching a call
/// site over doesn't change what goes on the wire.
@Suite struct RequestEncodingTests {
    private func expectBody(_ request: URLRequest, _ expected: String, sourceLocation: SourceLocation = #_sourceLocation) throws {
        let actual = try bodyJSON(request)
        let wanted = JSONValue(json: expected)
        #expect(actual == wanted, "\n\(actual.prettyString)\n!=\n\(wanted.prettyString)", sourceLocation: sourceLocation)
    }

    // MARK: OpenAI-compatible

    @Test func openAIDefaultMatchesLLMkitWithTemperatureLeftOut() throws {
        let client = OpenAICompatibleChatClient(apiKey: " sk-test ", provider: .openAI)
        let request = try client.makeURLRequest(dictation)
        #expect(request.url?.absoluteString == "https://api.openai.com/v1/chat/completions")
        #expect(request.httpMethod == "POST")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer sk-test")
        #expect(request.value(forHTTPHeaderField: "Content-Type") == "application/json")
        try expectBody(request, """
            {"model": "test-model", "stream": false, "messages": [
              {"role": "system", "content": "Clean up the transcript. Reply with the cleaned text only."},
              {"role": "user", "content": "um so the meeting is uh moved to thursday"}]}
            """)
    }

    @Test func openAIWithEveryField() throws {
        var chat = dictation
        chat.temperature = 0.3
        chat.maxTokens = 512
        chat.reasoning = .effort("low")
        chat.serviceTier = "priority"
        chat.extraBody = ["verbosity": "low"]
        chat.extraHeaders = ["OpenAI-Organization": "org-1"]
        let request = try OpenAICompatibleChatClient(apiKey: "k", provider: .openAI).makeURLRequest(chat)
        #expect(request.value(forHTTPHeaderField: "OpenAI-Organization") == "org-1")
        try expectBody(request, """
            {"model": "test-model", "stream": false, "temperature": 0.3, "max_completion_tokens": 512,
             "reasoning_effort": "low", "service_tier": "priority", "verbosity": "low",
             "messages": [
              {"role": "system", "content": "Clean up the transcript. Reply with the cleaned text only."},
              {"role": "user", "content": "um so the meeting is uh moved to thursday"}]}
            """)
    }

    @Test func extraBodyOverridesAndSendsNull() throws {
        var chat = dictation
        chat.temperature = 0.3
        chat.extraBody = ["temperature": .null, "reasoning_format": "hidden"]
        let body = try OpenAICompatibleChatClient(apiKey: "k", provider: .cerebras).body(for: chat)
        #expect(body["temperature"] == .null)
        #expect(body["reasoning_format"] == "hidden")
        let encoded = String(data: try JSONValue.object(body).encoded(), encoding: .utf8)!
        #expect(encoded.contains("\"temperature\":null"))
    }

    @Test func disabledReasoningIsEffortNone() throws {
        var chat = dictation
        chat.reasoning = .disabled
        let body = try OpenAICompatibleChatClient(apiKey: "k", provider: .groq).body(for: chat)
        #expect(body["reasoning_effort"] == "none")
    }

    @Test func providerPresetsUseTheirEndpointsAndTokenFields() throws {
        var chat = dictation
        chat.maxTokens = 100
        let expected: [(OpenAICompatibleProvider, String, String)] = [
            (.openAI, "https://api.openai.com/v1/chat/completions", "max_completion_tokens"),
            (.groq, "https://api.groq.com/openai/v1/chat/completions", "max_completion_tokens"),
            (.cerebras, "https://api.cerebras.ai/v1/chat/completions", "max_completion_tokens"),
            (.mistral, "https://api.mistral.ai/v1/chat/completions", "max_tokens"),
            (.together, "https://api.together.xyz/v1/chat/completions", "max_tokens"),
            (.gmiCloud, "https://api.gmi-serving.com/v1/chat/completions", "max_tokens"),
            (.baseten, "https://inference.baseten.co/v1/chat/completions", "max_tokens"),
            (.fireworks, "https://api.fireworks.ai/inference/v1/chat/completions", "max_tokens"),
            (.vercelAIGateway, "https://ai-gateway.vercel.sh/v1/chat/completions", "max_tokens"),
            (.custom(endpoint: URL(string: "http://localhost:11434/v1/chat/completions")!),
             "http://localhost:11434/v1/chat/completions", "max_tokens"),
        ]
        for (provider, url, field) in expected {
            let client = OpenAICompatibleChatClient(apiKey: "k", provider: provider)
            #expect(try client.makeURLRequest(chat).url?.absoluteString == url)
            #expect(try client.body(for: chat)[field] == 100, "\(provider.name)")
        }
    }

    @Test func openAIRejectsReasoningItCantExpress() {
        var chat = dictation
        chat.reasoning = .budget(tokens: 1000)
        #expect(throws: ChatError.self) { try OpenAICompatibleChatClient(apiKey: "k", provider: .openAI).body(for: chat) }
        chat.reasoning = .enabled
        #expect(throws: ChatError.self) { try OpenAICompatibleChatClient(apiKey: "k", provider: .openAI).body(for: chat) }
    }

    @Test func blankKeyIsMissingAPIKey() {
        #expect(throws: ChatError.missingAPIKey) {
            try OpenAICompatibleChatClient(apiKey: "  ", provider: .openAI).makeURLRequest(dictation)
        }
        #expect(throws: ChatError.missingAPIKey) { try AnthropicChatClient(apiKey: "").makeURLRequest(dictation) }
        #expect(throws: ChatError.missingAPIKey) { try GeminiChatClient(apiKey: "").makeURLRequest(dictation, version: .v1) }
        #expect(throws: ChatError.missingAPIKey) {
            try OpenRouterChatClient(apiKey: "").makeURLRequest(dictation, options: .init())
        }
        #expect(throws: ChatError.missingAPIKey) {
            try BedrockChatClient(apiKey: "Bearer ", region: "us-east-1").makeURLRequest(dictation)
        }
    }

    // MARK: Anthropic

    @Test func anthropicDefaultMatchesLLMkit() throws {
        var chat = dictation
        chat.model = "claude-haiku-5-5"
        let request = try AnthropicChatClient(apiKey: "sk-ant").makeURLRequest(chat)
        #expect(request.url?.absoluteString == "https://api.anthropic.com/v1/messages")
        #expect(request.value(forHTTPHeaderField: "x-api-key") == "sk-ant")
        #expect(request.value(forHTTPHeaderField: "anthropic-version") == "2023-06-01")
        #expect(request.value(forHTTPHeaderField: "anthropic-beta") == nil)
        try expectBody(request, """
            {"model": "claude-haiku-5-5", "max_tokens": 8192,
             "system": "Clean up the transcript. Reply with the cleaned text only.",
             "messages": [{"role": "user", "content": "um so the meeting is uh moved to thursday"}]}
            """)
    }

    @Test func anthropicSystemMessagesAreJoinedWhenNoSystemPrompt() throws {
        let chat = ChatRequest(
            model: "m",
            messages: [.system("One."), .user("Hi"), .system("Two."), .assistant("Hello"), .user("Bye")]
        )
        let body = try AnthropicChatClient.body(for: chat)
        #expect(body["system"] == "One.\nTwo.")
        #expect(body["messages"]?.arrayValue?.count == 3)
    }

    @Test func anthropicEffortUsesAdaptiveThinking() throws {
        var chat = dictation
        chat.reasoning = .effort("low")
        chat.serviceTier = "auto"
        let request = try AnthropicChatClient(apiKey: "k", betas: ["a-2026", "b-2026"]).makeURLRequest(chat)
        #expect(request.value(forHTTPHeaderField: "anthropic-beta") == "a-2026,b-2026")
        try expectBody(request, """
            {"model": "test-model", "max_tokens": 8192, "service_tier": "auto",
             "system": "Clean up the transcript. Reply with the cleaned text only.",
             "thinking": {"type": "adaptive", "display": "omitted"},
             "output_config": {"effort": "low"},
             "messages": [{"role": "user", "content": "um so the meeting is uh moved to thursday"}]}
            """)
    }

    @Test func anthropicBudgetAddsToMaxTokensAndDisabledIsExplicit() throws {
        var chat = dictation
        chat.maxTokens = 1000
        chat.reasoning = .budget(tokens: 2048)
        var body = try AnthropicChatClient.body(for: chat)
        #expect(body["max_tokens"] == 3048)
        #expect(body["thinking"] == ["type": "enabled", "budget_tokens": 2048])

        chat.reasoning = .disabled
        body = try AnthropicChatClient.body(for: chat)
        #expect(body["max_tokens"] == 1000)
        #expect(body["thinking"] == ["type": "disabled"])

        chat.extraBody = ["thinking": ["type": "between_tools"]]
        #expect(try AnthropicChatClient.body(for: chat)["thinking"] == ["type": "between_tools"])
    }

    // MARK: Gemini

    @Test func geminiDefaultMatchesLLMkit() throws {
        var chat = dictation
        chat.model = "gemini-3.6-flash"
        chat.messages = [.user("Hi"), .assistant("Hello"), .user("   "), .user("Fix this")]
        let client = GeminiChatClient(apiKey: "AIza-test")
        let request = try client.makeURLRequest(chat, version: .v1)
        #expect(request.url?.absoluteString == "https://generativelanguage.googleapis.com/v1/interactions")
        #expect(try client.makeURLRequest(chat, version: .v1beta).url?.absoluteString
            == "https://generativelanguage.googleapis.com/v1beta/interactions")
        #expect(request.value(forHTTPHeaderField: "x-goog-api-key") == "AIza-test")
        try expectBody(request, """
            {"model": "gemini-3.6-flash", "store": false,
             "system_instruction": "Clean up the transcript. Reply with the cleaned text only.",
             "input": [
               {"type": "user_input", "content": [{"type": "text", "text": "Hi"}]},
               {"type": "model_output", "content": [{"type": "text", "text": "Hello"}]},
               {"type": "user_input", "content": [{"type": "text", "text": "Fix this"}]}]}
            """)
    }

    @Test func geminiThinkingLevelPassesXHighThrough() throws {
        var chat = dictation
        chat.model = "models/gemini-4-argon"
        chat.reasoning = .effort("xhigh")
        chat.maxTokens = 2000
        chat.serviceTier = "priority"
        let body = try GeminiChatClient(apiKey: "k").body(for: chat)
        #expect(body["model"] == "gemini-4-argon")
        #expect(body["service_tier"] == "priority")
        #expect(body["generation_config"] == ["thinking_level": "xhigh", "max_output_tokens": 2000])
    }

    @Test func geminiRejectsWhatItCantExpress() {
        for reasoning: ChatRequest.Reasoning in [.disabled, .enabled, .budget(tokens: 100)] {
            var chat = dictation
            chat.reasoning = reasoning
            #expect(throws: ChatError.self) { try GeminiChatClient(apiKey: "k").body(for: chat) }
        }
        let prefilled = ChatRequest(model: "m", messages: [.user("Hi"), .assistant("Sure,")])
        #expect(throws: ChatError.self) { try GeminiChatClient(apiKey: "k").body(for: prefilled) }
        let empty = ChatRequest(model: "m", system: "S", messages: [.user("  ")])
        #expect(throws: ChatError.self) { try GeminiChatClient(apiKey: "k").body(for: empty) }
    }

    @Test func geminiVersionFromModelLists() {
        let v1: Set<String> = ["gemini-3.6-flash", "gemini-4-argon"]
        #expect(GeminiAPIVersion.forModel("gemini-3.6-flash", v1Models: v1) == .v1)
        #expect(GeminiAPIVersion.forModel("models/gemini-4-argon", v1Models: v1) == .v1)
        // Not on v1's list, with or without "-preview" in the name.
        #expect(GeminiAPIVersion.forModel("gemini-flash-latest", v1Models: v1) == .v1beta)
        #expect(GeminiAPIVersion.forModel("gemini-3.1-pro-preview", v1Models: v1) == .v1beta)
    }

    // MARK: OpenRouter

    @Test func openRouterMatchesLLMkitLowLatencyRequest() throws {
        var chat = dictation
        chat.model = "openai/gpt-6-luna:nitro"
        chat.temperature = 0.3
        let client = OpenRouterChatClient(
            apiKey: "sk-or",
            appReferer: URL(string: "https://tryvoiceink.com"),
            appTitle: "VoiceInk"
        )
        let options = OpenRouterChatClient.Options(
            reasoning: .init(effort: "none", exclude: true),
            provider: .init(
                sort: .throughput,
                preferredMaxLatency: .init(p90: 2.5),
                requireParameters: true,
                allowFallbacks: true
            )
        )
        let request = try client.makeURLRequest(chat, options: options)
        #expect(request.url?.absoluteString == "https://openrouter.ai/api/v1/chat/completions")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer sk-or")
        #expect(request.value(forHTTPHeaderField: "X-OpenRouter-Metadata") == "enabled")
        #expect(request.value(forHTTPHeaderField: "HTTP-Referer") == "https://tryvoiceink.com")
        #expect(request.value(forHTTPHeaderField: "X-OpenRouter-Title") == "VoiceInk")
        try expectBody(request, """
            {"model": "openai/gpt-6-luna:nitro", "stream": false, "temperature": 0.3,
             "reasoning": {"effort": "none", "exclude": true},
             "provider": {"sort": "throughput", "preferred_max_latency": {"p90": 2.5},
                          "require_parameters": true, "allow_fallbacks": true},
             "messages": [
              {"role": "system", "content": "Clean up the transcript. Reply with the cleaned text only."},
              {"role": "user", "content": "um so the meeting is uh moved to thursday"}]}
            """)
    }

    @Test func openRouterMapsUnifiedReasoningAndWiderPreferences() throws {
        var chat = dictation
        chat.reasoning = .budget(tokens: 512)
        let options = OpenRouterChatClient.Options(
            provider: .init(order: ["groq", "cerebras"], only: ["groq"], ignore: ["x"], quantizations: ["fp8"],
                            dataCollection: "deny", zdr: true)
        )
        let body = try OpenRouterChatClient.body(for: chat, options: options)
        #expect(body["reasoning"] == ["max_tokens": 512, "exclude": true])
        #expect(body["provider"] == JSONValue(json: """
            {"order": ["groq", "cerebras"], "only": ["groq"], "ignore": ["x"], "quantizations": ["fp8"],
             "data_collection": "deny", "zdr": true}
            """))

        chat.reasoning = .disabled
        #expect(try OpenRouterChatClient.body(for: chat, options: .init())["reasoning"]
            == ["enabled": false, "exclude": true])
        chat.reasoning = .effort("high")
        // An explicit OpenRouter reasoning object wins over the unified setting.
        #expect(try OpenRouterChatClient.body(for: chat, options: .init(reasoning: .init(enabled: true)))["reasoning"]
            == ["enabled": true])
    }

    // MARK: Ollama

    @Test func ollamaThinkTakesBoolOrLevel() throws {
        var chat = dictation
        chat.model = "gpt-oss:20b"
        chat.temperature = 0.3
        chat.reasoning = .effort("high")
        let client = OllamaChatClient(keepAlive: "30m")
        let request = try client.makeURLRequest(chat)
        #expect(request.url?.absoluteString == "http://localhost:11434/api/chat")
        #expect(request.value(forHTTPHeaderField: "Authorization") == nil)
        try expectBody(request, """
            {"model": "gpt-oss:20b", "stream": false, "think": "high", "keep_alive": "30m",
             "options": {"temperature": 0.3},
             "messages": [
              {"role": "system", "content": "Clean up the transcript. Reply with the cleaned text only."},
              {"role": "user", "content": "um so the meeting is uh moved to thursday"}]}
            """)

        chat.reasoning = .disabled
        chat.temperature = nil
        var body = try OllamaChatClient().body(for: chat)
        #expect(body["think"] == false)
        #expect(body["options"] == nil)
        chat.reasoning = .enabled
        chat.maxTokens = 64
        body = try OllamaChatClient().body(for: chat)
        #expect(body["think"] == true)
        #expect(body["options"] == ["num_predict": 64])
        chat.reasoning = .budget(tokens: 10)
        #expect(throws: ChatError.self) { try OllamaChatClient().body(for: chat) }
    }

    @Test func ollamaSendsAnOptionalBearerKey() throws {
        let request = try OllamaChatClient(baseURL: URL(string: "https://ollama.example.com")!, apiKey: "ok")
            .makeURLRequest(dictation)
        #expect(request.url?.absoluteString == "https://ollama.example.com/api/chat")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer ok")
    }

    // MARK: Bedrock

    @Test func bedrockDefaultMatchesInAppClient() throws {
        var chat = dictation
        chat.model = " global.anthropic.claude-sonnet-5-5 "
        let request = try BedrockChatClient(apiKey: "Bearer test-bedrock-key", region: " US-EAST-2 ").makeURLRequest(chat)
        #expect(request.url?.absoluteString
            == "https://bedrock-runtime.us-east-2.amazonaws.com/model/global.anthropic.claude-sonnet-5-5/converse")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer test-bedrock-key")
        #expect(request.value(forHTTPHeaderField: "Accept") == "application/json")
        try expectBody(request, """
            {"system": [{"text": "Clean up the transcript. Reply with the cleaned text only."}],
             "messages": [{"role": "user", "content": [{"text": "um so the meeting is uh moved to thursday"}]}]}
            """)
    }

    @Test func bedrockEncodesModelIDsAndARNsInThePath() throws {
        let arn = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.amazon.nova-2-lite-v1:0"
        let url = try BedrockChatClient.endpoint(region: "us-east-1", modelID: arn)
        #expect(url.absoluteString.hasSuffix(
            "/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3Ainference-profile%2Fus.amazon.nova-2-lite-v1%3A0/converse"))
        #expect(throws: ChatError.self) {
            try BedrockChatClient(apiKey: "k", region: "mars").makeURLRequest(dictation)
        }
    }

    @Test func bedrockMergesSameRoleTurnsAndDropsALeadingAssistant() throws {
        let chat = ChatRequest(
            model: "qwen.qwen3-next-80b-a3b",
            messages: [.assistant("Hi!"), .user("One"), .user("Two"), .system("Ignored here"), .assistant("Ok")]
        )
        let body = try BedrockChatClient.body(for: chat)
        #expect(body["messages"] == JSONValue(json: """
            [{"role": "user", "content": [{"text": "One\\n\\nTwo"}]},
             {"role": "assistant", "content": [{"text": "Ok"}]}]
            """))
        #expect(body["system"] == [["text": "Ignored here"]])
    }

    @Test func bedrockReasoningFieldsPerModelFamily() throws {
        func fields(_ model: String, _ reasoning: ChatRequest.Reasoning) throws -> JSONValue? {
            var chat = dictation
            chat.model = model
            chat.reasoning = reasoning
            return try BedrockChatClient.body(for: chat)["additionalModelRequestFields"]
        }
        #expect(try fields("global.anthropic.claude-sonnet-5-5", .effort("low")) == JSONValue(json: """
            {"thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "low"}}
            """))
        #expect(try fields("us.anthropic.claude-haiku-4-5", .disabled) == ["thinking": ["type": "disabled"]])
        #expect(try fields("openai.gpt-oss-120b-1:0", .effort("low")) == ["reasoning_effort": "low"])
        #expect(try fields("global.openai.gpt-6-luna", .effort("medium")) == ["reasoning": ["effort": "medium"]])
        #expect(try fields("global.openai.gpt-6-luna", .disabled) == ["reasoning": ["effort": "none"]])
        #expect(try fields("global.amazon.nova-2-lite-v1:0", .effort("low"))
            == ["reasoningConfig": ["type": "enabled", "maxReasoningEffort": "low"]])
        #expect(throws: ChatError.self) { try fields("qwen.qwen3-next-80b-a3b", .effort("low")) }
        #expect(throws: ChatError.self) { try fields("openai.gpt-oss-120b-1:0", .budget(tokens: 100)) }

        var chat = dictation
        chat.model = "us.anthropic.claude-haiku-4-5"
        chat.reasoning = .budget(tokens: 1024)
        chat.serviceTier = "priority"
        chat.temperature = 0.2
        let body = try BedrockChatClient.body(
            for: chat,
            additionalModelRequestFields: ["anthropic_beta": ["x"]]
        )
        #expect(body["inferenceConfig"] == ["maxTokens": 9216, "temperature": 0.2])
        #expect(body["serviceTier"] == ["type": "priority"])
        #expect(body["additionalModelRequestFields"] == JSONValue(json: """
            {"thinking": {"type": "enabled", "budget_tokens": 1024}, "anthropic_beta": ["x"]}
            """))
    }
}
