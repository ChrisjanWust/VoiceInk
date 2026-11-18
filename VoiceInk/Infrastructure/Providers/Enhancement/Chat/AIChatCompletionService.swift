import Foundation
import LLMkit
import VoiceInkChat

struct AIChatCompletionResult: Sendable {
    let text: String
    let openRouterCompletion: OpenRouterCompletion?
}

extension AIService {
    func performChatCompletion(
        provider: AIProvider,
        modelName: String?,
        messages: [ChatRequest.Message],
        systemPrompt: String? = nil,
        localUserPrompt: String? = nil,
        timeout: TimeInterval = 30,
        retryOnTimeout: Bool = false
    ) async throws -> AIChatCompletionResult {
        let resolvedModel = EnhancementModelMigration.resolvedModel(
            modelName?.isEmpty == false ? modelName! : selectedModel(for: provider),
            for: provider
        )
        let reasoningLevel = EnhancementReasoningSettings.savedLevel(for: provider, model: resolvedModel)
        if provider == .ollama, reasoningLevel != nil {
            // Only when a level is saved, so Default requests do no extra work.
            await loadOllamaThinkingMetadataIfNeeded(for: resolvedModel)
        }
        let fastMode = FastModeConfig.parameters(
            for: provider,
            modelName: resolvedModel,
            enabled: EnhancementFastModeSettings.isEnabled(for: provider, model: resolvedModel)
        )
        let requestTimeout = ReasoningConfig.requestTimeout(
            timeout,
            retryOnTimeout: retryOnTimeout,
            level: ReasoningConfig.effectiveLevel(
                reasoningLevel,
                for: provider,
                modelName: resolvedModel,
                modelInfo: reasoningModelInfo(for: provider, model: resolvedModel)
            )
        )
        .raised(toAtLeast: provider == .bedrock ? BedrockModelCatalog.minimumTimeout(forModel: resolvedModel) : nil)
        let llmkitMessages = messages.map { ChatMessage(role: $0.role.rawValue, content: $0.content) }

        if provider == .mistral,
            let reasoningEffort = ReasoningConfig.mistralReasoningEffort(
                for: resolvedModel,
                level: reasoningLevel,
                modelInfo: reasoningModelInfo(for: provider, model: resolvedModel)
            )
        {
            // Mistral returns reasoning as chunked `content`, which LLMkit can't decode, so a
            // picked level uses the in-app client. Default stays on the LLMkit path below.
            var request = MistralChatRequest(model: resolvedModel, messages: llmkitMessages, systemPrompt: systemPrompt)
            request.reasoningEffort = reasoningEffort
            let completion = try await MistralChatClient.chatCompletion(
                apiKey: try chatAPIKey(for: provider, modelName: resolvedModel),
                request: request,
                timeout: requestTimeout.seconds,
                retryOnTimeout: requestTimeout.retryOnTimeout
            )
            guard !completion.wasTruncated else { throw EnhancementError.outputTruncated }
            return AIChatCompletionResult(text: completion.text, openRouterCompletion: nil)
        }

        var request = ChatRequest(
            model: resolvedModel,
            system: systemPrompt,
            messages: messages,
            timeout: requestTimeout.seconds,
            retryOnTimeout: requestTimeout.retryOnTimeout
        )
        let client: any ChatClient

        // Groups not yet on VoiceInkChat return from their case.
        let result: String
        var openRouterCompletion: OpenRouterCompletion? = nil
        switch provider {
        case .gemini:
            request.reasoning = ReasoningConfig.geminiThinkingLevel(for: resolvedModel, level: reasoningLevel)
                .map(ChatRequest.Reasoning.effort)
            client = GeminiChatClient(
                apiKey: try chatAPIKey(for: provider, modelName: resolvedModel),
                apiVersion: .forModel(resolvedModel)
            )
        case .anthropic:
            if let reasoning = ReasoningConfig.anthropicReasoning(for: resolvedModel, level: reasoningLevel) {
                // LLMkit can't send `thinking` or `output_config`, so a picked level uses the
                // in-app client. Default stays on LLMkit, unchanged.
                var request = AnthropicMessagesRequest(
                    model: resolvedModel,
                    messages: llmkitMessages,
                    systemPrompt: systemPrompt
                )
                reasoning.apply(to: &request)
                let completion = try await AnthropicMessagesClient.createMessage(
                    apiKey: try chatAPIKey(for: provider, modelName: resolvedModel),
                    request: request,
                    timeout: requestTimeout.seconds,
                    retryOnTimeout: requestTimeout.retryOnTimeout
                )
                guard !completion.wasTruncated else { throw EnhancementError.outputTruncated }
                result = completion.text
            } else {
                result = try await AnthropicLLMClient.chatCompletion(
                    apiKey: try chatAPIKey(for: provider, modelName: resolvedModel),
                    model: resolvedModel,
                    messages: llmkitMessages,
                    systemPrompt: systemPrompt,
                    timeout: requestTimeout.seconds,
                    retryOnTimeout: requestTimeout.retryOnTimeout
                )
            }
            return AIChatCompletionResult(text: result, openRouterCompletion: openRouterCompletion)
        case .openRouter:
            let policy = OpenRouterRequestPolicy.lowLatency(
                modelName: resolvedModel,
                modelMetadata: openRouterModelMetadata(for: resolvedModel),
                reasoningLevel: reasoningLevel
            )
            let completion = try await OpenRouterClient.chatCompletion(
                apiKey: try chatAPIKey(for: provider, modelName: resolvedModel),
                model: policy.model,
                messages: llmkitMessages,
                systemPrompt: systemPrompt,
                temperature: policy.temperature,
                reasoning: policy.reasoning,
                provider: policy.provider,
                includeRouterMetadata: true,
                appReferer: URL(string: "https://tryvoiceink.com"),
                appTitle: "VoiceInk",
                timeout: requestTimeout.seconds,
                retryOnTimeout: requestTimeout.retryOnTimeout
            )
            guard !OpenRouterRequestPolicy.outputWasTruncated(finishReason: completion.finishReason) else {
                throw EnhancementError.outputTruncated
            }
            result = completion.text
            openRouterCompletion = completion
            return AIChatCompletionResult(text: result, openRouterCompletion: openRouterCompletion)
        case .bedrock:
            result = try await bedrockChatCompletion(
                apiKey: try chatAPIKey(for: provider, modelName: resolvedModel),
                model: resolvedModel,
                messages: llmkitMessages,
                systemPrompt: systemPrompt,
                reasoning: BedrockReasoning.fields(
                    for: reasoningLevel,
                    model: reasoningModelInfo(for: .bedrock, model: resolvedModel).bedrockModel
                ),
                timeout: requestTimeout.seconds,
                retryOnTimeout: requestTimeout.retryOnTimeout
            )
            return AIChatCompletionResult(text: result, openRouterCompletion: openRouterCompletion)
        case .custom:
            guard
                let customConfiguration = CustomAIProviderManager.shared.requestConfiguration(forModel: resolvedModel),
                let endpoint = URL(string: customConfiguration.baseURL)
            else {
                throw EnhancementError.notConfigured
            }
            // Saved under the display name; sent as the model name the endpoint expects.
            request.model = customConfiguration.modelName
            request.temperature = ReasoningConfig.temperature(for: .custom, modelName: resolvedModel, level: reasoningLevel)
            request.reasoning = ReasoningConfig.getReasoningParameter(
                for: .custom,
                modelName: resolvedModel,
                level: reasoningLevel
            ).map(ChatRequest.Reasoning.effort)
            client = OpenAICompatibleChatClient(apiKey: customConfiguration.apiKey, provider: .custom(endpoint: endpoint))
        case .voiceInkRefine:
            throw EnhancementError.customError(
                String(localized: "VoiceInk Refine only supports transcript cleanup.")
            )
        case .ollama:
            result = try await enhanceWithOllama(
                text: localUserPrompt ?? chatPrompt(from: llmkitMessages),
                systemPrompt: systemPrompt ?? "",
                model: resolvedModel,
                think: ReasoningConfig.ollamaThink(
                    level: reasoningLevel,
                    modelName: resolvedModel,
                    modelInfo: reasoningModelInfo(for: .ollama, model: resolvedModel)
                ),
                timeout: requestTimeout.seconds,
                retryOnTimeout: requestTimeout.retryOnTimeout
            )
            return AIChatCompletionResult(text: result, openRouterCompletion: openRouterCompletion)
        case .localCLI:
            result = try await enhanceWithLocalCLI(
                systemPrompt: systemPrompt ?? "",
                userPrompt: localUserPrompt ?? chatPrompt(from: llmkitMessages)
            )
            return AIChatCompletionResult(text: result, openRouterCompletion: openRouterCompletion)
        default:
            guard let chatProvider = provider.openAICompatibleChatProvider else {
                throw EnhancementError.notConfigured
            }
            request.temperature = ReasoningConfig.temperature(for: provider, modelName: resolvedModel, level: reasoningLevel)
            request.reasoning = ReasoningConfig.getReasoningParameter(
                for: provider,
                modelName: resolvedModel,
                level: reasoningLevel
            ).map(ChatRequest.Reasoning.effort)
            request.extraBody = ReasoningConfig.extraBodyFields(for: provider, modelName: resolvedModel, level: reasoningLevel)
            client = OpenAICompatibleChatClient(
                apiKey: try chatAPIKey(for: provider, modelName: resolvedModel),
                provider: chatProvider
            )
        }

        // Fast mode's fields go last, so they win over anything else in the body.
        request.extraBody.merge(fastMode?.bodyFields ?? [:]) { _, fast in fast }
        let completion = try await client.complete(request)
        try Self.requireFinishedAnswer(completion)
        return AIChatCompletionResult(text: completion.text, openRouterCompletion: nil)
    }

    /// Only a finished answer is used. A cut-off, filtered or failed one throws, so it's never
    /// pasted: a guardrail's replacement text, for example, comes back with text.
    private static func requireFinishedAnswer(_ result: ChatResult) throws {
        switch result.finishReason {
        case .stop:
            return
        case .length:
            throw EnhancementError.outputTruncated
        case .contentFilter:
            throw EnhancementError.customError(String(localized: "The AI provider blocked or filtered this response."))
        case .toolCalls, .error, .other:
            let reason = result.rawFinishReason ?? "none"
            throw EnhancementError.customError(
                String(localized: "The AI provider stopped without finishing its response (finish reason: \(reason)).")
            )
        }
    }

    func completeChat(
        provider: AIProvider,
        modelName: String?,
        messages: [ChatRequest.Message],
        systemPrompt: String? = nil,
        timeout: TimeInterval = 30,
        retryOnTimeout: Bool = false
    ) async throws -> String {
        let completion = try await performChatCompletion(
            provider: provider,
            modelName: modelName,
            messages: messages,
            systemPrompt: systemPrompt,
            timeout: timeout,
            retryOnTimeout: retryOnTimeout
        )
        let result = completion.text
        let filteredResult = AIEnhancementOutputFilter.filter(result)
        guard provider != .openRouter || !filteredResult.isEmpty else {
            throw EnhancementError.enhancementFailed
        }
        return filteredResult
    }

    private func chatAPIKey(for provider: AIProvider, modelName: String) throws -> String {
        if provider == .custom {
            guard let customConfiguration = CustomAIProviderManager.shared.requestConfiguration(forModel: modelName)
            else {
                throw EnhancementError.notConfigured
            }
            return customConfiguration.apiKey
        }

        guard let key = APIKeyManager.shared.getAPIKey(forProvider: provider.rawValue), !key.isEmpty else {
            throw EnhancementError.notConfigured
        }
        return key
    }

    private func bedrockChatCompletion(
        apiKey: String,
        model: String,
        messages: [ChatMessage],
        systemPrompt: String?,
        reasoning: BedrockReasoningFields?,
        timeout: TimeInterval,
        retryOnTimeout: Bool
    ) async throws -> String {
        // Converse takes the system prompt as its own field, like Anthropic's Messages API.
        let systemMessages = messages.filter { $0.role == "system" }.map(\.content)
        let system = systemPrompt ?? (systemMessages.isEmpty ? nil : systemMessages.joined(separator: "\n"))
        // Converse requires turns to alternate and start with the user. The Assistant can leave two user
        // turns in a row after a failed reply, so merge same-role neighbours and drop a leading assistant turn.
        var bedrockMessages: [BedrockMessage] = []
        for message in messages {
            let role: BedrockMessage.Role
            switch message.role {
            case "user":
                role = .user
            case "assistant":
                role = .assistant
            default:
                continue
            }
            if bedrockMessages.isEmpty, role == .assistant {
                continue
            }
            if let last = bedrockMessages.last, last.role == role {
                bedrockMessages[bedrockMessages.count - 1] = BedrockMessage(
                    role: role,
                    text: last.text + "\n\n" + message.content
                )
            } else {
                bedrockMessages.append(BedrockMessage(role: role, text: message.content))
            }
        }

        do {
            let completion = try await BedrockConverseClient.converse(
                apiKey: apiKey,
                region: BedrockRegion.current,
                modelID: model,
                systemPrompt: system,
                messages: bedrockMessages,
                reasoning: reasoning,
                timeout: timeout,
                retryOnTimeout: retryOnTimeout
            )
            return completion.text
        } catch let error as BedrockError {
            switch error {
            case .timeout:
                throw EnhancementError.timeout
            case .outputTruncated:
                throw EnhancementError.outputTruncated
            case .missingAPIKey:
                throw EnhancementError.notConfigured
            default:
                throw EnhancementError.customError(error.localizedDescription)
            }
        }
    }

    private func chatPrompt(from messages: [ChatMessage]) -> String {
        let formattedMessages = messages.map { message in
            let label: String
            switch message.role {
            case "assistant":
                label = "assistant"
            case "user":
                label = "user"
            case "system":
                label = "system"
            default:
                label = "other"
            }
            return """
                <message role="\(label)">
                \(message.content)
                </message>
                """
        }
        .joined(separator: "\n\n")

        return """
            <conversation>
            \(formattedMessages)
            </conversation>
            """
    }
}

private extension AIProvider {
    /// The VoiceInkChat preset for providers that speak OpenAI Chat Completions.
    var openAICompatibleChatProvider: OpenAICompatibleProvider? {
        switch self {
        case .openAI: return .openAI
        case .groq: return .groq
        case .cerebras: return .cerebras
        case .mistral: return .mistral
        case .togetherAI: return .together
        case .gmiCloud: return .gmiCloud
        case .baseten: return .baseten
        case .fireworks: return .fireworks
        case .vercelAIGateway: return .vercelAIGateway
        default: return nil
        }
    }
}
