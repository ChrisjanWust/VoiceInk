import Foundation
import VoiceInkChat

struct AIChatCompletionResult: Sendable {
    let text: String
    /// The full result for OpenRouter, which says which upstream provider served it.
    let openRouterResult: ChatResult?
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

        var request = ChatRequest(
            model: resolvedModel,
            system: systemPrompt,
            messages: messages,
            timeout: requestTimeout.seconds,
            retryOnTimeout: requestTimeout.retryOnTimeout
        )
        let client: any ChatClient

        switch provider {
        case .gemini:
            request.reasoning = ReasoningConfig.geminiThinkingLevel(for: resolvedModel, level: reasoningLevel)
                .map(ChatRequest.Reasoning.effort)
            client = GeminiChatClient(
                apiKey: try chatAPIKey(for: provider, modelName: resolvedModel),
                apiVersion: .forModel(resolvedModel)
            )
        case .anthropic:
            ReasoningConfig.anthropicReasoning(for: resolvedModel, level: reasoningLevel)?.apply(to: &request)
            client = AnthropicChatClient(apiKey: try chatAPIKey(for: provider, modelName: resolvedModel))
        case .openRouter:
            let policy = OpenRouterRequestPolicy.lowLatency(
                modelName: resolvedModel,
                modelMetadata: openRouterModelMetadata(for: resolvedModel),
                reasoningLevel: reasoningLevel
            )
            request.model = policy.model
            request.temperature = policy.temperature
            request.reasoning = policy.reasoning
            client = RoutedOpenRouterClient(
                client: OpenRouterChatClient(
                    apiKey: try chatAPIKey(for: provider, modelName: resolvedModel),
                    appReferer: URL(string: "https://tryvoiceink.com"),
                    appTitle: "VoiceInk"
                ),
                preferences: policy.provider
            )
        case .bedrock:
            let reasoning = BedrockReasoning.fields(
                for: reasoningLevel,
                model: reasoningModelInfo(for: .bedrock, model: resolvedModel).bedrockModel
            )
            request.maxTokens = reasoning?.maxTokens
            client = BedrockEnhancementClient(
                apiKey: try chatAPIKey(for: provider, modelName: resolvedModel),
                region: BedrockRegion.current,
                additionalModelRequestFields: reasoning?.additionalModelRequestFields ?? [:]
            )
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
            request.temperature = ReasoningConfig.defaultTemperature
            request.reasoning = ReasoningConfig.ollamaReasoning(
                level: reasoningLevel,
                modelName: resolvedModel,
                modelInfo: reasoningModelInfo(for: .ollama, model: resolvedModel)
            )
            client = try ollamaChatClient()
        case .localCLI:
            let text = try await enhanceWithLocalCLI(
                systemPrompt: systemPrompt ?? "",
                userPrompt: localUserPrompt ?? chatPrompt(from: messages)
            )
            return AIChatCompletionResult(text: text, openRouterResult: nil)
        default:
            guard let chatProvider = provider.openAICompatibleChatProvider else {
                throw EnhancementError.notConfigured
            }
            request.temperature = ReasoningConfig.temperature(for: provider, modelName: resolvedModel, level: reasoningLevel)
            request.reasoning = ReasoningConfig.getReasoningParameter(
                for: provider,
                modelName: resolvedModel,
                level: reasoningLevel,
                modelInfo: reasoningModelInfo(for: provider, model: resolvedModel)
            ).map(ChatRequest.Reasoning.effort)
            request.extraBody = ReasoningConfig.extraBodyFields(for: provider, modelName: resolvedModel, level: reasoningLevel)
            client = OpenAICompatibleChatClient(
                apiKey: try chatAPIKey(for: provider, modelName: resolvedModel),
                provider: chatProvider
            )
        }

        // Fast mode's fields go last, so they win over anything else in the body.
        request.extraBody.merge(fastMode?.bodyFields ?? [:]) { _, fast in fast }
        let completion: ChatResult
        do {
            completion = try await client.complete(request)
        } catch let error as BedrockError {
            switch error {
            case .timeout: throw EnhancementError.timeout
            case .outputTruncated: throw EnhancementError.outputTruncated
            case .missingAPIKey: throw EnhancementError.notConfigured
            default: throw EnhancementError.customError(error.localizedDescription)
            }
        }
        try Self.requireFinishedAnswer(completion, from: provider)
        return AIChatCompletionResult(text: completion.text, openRouterResult: provider == .openRouter ? completion : nil)
    }

    /// Only a finished answer is used. A cut-off, filtered or failed one throws, so it's never
    /// pasted: a guardrail's replacement text, for example, comes back with text.
    ///
    /// A Custom endpoint can be any server, and some name a normal finish differently (such as
    /// `eos_token`) or leave it out, so there an unrecognised finish with text is accepted.
    private static func requireFinishedAnswer(_ result: ChatResult, from provider: AIProvider) throws {
        switch result.finishReason {
        case .stop:
            return
        case .other where provider == .custom:
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

    /// The conversation as one prompt, for Local CLI, which takes a single input.
    private func chatPrompt(from messages: [ChatRequest.Message]) -> String {
        let formattedMessages = messages.map { message in
            """
                <message role="\(message.role.rawValue)">
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

/// OpenRouter with VoiceInk's routing preferences.
private struct RoutedOpenRouterClient: ChatClient {
    let client: OpenRouterChatClient
    let preferences: OpenRouterChatClient.ProviderPreferences

    func complete(_ request: ChatRequest) async throws -> ChatResult {
        try await client.complete(request, providerPreferences: preferences)
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
