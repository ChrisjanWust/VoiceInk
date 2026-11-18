import Foundation
import LLMkit

struct AIChatCompletionResult: Sendable {
    let text: String
    let openRouterCompletion: OpenRouterCompletion?
}

extension AIService {
    func performChatCompletion(
        provider: AIProvider,
        modelName: String?,
        messages: [ChatMessage],
        systemPrompt: String? = nil,
        localUserPrompt: String? = nil,
        timeout: TimeInterval = 30,
        retryOnTimeout: Bool = false
    ) async throws -> AIChatCompletionResult {
        let resolvedModel = CerebrasModelMigration.resolvedModel(
            modelName?.isEmpty == false ? modelName! : selectedModel(for: provider),
            for: provider
        )

        let result: String
        var openRouterCompletion: OpenRouterCompletion? = nil
        switch provider {
        case .gemini:
            let geminiModel = GeminiModelMigration.resolvedModel(resolvedModel, for: .gemini)
            result = try await GeminiLLMClient.chatCompletion(
                apiKey: try chatAPIKey(for: provider, modelName: geminiModel),
                model: geminiModel,
                messages: messages,
                systemPrompt: systemPrompt,
                thinkingLevel: ReasoningConfig.geminiThinkingLevel(for: geminiModel),
                store: false,
                timeout: timeout,
                retryOnTimeout: retryOnTimeout
            )
        case .anthropic:
            result = try await AnthropicLLMClient.chatCompletion(
                apiKey: try chatAPIKey(for: provider, modelName: resolvedModel),
                model: resolvedModel,
                messages: messages,
                systemPrompt: systemPrompt,
                timeout: timeout,
                retryOnTimeout: retryOnTimeout
            )
        case .openRouter:
            let policy = OpenRouterRequestPolicy.lowLatency(
                modelName: resolvedModel,
                modelMetadata: openRouterModelMetadata(for: resolvedModel)
            )
            let completion = try await OpenRouterClient.chatCompletion(
                apiKey: try chatAPIKey(for: provider, modelName: resolvedModel),
                model: policy.model,
                messages: messages,
                systemPrompt: systemPrompt,
                temperature: policy.temperature,
                reasoning: policy.reasoning,
                provider: policy.provider,
                includeRouterMetadata: true,
                appReferer: URL(string: "https://tryvoiceink.com"),
                appTitle: "VoiceInk",
                timeout: timeout,
                retryOnTimeout: retryOnTimeout
            )
            guard !OpenRouterRequestPolicy.outputWasTruncated(finishReason: completion.finishReason) else {
                throw EnhancementError.outputTruncated
            }
            result = completion.text
            openRouterCompletion = completion
        case .bedrock:
            result = try await bedrockChatCompletion(
                apiKey: try chatAPIKey(for: provider, modelName: resolvedModel),
                model: resolvedModel,
                messages: messages,
                systemPrompt: systemPrompt,
                timeout: timeout,
                retryOnTimeout: retryOnTimeout
            )
        case .custom:
            guard
                let customConfiguration = CustomAIProviderManager.shared.requestConfiguration(forModel: resolvedModel),
                let baseURL = URL(string: customConfiguration.baseURL)
            else {
                throw EnhancementError.notConfigured
            }
            result = try await OpenAILLMClient.chatCompletion(
                baseURL: baseURL,
                apiKey: customConfiguration.apiKey,
                model: customConfiguration.modelName,
                messages: messages,
                systemPrompt: systemPrompt,
                temperature: 0.3,
                timeout: timeout,
                retryOnTimeout: retryOnTimeout
            )
        case .voiceInkRefine:
            throw EnhancementError.customError(
                String(localized: "VoiceInk Refine only supports transcript cleanup.")
            )
        case .ollama:
            result = try await enhanceWithOllama(
                text: localUserPrompt ?? chatPrompt(from: messages),
                systemPrompt: systemPrompt ?? "",
                model: resolvedModel,
                timeout: timeout,
                retryOnTimeout: retryOnTimeout
            )
        case .localCLI:
            result = try await enhanceWithLocalCLI(
                systemPrompt: systemPrompt ?? "",
                userPrompt: localUserPrompt ?? chatPrompt(from: messages)
            )
        default:
            guard let baseURL = URL(string: provider.baseURL) else {
                throw EnhancementError.notConfigured
            }
            let reasoningEffort = ReasoningConfig.getReasoningParameter(
                for: provider,
                modelName: resolvedModel
            )
            let extraBody = ReasoningConfig.getExtraBodyParameters(
                for: provider,
                modelName: resolvedModel
            )
            result = try await OpenAILLMClient.chatCompletion(
                baseURL: baseURL,
                apiKey: try chatAPIKey(for: provider, modelName: resolvedModel),
                model: resolvedModel,
                messages: messages,
                systemPrompt: systemPrompt,
                temperature: 0.3,
                reasoningEffort: reasoningEffort,
                extraBody: extraBody,
                timeout: timeout,
                retryOnTimeout: retryOnTimeout
            )
        }

        return AIChatCompletionResult(text: result, openRouterCompletion: openRouterCompletion)
    }

    func completeChat(
        provider: AIProvider,
        modelName: String?,
        messages: [ChatMessage],
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
