import Foundation
// OpenRouterModel: the model catalog, which speech-to-text shares.
import LLMkit
import VoiceInkChat

/// VoiceInk's OpenRouter request. Reasoning text is always excluded from the response
/// (`OpenRouterChatClient` sends `exclude: true` with every `reasoning` object).
struct OpenRouterRequestPolicy {
    let model: String
    let temperature: Double?
    let reasoning: ChatRequest.Reasoning?
    let provider: OpenRouterChatClient.ProviderPreferences

    static func lowLatency(
        modelName: String,
        modelMetadata: OpenRouterModel?,
        reasoningLevel: ReasoningLevel? = nil
    ) -> OpenRouterRequestPolicy {
        let supportedParameters = Set(modelMetadata?.supportedParameters ?? [])
        let selectedLevel = ReasoningConfig.effectiveLevel(
            reasoningLevel,
            for: .openRouter,
            modelName: modelName,
            modelInfo: ReasoningModelInfo(openRouterModel: modelMetadata)
        )
        let selectedReasoning = selectedLevel.map { reasoning(for: $0, capabilities: modelMetadata?.reasoning) }
        // Reasoning models such as OpenAI GPT-5 reject `temperature` once reasoning is on.
        let sendsTemperature = supportedParameters.contains("temperature")
            && (selectedLevel == nil || selectedLevel == .off)

        return OpenRouterRequestPolicy(
            model: nitroVariant(of: modelName),
            temperature: sendsTemperature ? 0.3 : nil,
            reasoning: selectedReasoning ?? reasoningConfiguration(
                capabilities: modelMetadata?.reasoning,
                supportsReasoningParameter: supportedParameters.contains("reasoning")
            ),
            provider: OpenRouterChatClient.ProviderPreferences(
                sort: .throughput,
                preferredMaxLatencyP90: 2.5,
                requireParameters: true,
                allowFallbacks: true
            )
        )
    }

    private static func nitroVariant(of modelName: String) -> String {
        guard let finalComponent = modelName.split(separator: "/").last,
              !finalComponent.contains(":") else {
            return modelName
        }
        return "\(modelName):nitro"
    }

    private static func reasoningConfiguration(
        capabilities: OpenRouterReasoningCapabilities?,
        supportsReasoningParameter: Bool
    ) -> ChatRequest.Reasoning? {
        guard supportsReasoningParameter, let capabilities else {
            return nil
        }

        if capabilities.mandatory {
            // Reasoning can't be turned off, so ask for the least. A model that lists no
            // efforts reasons at its own default.
            return capabilities.supportedEfforts.last.map(ChatRequest.Reasoning.effort) ?? .enabled
        }

        if capabilities.supportedEfforts.contains("none") {
            return .effort("none")
        }

        return .disabled
    }

    // A level the user picked.
    private static func reasoning(
        for level: ReasoningLevel,
        capabilities: OpenRouterReasoningCapabilities?
    ) -> ChatRequest.Reasoning {
        if level == .off {
            if capabilities?.supportedEfforts.contains("none") ?? true {
                return .effort("none")
            }
            return .disabled
        }
        return level.effortValue.map(ChatRequest.Reasoning.effort) ?? .enabled
    }
}
