import Foundation
import LLMkit

struct OpenRouterRequestPolicy {
    let model: String
    let temperature: Double?
    let reasoning: OpenRouterReasoning?
    let provider: OpenRouterProviderPreferences

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
            provider: OpenRouterProviderPreferences(
                sort: .throughput,
                preferredMaxLatency: OpenRouterPercentileThresholds(p90: 2.5),
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
    ) -> OpenRouterReasoning? {
        guard supportsReasoningParameter, let capabilities else {
            return nil
        }

        if capabilities.mandatory {
            let minimumEffort = capabilities.supportedEfforts.last
            return OpenRouterReasoning(effort: minimumEffort, exclude: true)
        }

        if capabilities.supportedEfforts.contains("none") {
            return OpenRouterReasoning(effort: "none", exclude: true)
        }

        return OpenRouterReasoning(enabled: false, exclude: true)
    }

    // A level the user picked. Reasoning text is always excluded from the response.
    private static func reasoning(
        for level: ReasoningLevel,
        capabilities: OpenRouterReasoningCapabilities?
    ) -> OpenRouterReasoning {
        if level == .off {
            if capabilities?.supportedEfforts.contains("none") ?? true {
                return OpenRouterReasoning(effort: "none", exclude: true)
            }
            return OpenRouterReasoning(enabled: false, exclude: true)
        }
        return OpenRouterReasoning(effort: level.effortValue, exclude: true)
    }

    static func outputWasTruncated(finishReason: String?) -> Bool {
        guard let finishReason else { return false }
        return ["length", "max_tokens", "max_output_tokens"].contains(finishReason.lowercased())
    }
}
