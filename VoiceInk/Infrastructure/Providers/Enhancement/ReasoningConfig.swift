import Foundation
import LLMkit

/// What VoiceInk knows about a model's reasoning support beyond its ID.
struct ReasoningModelInfo {
    /// The OpenRouter catalog entry, when the catalog has loaded.
    var openRouterModel: OpenRouterModel? = nil
    /// Whether an Ollama model lists the "thinking" capability. `nil` when unknown.
    var ollamaSupportsThinking: Bool? = nil
}

/// Reasoning parameters for enhancement requests.
///
/// Each provider has two parts here:
/// - the levels a user may pick for a model (`supportedLevels`), and
/// - how a level becomes request parameters.
///
/// A `nil` level means "Default": the request is exactly VoiceInk's built-in
/// low-latency request for that model, as listed in the model sets below.
struct ReasoningConfig {

    // MARK: - Gemini

    // Gemini 3.8 Flash, 3.7 Flash, and 3.1 Pro Preview do not support "minimal".
    static let geminiLowThinkingModels: Set<String> = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.1-pro-preview",
    ]

    // These models support "minimal", optimized for low-latency instruction following.
    static let geminiMinimalThinkingModels: Set<String> = [
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
    ]

    // Thinking is off by default. LLMkit sends non-preview models to `/v1/interactions`,
    // which rejects every thinking level for these ("Thinking is not enabled for api
    // version v1"), so they stay on Default. (`/v1beta` accepts only "low" and "high".)
    static let geminiThinkingOffByDefaultModels: Set<String> = [
        "gemini-2.5-flash-lite"
    ]

    // Gemini 2.5 Flash-Lite gets no default level because its Interactions
    // API default has thinking off.
    static func geminiThinkingLevel(for modelName: String, level: ReasoningLevel? = nil) -> GeminiThinkingLevel? {
        if let level = effectiveLevel(level, for: .gemini, modelName: modelName) {
            return geminiThinkingLevel(from: level)
        }
        if geminiMinimalThinkingModels.contains(modelName) {
            return .minimal
        }
        if geminiLowThinkingModels.contains(modelName) {
            return .low
        }
        return nil
    }

    private static func geminiThinkingLevel(from level: ReasoningLevel) -> GeminiThinkingLevel? {
        switch level {
        case .minimal: return .minimal
        case .low: return .low
        case .medium: return .medium
        case .high: return .high
        case .off, .xhigh, .max, .enabled: return nil
        }
    }

    // Verified live on 2026-11-17. The API also lists "xhigh", but LLMkit's
    // GeminiThinkingLevel can't send it yet.
    private static func geminiLevels(for modelName: String) -> [ReasoningLevel] {
        if geminiThinkingOffByDefaultModels.contains(modelName) {
            return []
        }
        if geminiLowThinkingModels.contains(modelName) {
            return [.low, .medium, .high]
        }
        return [.minimal, .low, .medium, .high]
    }

    // MARK: - OpenAI

    // These OpenAI models support explicit "none" for low-latency enhancement.
    // GPT-4.1 models need no parameter.
    static let openAINoneReasoningModels: Set<String> = [
        "gpt-6-luna",
        "gpt-6-sol",
        "gpt-5.6-luna",
        "gpt-5.6-terra",
        "gpt-5.6-sol",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    ]

    // MARK: - Cerebras

    // Cerebras GPT-OSS has no true "none"; use lowest effort.
    static let cerebrasGPTOSSMinimumReasoningModels: Set<String> = [
        "gpt-oss-120b"
    ]

    // Cerebras Qwen supports "none" for low-latency requests.
    static let cerebrasNoneReasoningModels: Set<String> = [
        "qwen-3.8-27b"
    ]

    // Verified live on 2026-11-17: Qwen takes none/low/medium/high, GPT-OSS takes
    // low/medium/high, and both reject minimal/xhigh/max.
    private static func cerebrasLevels(for modelName: String) -> [ReasoningLevel] {
        if cerebrasGPTOSSMinimumReasoningModels.contains(modelName) {
            return [.low, .medium, .high]
        }
        if cerebrasNoneReasoningModels.contains(modelName) {
            return [.off, .low, .medium, .high]
        }
        return [.off, .low, .medium, .high, .max]
    }

    // MARK: - Groq

    // Groq GPT-OSS has no true "none"; use lowest effort.
    static let groqGPTOSSMinimumReasoningModels: Set<String> = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    ]

    // Groq Qwen supports "none" for non-thinking, low-latency requests.
    static let groqNoneReasoningModels: Set<String> = [
        "qwen/qwen3.8-27b"
    ]

    private static func groqLevels(for modelName: String) -> [ReasoningLevel] {
        if groqGPTOSSMinimumReasoningModels.contains(modelName) {
            return [.low, .medium, .high]
        }
        if groqNoneReasoningModels.contains(modelName) {
            return [.off, .low, .medium, .high]
        }
        return [.off, .low, .medium, .high, .xhigh]
    }

    // MARK: - OpenRouter

    private static func openRouterLevels(for model: OpenRouterModel?) -> [ReasoningLevel] {
        guard let model else {
            // No catalog entry yet: offer every OpenRouter effort and let the user choose.
            return openAICompatibleEffortLevels
        }
        guard model.supportedParameters.contains("reasoning") else {
            return []
        }
        guard let capabilities = model.reasoning else {
            return openAICompatibleEffortLevels
        }

        let efforts = Set(capabilities.supportedEfforts.compactMap(ReasoningLevel.init(effortValue:)))
        let effortLevels = efforts.subtracting([.off]).isEmpty
            ? Set(openAICompatibleEffortLevels).subtracting([.off])
            : efforts.subtracting([.off])
        let canTurnOff = !capabilities.mandatory

        return ReasoningLevel.allCases.filter { level in
            level == .off ? canTurnOff : effortLevels.contains(level)
        }
    }

    // MARK: - Shared

    // Every documented OpenAI-style effort. Used where a model's own support is unknown.
    static let openAICompatibleEffortLevels: [ReasoningLevel] = [
        .off, .minimal, .low, .medium, .high, .xhigh, .max,
    ]

    /// Levels the user may pick for this model, besides Default. Empty means the
    /// model only runs on Default, either because it has no reasoning control or
    /// because LLMkit cannot send that provider's control yet.
    ///
    /// Adding reasoning for a provider (new providers start on Default only):
    /// 1. Return its levels here, from a `<provider>Levels(for:)` helper with the
    ///    provider's model sets above.
    /// 2. Map a level to request parameters: `getReasoningParameter` and
    ///    `getExtraBodyParameters` for OpenAI-compatible APIs, or a dedicated
    ///    function like `geminiThinkingLevel(for:level:)` for native APIs.
    /// 3. Pass the level through in `AIService.performChatCompletion`.
    /// 4. If Default sends a parameter today, describe it in `defaultLevel`.
    static func supportedLevels(
        for provider: AIProvider,
        modelName: String,
        modelInfo: ReasoningModelInfo = ReasoningModelInfo()
    ) -> [ReasoningLevel] {
        switch provider {
        case .gemini:
            return geminiLevels(for: modelName)
        case .cerebras:
            return cerebrasLevels(for: modelName)
        case .groq:
            return groqLevels(for: modelName)
        case .openRouter:
            return openRouterLevels(for: modelInfo.openRouterModel)
        case .ollama:
            // Ollama rejects `think: true` for models without the thinking capability.
            // LLMkit sends `think` as a Bool, so "low"/"medium"/"high" are not available.
            return modelInfo.ollamaSupportsThinking == false ? [] : [.enabled]
        case .custom:
            return openAICompatibleEffortLevels
        case .openAI, .anthropic, .mistral:
            // Need LLMkit support. OpenAI GPT-5 models reject `temperature` unless
            // `reasoning_effort` is "none", and OpenAILLMClient always sends it.
            // Anthropic needs `output_config.effort`; Mistral returns thinking as
            // chunked `content` that OpenAILLMClient can't decode.
            return []
        default:
            // Transcription-only providers, VoiceInk Refine, Local CLI, and any
            // provider without a mapping yet stay on Default.
            return []
        }
    }

    /// Returns the level when this model accepts it, otherwise `nil` (Default).
    static func effectiveLevel(
        _ level: ReasoningLevel?,
        for provider: AIProvider,
        modelName: String,
        modelInfo: ReasoningModelInfo = ReasoningModelInfo()
    ) -> ReasoningLevel? {
        guard let level,
            supportedLevels(for: provider, modelName: modelName, modelInfo: modelInfo).contains(level)
        else {
            return nil
        }
        return level
    }

    /// The level Default sends for this model, for display. `nil` means Default sends
    /// no reasoning parameter, so the provider's own default applies.
    static func defaultLevel(
        for provider: AIProvider,
        modelName: String,
        modelInfo: ReasoningModelInfo = ReasoningModelInfo()
    ) -> ReasoningLevel? {
        switch provider {
        case .gemini:
            switch geminiThinkingLevel(for: modelName) {
            case .minimal: return .minimal
            case .low: return .low
            case .medium: return .medium
            case .high: return .high
            case nil: return geminiThinkingOffByDefaultModels.contains(modelName) ? .off : nil
            }
        case .openRouter:
            guard let openRouterModel = modelInfo.openRouterModel,
                openRouterModel.supportedParameters.contains("reasoning"),
                let capabilities = openRouterModel.reasoning
            else {
                return nil
            }
            if capabilities.mandatory {
                return capabilities.supportedEfforts.last.flatMap(ReasoningLevel.init(effortValue:))
            }
            return .off
        case .ollama:
            return .off
        default:
            return getReasoningParameter(for: provider, modelName: modelName)
                .flatMap(ReasoningLevel.init(effortValue:))
        }
    }

    // MARK: - OpenAI-compatible requests

    static func getReasoningParameter(
        for provider: AIProvider,
        modelName: String,
        level: ReasoningLevel? = nil
    ) -> String? {
        if let level = effectiveLevel(level, for: provider, modelName: modelName) {
            return level.effortValue
        }

        switch provider {
        case .openAI:
            if openAINoneReasoningModels.contains(modelName) { return "none" }
        case .cerebras:
            if cerebrasGPTOSSMinimumReasoningModels.contains(modelName) {
                return "low"
            } else if cerebrasNoneReasoningModels.contains(modelName) {
                return "none"
            }
        case .groq:
            if groqGPTOSSMinimumReasoningModels.contains(modelName) {
                return "low"
            } else if groqNoneReasoningModels.contains(modelName) {
                return "none"
            }
        default:
            return nil
        }
        return nil
    }

    // Provider-specific body params for hiding reasoning.
    static func getExtraBodyParameters(
        for provider: AIProvider,
        modelName: String,
        level: ReasoningLevel? = nil
    ) -> [String: Any]? {
        if provider == .cerebras && cerebrasGPTOSSMinimumReasoningModels.contains(modelName) {
            return ["reasoning_format": "hidden"]
        } else if provider == .groq && groqGPTOSSMinimumReasoningModels.contains(modelName) {
            return ["include_reasoning": false]
        }

        // Groq returns reasoning unless told not to. Only added when the user turns
        // reasoning on, so Default requests stay unchanged.
        if provider == .groq,
            let level = effectiveLevel(level, for: provider, modelName: modelName),
            level != .off
        {
            return ["include_reasoning": false]
        }
        return nil
    }

    // MARK: - Ollama

    /// Ollama's `think` value. Default keeps thinking off for low latency.
    static func ollamaThink(level: ReasoningLevel?, modelName: String, modelInfo: ReasoningModelInfo) -> Bool {
        effectiveLevel(level, for: .ollama, modelName: modelName, modelInfo: modelInfo) == .enabled
    }
}
