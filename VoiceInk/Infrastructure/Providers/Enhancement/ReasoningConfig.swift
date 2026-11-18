import Foundation
import LLMkit

/// What VoiceInk knows about a model's reasoning support beyond its ID.
struct ReasoningModelInfo {
    /// The OpenRouter catalog entry, when the catalog has loaded.
    var openRouterModel: OpenRouterModel? = nil
    /// Whether an Ollama model lists the "thinking" capability. `nil` when unknown.
    var ollamaSupportsThinking: Bool? = nil
    /// The named thinking levels an Ollama model takes, such as `["low", "medium", "high"]`
    /// for gpt-oss, from `/api/show`'s `thinking.values`. Empty for models with only an
    /// on/off switch, and `nil` when unknown.
    var ollamaThinkingLevels: [String]? = nil
    /// The other IDs of a Mistral model, from the live list's aliases. For
    /// `mistral-small-latest` this is the dated ID, such as `mistral-small-2603`. Empty when
    /// unknown (offline, no key, or the built-in list).
    var mistralAliasTargets: [String] = []
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
        "gemini-4-carbon",
        "gemini-4-argon",
        "gemini-4-hydrogen",
    ]

    // These models support "minimal", optimized for low-latency instruction following.
    static let geminiMinimalThinkingModels: Set<String> = [
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
    ]

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
        "gpt-6.1-terra",
        "gpt-5.2",
    ]

    // These OpenAI models reject "none", so Default uses the lowest effort, "low". They
    // reject `temperature: 0.3` even without `reasoning_effort`, so they always get
    // `temperature: null`.
    static let openAIEffortRequiredModels: Set<String> = [
        "gpt-6.2-sol",
        "gpt-6.2-astra",
        "gpt-6.1-sol",
        "gpt-6-astra",
    ]

    /// Whether VoiceInk's Default request works for this OpenAI model.
    static func isOpenAIDefaultRequestSupported(_ modelName: String) -> Bool {
        openAINoneReasoningModels.contains(modelName) || openAIEffortRequiredModels.contains(modelName)
    }

    // Non-reasoning models reject `reasoning_effort`.
    static let openAINonReasoningModelPrefixes = ["gpt-4", "gpt-3.5", "chatgpt-4o"]

    // Verified live on 2026-11-17. No listed model accepts "minimal" or "max".
    private static func openAILevels(for modelName: String) -> [ReasoningLevel] {
        if openAINonReasoningModelPrefixes.contains(where: { modelName.hasPrefix($0) }) {
            return []
        }
        if openAINoneReasoningModels.contains(modelName) {
            return [.off, .low, .medium, .high, .xhigh]
        }
        if openAIEffortRequiredModels.contains(modelName) {
            return [.low, .medium, .high, .xhigh]
        }
        return openAICompatibleEffortLevels
    }

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

    // MARK: - Anthropic

    // From Anthropic's effort and thinking docs (2026-11-18). IDs are matched without a
    // trailing date, so `claude-haiku-4-5-20251001` is `claude-haiku-4-5`.

    // Adaptive thinking, with every effort from low to max.
    static let anthropicXHighEffortModels: Set<String> = [
        "claude-fable-5-2", "claude-mythos-5-2", "claude-fable-5-1", "claude-mythos-5-1",
        "claude-fable-5", "claude-mythos-5", "claude-opus-5-5", "claude-opus-5",
        "claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-5-5", "claude-sonnet-5",
        "claude-haiku-5-5",
    ]

    // Adaptive thinking, with low to max but no xhigh.
    static let anthropicMaxEffortModels: Set<String> = [
        "claude-opus-4-6", "claude-sonnet-4-6", "claude-mythos-preview",
    ]

    // Extended thinking only (`budget_tokens`). They reject adaptive thinking.
    static let anthropicBudgetThinkingModels: Set<String> = [
        "claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5",
        "claude-opus-4-1", "claude-opus-4", "claude-sonnet-4", "claude-3-7-sonnet",
    ]

    // The one extended-thinking model that also takes effort. The docs say to set both.
    static let anthropicBudgetAndEffortModels: Set<String> = ["claude-opus-4-5"]

    // Models that think by default, so Default already reasons. Thinking can be turned off
    // with `disabled` on some and `between_tools` on Sonnet 5.5; the rest can't turn it off.
    static let anthropicThinkingByDefaultModels: Set<String> = [
        "claude-fable-5-2", "claude-mythos-5-2", "claude-fable-5-1", "claude-mythos-5-1",
        "claude-fable-5", "claude-mythos-5", "claude-mythos-preview", "claude-opus-5-5",
        "claude-opus-5", "claude-sonnet-5-5", "claude-sonnet-5", "claude-haiku-5-5",
    ]
    static let anthropicDisabledThinkingModels: Set<String> = [
        "claude-opus-5", "claude-sonnet-5", "claude-haiku-5-5",
    ]
    static let anthropicBetweenToolsModels: Set<String> = ["claude-sonnet-5-5"]

    // Models whose API default effort is medium rather than high.
    static let anthropicMediumDefaultEffortModels: Set<String> = ["claude-opus-5-5", "claude-haiku-5-5"]

    /// LLMkit's `max_tokens`, which Default sends. Thinking counts toward it.
    static let anthropicDefaultMaxTokens = 8192

    /// Strips a trailing `-YYYYMMDD` snapshot date.
    static func anthropicModelFamily(_ modelName: String) -> String {
        let model = modelName.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return model.replacingOccurrences(of: #"-\d{8}$"#, with: "", options: .regularExpression)
    }

    private static func anthropicLevels(for modelName: String) -> [ReasoningLevel] {
        let family = anthropicModelFamily(modelName)
        if anthropicBudgetThinkingModels.contains(family) {
            // Thinking is off by default, so Off would repeat Default.
            return [.low, .medium, .high]
        }
        let canTurnOff =
            anthropicDisabledThinkingModels.contains(family) || anthropicBetweenToolsModels.contains(family)
        let efforts: [ReasoningLevel] =
            anthropicMaxEffortModels.contains(family)
            ? [.low, .medium, .high, .max]
            : [.low, .medium, .high, .xhigh, .max]
        return (canTurnOff ? [.off] : []) + efforts
    }

    /// The effort Default runs at, for models that think by default. `nil` for models
    /// that don't think unless asked.
    private static func anthropicDefaultEffort(for modelName: String) -> ReasoningLevel? {
        let family = anthropicModelFamily(modelName)
        guard anthropicThinkingByDefaultModels.contains(family) else { return nil }
        return anthropicMediumDefaultEffortModels.contains(family) ? .medium : .high
    }

    /// The request fields a picked Anthropic level sets.
    struct AnthropicReasoning: Equatable {
        var thinking: AnthropicThinking
        var effort: String?
        var maxTokens: Int

        func apply(to request: inout AnthropicMessagesRequest) {
            request.thinking = thinking
            request.effort = effort
            request.maxTokens = maxTokens
        }
    }

    /// The reasoning fields for a picked level, or `nil` for Default, which LLMkit sends
    /// without any. Thinking blocks are requested without text (`display: omitted`), since
    /// VoiceInk never shows them.
    static func anthropicReasoning(for modelName: String, level: ReasoningLevel?) -> AnthropicReasoning? {
        guard let level = effectiveLevel(level, for: .anthropic, modelName: modelName) else {
            return nil
        }
        let family = anthropicModelFamily(modelName)

        if anthropicBudgetThinkingModels.contains(family) {
            let budget: Int
            switch level {
            case .low: budget = 1024
            case .medium: budget = 4096
            default: budget = 16_000
            }
            return AnthropicReasoning(
                thinking: .enabled(budgetTokens: budget, display: .omitted),
                effort: anthropicBudgetAndEffortModels.contains(family) ? level.effortValue : nil,
                maxTokens: anthropicDefaultMaxTokens + budget
            )
        }

        if level == .off {
            return AnthropicReasoning(
                thinking: anthropicBetweenToolsModels.contains(family) ? .betweenTools : .disabled,
                effort: nil,
                maxTokens: anthropicDefaultMaxTokens
            )
        }

        // xhigh and max think longer, and thinking counts toward `max_tokens`.
        let maxTokens = [.xhigh, .max].contains(level) ? 32_000 : anthropicDefaultMaxTokens
        return AnthropicReasoning(thinking: .adaptive(display: .omitted), effort: level.effortValue, maxTokens: maxTokens)
    }

    // MARK: - Mistral

    // From Mistral's reasoning docs (2026-11-18), not live-tested: no Mistral key was
    // available. `reasoning_effort` is the only control. Only the model IDs and values the
    // docs name are offered. A `-latest` alias gets the levels of the dated ID it points to
    // in the live list (`ReasoningModelInfo.mistralAliasTargets`). Without that mapping, and
    // for any other ID, the model stays on Default.

    // Mistral Small 4 and Medium 3.5 take "none" or "high".
    static let mistralNoneOrHighModels: Set<String> = [
        "mistral-small-2603",
        "mistral-medium-3-5",
    ]

    // Mistral Large 4 takes low, medium and high, and uses high when it's omitted.
    static let mistralLowToHighModels: Set<String> = [
        "mistral-large-4",
        "mistral-large-4-0",
    ]

    // GLM 5.3 takes low, high and max, and can't turn reasoning off.
    static let mistralLowHighMaxModels: Set<String> = [
        "zai-glm-5-3",
        "zai-glm-5-3-flash",
    ]

    /// The ID the Mistral docs name for this model: the ID itself, or the first dated ID a
    /// live-list alias points to that has documented levels. Requests still send `modelName`.
    private static func mistralDocumentedModel(for modelName: String, aliasTargets: [String]) -> String {
        ([modelName] + aliasTargets).first { !mistralDocumentedLevels(for: $0).isEmpty } ?? modelName
    }

    private static func mistralLevels(for modelName: String, aliasTargets: [String]) -> [ReasoningLevel] {
        mistralDocumentedLevels(for: mistralDocumentedModel(for: modelName, aliasTargets: aliasTargets))
    }

    private static func mistralDocumentedLevels(for modelName: String) -> [ReasoningLevel] {
        if mistralNoneOrHighModels.contains(modelName) {
            return [.off, .high]
        }
        if mistralLowToHighModels.contains(modelName) {
            return [.low, .medium, .high]
        }
        if mistralLowHighMaxModels.contains(modelName) {
            return [.low, .high, .max]
        }
        return []
    }

    private static func mistralDefaultLevel(for modelName: String, aliasTargets: [String]) -> ReasoningLevel? {
        mistralLowToHighModels.contains(mistralDocumentedModel(for: modelName, aliasTargets: aliasTargets))
            ? .high : nil
    }

    /// The `reasoning_effort` for a picked level, or `nil` for Default, which LLMkit sends
    /// without one.
    static func mistralReasoningEffort(
        for modelName: String,
        level: ReasoningLevel?,
        modelInfo: ReasoningModelInfo = ReasoningModelInfo()
    ) -> String? {
        effectiveLevel(level, for: .mistral, modelName: modelName, modelInfo: modelInfo)?.effortValue
    }

    // MARK: - Shared

    // Every documented OpenAI-style effort. Used where a model's own support is unknown.
    static let openAICompatibleEffortLevels: [ReasoningLevel] = [
        .off, .minimal, .low, .medium, .high, .xhigh, .max,
    ]

    /// Levels the user may pick for this model, besides Default. Empty means the
    /// model only runs on Default, either because it has no reasoning control or
    /// because VoiceInk cannot send that provider's control yet.
    ///
    /// Adding reasoning for a provider (new providers start on Default only):
    /// 1. Return its levels here, from a `<provider>Levels(for:)` helper with the
    ///    provider's model sets above.
    /// 2. Map a level to request parameters: `getReasoningParameter` and
    ///    `getExtraBodyParameters` for OpenAI-compatible APIs, or a dedicated
    ///    function like `geminiThinkingLevel(for:level:)` for native APIs. When
    ///    LLMkit can't send the parameter or read the reply, send picked levels
    ///    through a small in-app client (like `AnthropicMessagesClient`) and keep
    ///    Default on LLMkit.
    /// 3. Pass the level through in `AIService.performChatCompletion`.
    /// 4. If Default sends a parameter today, describe it in `defaultLevel`.
    static func supportedLevels(
        for provider: AIProvider,
        modelName: String,
        modelInfo: ReasoningModelInfo = ReasoningModelInfo()
    ) -> [ReasoningLevel] {
        let tooSlow = tooSlowLevels[provider]?[modelName] ?? []
        return providerLevels(for: provider, modelName: modelName, modelInfo: modelInfo)
            .filter { !tooSlow.contains($0) }
    }

    // Levels too slow for dictation in live measurements (median above ~45 s or max above
    // ~120 s), so they aren't offered for these models. OpenAI and Gemini were measured on
    // 2026-11-17; Anthropic and Ollama on 2026-11-18. Ollama runs locally, so its numbers
    // come from one Mac (gpt-oss:20b High: 24-150 s, two runs timed out).
    static let tooSlowLevels: [AIProvider: [String: Set<ReasoningLevel>]] = [
        .openAI: ["gpt-5.4": [.xhigh], "gpt-5.4-mini": [.xhigh]],
        .gemini: ["gemini-4-carbon": [.high], "gemini-4-argon": [.high]],
        .anthropic: ["claude-sonnet-5": [.max]],
        .ollama: ["gpt-oss:20b": [.high], "gpt-oss:latest": [.high], "gpt-oss": [.high]],
    ]

    private static func providerLevels(
        for provider: AIProvider,
        modelName: String,
        modelInfo: ReasoningModelInfo
    ) -> [ReasoningLevel] {
        switch provider {
        case .openAI:
            return openAILevels(for: modelName)
        case .gemini:
            return geminiLevels(for: modelName)
        case .cerebras:
            return cerebrasLevels(for: modelName)
        case .groq:
            return groqLevels(for: modelName)
        case .openRouter:
            return openRouterLevels(for: modelInfo.openRouterModel)
        case .ollama:
            return ollamaLevels(for: modelInfo)
        case .custom:
            return openAICompatibleEffortLevels
        case .anthropic:
            return anthropicLevels(for: modelName)
        case .mistral:
            return mistralLevels(for: modelName, aliasTargets: modelInfo.mistralAliasTargets)
        case .bedrock:
            // Converse sends no reasoning parameters, so each Bedrock model runs on its
            // own default. Mapping levels needs per-model `additionalModelRequestFields`.
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
            case nil: return nil
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
            // Models with named levels (gpt-oss) can't turn thinking off, and `think: false`
            // didn't run them at their lowest level in live tests: Default took 25-80 s on
            // gpt-oss:20b and Low 2-4 s. So don't name what Default does for them.
            if ollamaLevels(for: modelInfo) != [.enabled] {
                return nil
            }
            return .off
        case .anthropic:
            return anthropicDefaultEffort(for: modelName)
        case .mistral:
            return mistralDefaultLevel(for: modelName, aliasTargets: modelInfo.mistralAliasTargets)
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
            if openAIEffortRequiredModels.contains(modelName) { return "low" }
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

    // Providers verified live to accept `temperature: null`. OpenAI GPT-5 and GPT-6
    // models reject any other temperature once reasoning is on, and LLMkit always
    // sends 0.3, so a picked level replaces it with null. Groq and Mistral keep 0.3.
    static let nullTemperatureProviders: Set<AIProvider> = [.openAI, .cerebras, .custom]

    // Provider-specific body params: hiding reasoning, and nulling `temperature`.
    static func getExtraBodyParameters(
        for provider: AIProvider,
        modelName: String,
        level: ReasoningLevel? = nil
    ) -> [String: Any]? {
        let selectedLevel = effectiveLevel(level, for: provider, modelName: modelName)
        var parameters: [String: Any] = [:]

        if provider == .cerebras && cerebrasGPTOSSMinimumReasoningModels.contains(modelName) {
            parameters["reasoning_format"] = "hidden"
        } else if provider == .groq && groqGPTOSSMinimumReasoningModels.contains(modelName) {
            parameters["include_reasoning"] = false
        } else if provider == .groq, let selectedLevel, selectedLevel != .off {
            // Groq returns reasoning unless told not to. Only added when the user
            // turns reasoning on, so Default requests stay unchanged.
            parameters["include_reasoning"] = false
        }

        if (selectedLevel != nil && nullTemperatureProviders.contains(provider))
            || (provider == .openAI && openAIEffortRequiredModels.contains(modelName))
        {
            parameters["temperature"] = NSNull()
        }

        return parameters.isEmpty ? nil : parameters
    }

    // MARK: - Request timeout

    struct RequestTimeout: Equatable {
        let seconds: TimeInterval
        let retryOnTimeout: Bool
    }

    /// The minimum request timeout for a picked level, set well above the slowest
    /// responses measured live on 2026-11-17 (VoiceInk's Default prompt and a
    /// 121-word dictation). `nil` keeps the configured timeout, for levels that
    /// answer as fast as Default.
    static func timeoutFloor(for level: ReasoningLevel) -> TimeInterval? {
        switch level {
        case .off, .minimal, .low:
            return nil
        case .medium:
            return 90
        case .high, .xhigh, .max, .enabled:
            return 120
        }
    }

    /// The timeout for a request at this level. A larger configured timeout wins, and
    /// Default (`nil`) keeps the configured timeout and retry setting unchanged.
    static func requestTimeout(
        _ timeout: TimeInterval,
        retryOnTimeout: Bool,
        level: ReasoningLevel?
    ) -> RequestTimeout {
        guard let level, let floor = timeoutFloor(for: level), floor > timeout else {
            return RequestTimeout(seconds: timeout, retryOnTimeout: retryOnTimeout)
        }
        // LLMkit applies the timeout to each of up to three attempts, so a retry would
        // triple the wait. A reasoning request that outlasts the floor is unlikely to
        // finish on a second try.
        return RequestTimeout(seconds: floor, retryOnTimeout: false)
    }

    // MARK: - Ollama

    // Ollama returns 400 for `think: true` on models without the thinking capability. A
    // model whose `/api/show` lists named levels takes those instead of an on/off switch.
    private static func ollamaLevels(for modelInfo: ReasoningModelInfo) -> [ReasoningLevel] {
        if modelInfo.ollamaSupportsThinking == false {
            return []
        }
        let named = (modelInfo.ollamaThinkingLevels ?? []).compactMap(ReasoningLevel.init(effortValue:))
        let levels = ReasoningLevel.allCases.filter { $0 != .off && named.contains($0) }
        return levels.isEmpty ? [.enabled] : levels
    }

    /// Default keeps `think: false` for low latency.
    static func ollamaThink(level: ReasoningLevel?, modelName: String, modelInfo: ReasoningModelInfo) -> OllamaThink {
        switch effectiveLevel(level, for: .ollama, modelName: modelName, modelInfo: modelInfo) {
        case .enabled:
            return .enabled(true)
        case let level?:
            if let value = level.effortValue {
                return .level(value)
            }
            return .enabled(false)
        case nil:
            return .enabled(false)
        }
    }
}
