import Foundation

/// What Fast mode adds to one enhancement request, independent of the client that sends it.
/// A request path merges `bodyFields` into the top level of its JSON body and sets `headers`.
struct FastModeRequestParameters {
    var bodyFields: [String: Any]
    var headers: [String: String] = [:]
}

/// Fast mode for enhancement requests: a provider's faster, higher-priced processing tier.
///
/// Off (the default) adds nothing, so the request is exactly VoiceInk's standard request. On
/// adds the provider's own fast-tier fields, and only for models where live measurements with
/// VoiceInk's Default prompt and a 115-word dictation showed the tier is accepted and lowers
/// latency. Every other model hides the toggle.
///
/// Fast mode only changes the processing tier, so it combines with any reasoning level and
/// leaves the reasoning timeout floors unchanged.
///
/// Adding a provider:
/// 1. Return its fields from `mapping(for:modelName:)`, for its measured model set.
/// 2. Apply `parameters(...)` in that provider's request path in `AIService.performChatCompletion`.
/// 3. Add it to `appliedProviders`, which is what shows the toggle.
enum FastModeConfig {

    // MARK: - OpenAI

    /// Models that accept `service_tier: "priority"` (OpenAI's Fast mode, formerly Priority
    /// processing) on Chat Completions and answered faster with it. Measured live on
    /// 2026-11-18: 6 to 10 interleaved runs per model, median 11-68% lower.
    ///
    /// Left out: gpt-5.4-nano (served on the default tier), and gpt-6-sol and gpt-5.6-sol
    /// (accepted, but no lower median over 10 runs).
    static let openAIPriorityModels: Set<String> = [
        "gpt-6-luna",
        "gpt-6.1-terra",
        "gpt-6.1-sol",
        "gpt-6.2-sol",
        "gpt-6.2-astra",
        "gpt-6-astra",
        "gpt-5.6-luna",
        "gpt-5.6-terra",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.2",
    ]

    // MARK: - Gemini

    /// Models that accept `service_tier: "priority"` (Priority inference) on the Interactions
    /// API and answered faster with it in two sweeps on 2026-11-18 (16 interleaved runs per
    /// model): median 7-69% lower, with the standard tier's long queue spikes gone for Gemini 4
    /// Carbon and Argon. Left out: the other Flash models and Gemini 4 Hydrogen, which were no
    /// faster or slower.
    static let geminiPriorityModels: Set<String> = [
        "gemini-4-carbon",
        "gemini-4-argon",
        "gemini-3.1-pro-preview",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    ]

    // MARK: - Anthropic

    /// Models that take `speed: "fast"` on the Messages API without a beta header, measured
    /// live on 2026-11-18: Claude Opus 5.5 median 1.76 s to 0.83 s over 8 interleaved runs.
    /// Sonnet 5.5 and Haiku 5.5 reject `speed`. Opus 5 needs the fast-mode beta header and a
    /// waitlisted rate limit, and Opus 4.8 loses fast mode on 2026-11-30, so neither is listed.
    static let anthropicFastModels: Set<String> = [
        "claude-opus-5-5"
    ]

    // MARK: - Amazon Bedrock

    /// Model IDs that take `performanceConfig: { latency: "optimized" }` on Converse, measured
    /// live on 2026-11-18: Nova Pro median 1.21 s to 0.99 s over 8 interleaved runs. Only the
    /// `us.` inference profile accepts it (the in-Region ID returns 400 in us-east-1, us-east-2
    /// and us-west-2), and that profile only runs from US Regions. No current model lists
    /// `latencyOptimizationSupported: true` in ListFoundationModels, so the ID is listed here.
    static let bedrockLatencyOptimizedModels: Set<String> = [
        "us.amazon.nova-pro-v1:0"
    ]

    // MARK: - Shared

    /// Providers whose request path applies the parameters, so the toggle shows for them.
    ///
    /// Not here yet: Gemini requests go through LLMkit's `GeminiLLMClient`, which can't add body
    /// fields; Anthropic and Bedrock wait for their in-app clients to carry the fields. Add each
    /// once its request path applies `parameters(...)`.
    static let appliedProviders: Set<AIProvider> = [.openAI]

    /// The provider's fast-tier fields for this model, or `nil` when the model has none.
    ///
    /// Not mapped: Groq's faster Performance tier is enterprise-only, Cerebras service tiers are
    /// a private preview on dedicated endpoints (shared inference ignores `service_tier`),
    /// Mistral's Priority tier needs a sales contract, and Ollama and custom endpoints have no
    /// standard fast tier.
    static func mapping(for provider: AIProvider, modelName: String) -> FastModeRequestParameters? {
        switch provider {
        case .openAI where openAIPriorityModels.contains(modelName):
            return FastModeRequestParameters(bodyFields: ["service_tier": "priority"])
        case .gemini where geminiPriorityModels.contains(modelName):
            return FastModeRequestParameters(bodyFields: ["service_tier": "priority"])
        case .anthropic where anthropicFastModels.contains(modelName):
            return FastModeRequestParameters(bodyFields: ["speed": "fast"])
        case .bedrock where bedrockLatencyOptimizedModels.contains(modelName):
            return FastModeRequestParameters(bodyFields: ["performanceConfig": ["latency": "optimized"]])
        default:
            return nil
        }
    }

    /// Whether this model has a fast option VoiceInk sends. The toggle is hidden otherwise.
    static func isSupported(for provider: AIProvider, modelName: String) -> Bool {
        appliedProviders.contains(provider) && mapping(for: provider, modelName: modelName) != nil
    }

    /// What to add to a request. `nil` when Fast mode is off or the model doesn't support it,
    /// so a saved choice for a model that lost support is ignored.
    static func parameters(
        for provider: AIProvider,
        modelName: String,
        enabled: Bool
    ) -> FastModeRequestParameters? {
        guard enabled, isSupported(for: provider, modelName: modelName) else {
            return nil
        }
        return mapping(for: provider, modelName: modelName)
    }

    /// Merges the body fields into an LLMkit `extraBody`. Without parameters, `extraBody` is
    /// returned unchanged. LLMkit can't set headers, so only header-free mappings go this way.
    static func merging(_ parameters: FastModeRequestParameters?, into extraBody: [String: Any]?) -> [String: Any]? {
        guard let parameters else { return extraBody }
        assert(parameters.headers.isEmpty, "LLMkit's extraBody can't carry Fast mode headers")
        return (extraBody ?? [:]).merging(parameters.bodyFields) { _, fast in fast }
    }

    /// One line shown under the toggle while it's on, about the extra cost.
    static func costCaption(for provider: AIProvider) -> String {
        switch provider {
        case .openAI:
            return String(localized: "Uses OpenAI priority processing, billed at a higher rate than standard.")
        case .gemini:
            return String(localized: "Uses Gemini priority inference, billed at 75–100% above the standard rate.")
        case .anthropic:
            return String(localized: "Uses Anthropic fast mode, billed at a higher rate than standard.")
        case .bedrock:
            return String(localized: "Uses Bedrock latency-optimized inference, billed at a higher rate.")
        default:
            return String(localized: "Uses priority processing, billed at a higher rate.")
        }
    }
}
