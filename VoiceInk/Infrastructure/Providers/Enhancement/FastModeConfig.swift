import Foundation
import VoiceInkChat

/// What Fast mode adds to one enhancement request, independent of the client that sends it.
/// `bodyFields` go into the top level of the JSON body, through `ChatRequest.extraBody`.
struct FastModeRequestParameters {
    var bodyFields: [String: JSONValue]
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
/// 2. `AIService.performChatCompletion` adds `parameters(...)` to every request's `extraBody`.
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
    static let appliedProviders: Set<AIProvider> = [.openAI, .bedrock]

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

    /// One line shown under the toggle while it's on, about the extra cost.
    static func costCaption(for provider: AIProvider) -> String {
        switch provider {
        case .openAI:
            return String(localized: "Uses OpenAI priority processing, billed at a higher rate than standard.")
        case .bedrock:
            return String(localized: "Uses Bedrock latency-optimized inference, billed at a higher rate.")
        default:
            return String(localized: "Uses priority processing, billed at a higher rate.")
        }
    }
}
