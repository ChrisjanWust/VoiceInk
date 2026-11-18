import Foundation

/// Together AI, GMI Cloud, Baseten, Fireworks AI and Vercel AI Gateway: OpenAI-compatible
/// hosts of mostly open models. Requests go through LLMkit's `OpenAILLMClient` like Groq and
/// Cerebras. This file holds what differs per host: built-in models, models that fail
/// VoiceInk's request, and each model's reasoning control.
///
/// Many of these models reason by default, which takes 5–100 s for a dictation. Where a model
/// can turn that off, Default sends `reasoning_effort: "none"` (or the lowest effort, for
/// models that reject `none`), the same low-latency request Cerebras and Groq Qwen get.
/// Reasoning comes back in `reasoning_content` or `reasoning`, which LLMkit doesn't read, and
/// `AIEnhancementOutputFilter` drops any `<think>` block left in `content`.
///
/// Every table entry was measured live on 2026-11-18 with VoiceInk's Default prompt and a
/// 106-word dictation. A model that isn't listed (one the host added later) sends no
/// reasoning parameter and offers no levels, so it runs on the host's own default.
enum HostedModelProviders {
    static let providers: Set<AIProvider> = [.togetherAI, .gmiCloud, .baseten, .fireworks, .vercelAIGateway]

    static func apiKeyURL(for provider: AIProvider) -> URL? {
        switch provider {
        case .togetherAI: return URL(string: "https://api.together.ai/settings/api-keys")
        case .gmiCloud: return URL(string: "https://console.gmicloud.ai/user-setting/organization/api-key")
        case .baseten: return URL(string: "https://app.baseten.co/settings/api_keys")
        case .fireworks: return URL(string: "https://app.fireworks.ai/settings/users/api-keys")
        case .vercelAIGateway: return URL(string: "https://vercel.com/d?to=%2F%5Bteam%5D%2F%7E%2Fai-gateway%2Fapi-keys")
        default: return nil
        }
    }

    /// Neutral SF Symbols: these providers ship without a bundled logo.
    static func symbolName(for provider: AIProvider) -> String? {
        switch provider {
        case .togetherAI: return "square.grid.3x3.fill"
        case .gmiCloud: return "server.rack"
        case .baseten: return "cube.fill"
        case .fireworks: return "flame.fill"
        case .vercelAIGateway: return "point.3.connected.trianglepath.dotted"
        default: return nil
        }
    }

    // MARK: - Models

    /// One model's reasoning control, measured live.
    struct ModelReasoning: Equatable {
        /// What Default sends as `reasoning_effort`. `nil` sends nothing.
        let defaultEffort: String?
        /// The levels the model accepted, minus any too slow for dictation.
        let levels: [ReasoningLevel]

        /// Reasoning can be turned off with `none`, and Default does that.
        static func off(_ levels: [ReasoningLevel]) -> ModelReasoning {
            ModelReasoning(defaultEffort: "none", levels: [.off] + levels)
        }

        /// No reasoning parameter on Default.
        static func hostDefault(_ levels: [ReasoningLevel] = []) -> ModelReasoning {
            ModelReasoning(defaultEffort: nil, levels: levels)
        }
    }

    static func builtInModels(for provider: AIProvider) -> [String] {
        switch provider {
        case .togetherAI: return togetherAIBuiltInModels
        case .gmiCloud: return gmiCloudBuiltInModels
        case .baseten: return basetenBuiltInModels
        case .fireworks: return fireworksBuiltInModels
        case .vercelAIGateway: return vercelAIGatewayBuiltInModels
        default: return []
        }
    }

    /// Listed models that fail VoiceInk's request, so they're left out of the live list.
    static func isUnusable(_ modelName: String, for provider: AIProvider) -> Bool {
        let id = modelName.lowercased()
        switch provider {
        case .togetherAI:
            return togetherAIUnusableModels.contains(id)
                // Together serves Alibaba's hosted Qwen models (Plus, Max, Flash) only with
                // `stream: true`, and LLMkit doesn't stream.
                || id.range(of: #"^qwen/qwen[0-9.]+-(plus|max|flash)"#, options: .regularExpression) != nil
        case .gmiCloud:
            return gmiCloudUnusableModels.contains(id)
        case .baseten:
            return basetenUnusableModels.contains(id) || id.hasPrefix("decision-models/")
        case .fireworks:
            return fireworksUnusableModels.contains(id)
        case .vercelAIGateway:
            return vercelAIGatewayUnusableTokens.contains { id.contains($0) }
        default:
            return false
        }
    }

    static func reasoning(for provider: AIProvider, modelName: String) -> ModelReasoning? {
        switch provider {
        case .togetherAI: return togetherAIReasoning[modelName]
        case .gmiCloud: return gmiCloudReasoning[modelName]
        case .baseten: return basetenReasoning[modelName]
        case .fireworks: return fireworksReasoning[modelName]
        case .vercelAIGateway: return vercelAIGatewayReasoning(for: modelName)
        default: return nil
        }
    }

    /// The levels offered for this model. Empty for models not measured.
    static func levels(for provider: AIProvider, modelName: String) -> [ReasoningLevel] {
        reasoning(for: provider, modelName: modelName)?.levels ?? []
    }

    /// The `reasoning_effort` Default sends, or `nil` for none.
    static func defaultEffort(for provider: AIProvider, modelName: String) -> String? {
        reasoning(for: provider, modelName: modelName)?.defaultEffort
    }

    // MARK: - Vercel AI Gateway

    /// The gateway maps `reasoning_effort` onto each upstream provider's own control, and its
    /// catalog lists each model's efforts and whether reasoning can be switched off
    /// (`reasoning_options`). In live tests on 130 models, `none` returned 200 on all but two,
    /// turned reasoning off on every model that reasons, and changed nothing on models that
    /// don't. So Default sends `none` unless the catalog lists efforts without `none` or a
    /// toggle (then the lowest listed effort, which the catalog promises), with exceptions
    /// from the live tests. The levels come from the catalog, minus those measured too slow.
    private static func vercelAIGatewayReasoning(for modelName: String) -> ModelReasoning {
        let options = VercelAIGatewayReasoningStore.shared.options(for: modelName)
            ?? vercelAIGatewayBuiltInOptions[modelName]
        let defaultEffort = vercelAIGatewayDefaultEffort(for: modelName, options: options)
        let tooSlow = vercelAIGatewayTooSlowLevels[modelName] ?? []
        let efforts = Set((options?.efforts ?? []).compactMap(ReasoningLevel.init(effortValue:)))
        let levels = ReasoningLevel.allCases.filter { level in
            if level == .off { return defaultEffort == "none" }
            return efforts.contains(level) && !tooSlow.contains(level)
        }
        return ModelReasoning(defaultEffort: defaultEffort, levels: levels)
    }

    private static func vercelAIGatewayDefaultEffort(
        for modelName: String,
        options: VercelAIGatewayReasoningOptions?
    ) -> String? {
        if vercelAIGatewayRejectsNone.contains(modelName) {
            return vercelAIGatewayDefaultEffortOverrides[modelName]
        }
        guard let options, !options.efforts.isEmpty else {
            // No efforts listed: either no reasoning control, where `none` is accepted and
            // changes nothing, or only a toggle, which `none` switches off.
            return "none"
        }
        if options.canToggle || options.efforts.contains("none") || vercelAIGatewayAcceptsNone.contains(modelName) {
            return "none"
        }
        return ReasoningLevel.allCases.first { level in
            level.effortValue.map(options.efforts.contains) == true
        }?.effortValue
    }
}

/// What a Vercel AI Gateway catalog entry says about reasoning.
struct VercelAIGatewayReasoningOptions: Equatable, Sendable {
    /// `reasoning_effort` values the model takes, such as `none`, `low` or `xhigh`.
    let efforts: [String]
    /// Whether reasoning can be switched on and off.
    let canToggle: Bool
}

extension ProviderModelCatalog {
    static func isUnusableHostedModel(_ modelName: String, for provider: AIProvider) -> Bool {
        HostedModelProviders.isUnusable(modelName, for: provider)
    }
}

/// The reasoning options from the last Vercel AI Gateway catalog fetch, keyed by model ID.
final class VercelAIGatewayReasoningStore: @unchecked Sendable {
    static let shared = VercelAIGatewayReasoningStore()

    private static let defaultsKey = "VercelAIGatewayReasoningOptions"
    private static let effortsKey = "efforts"
    private static let toggleKey = "toggle"
    private let lock = NSLock()
    private let defaults = UserDefaults.standard
    private var cache: [String: VercelAIGatewayReasoningOptions]?

    private init() {}

    func options(for modelName: String) -> VercelAIGatewayReasoningOptions? {
        lock.withLock {
            if cache == nil {
                let saved = defaults.dictionary(forKey: Self.defaultsKey) as? [String: [String: Any]] ?? [:]
                cache = saved.mapValues { entry in
                    VercelAIGatewayReasoningOptions(
                        efforts: entry[Self.effortsKey] as? [String] ?? [],
                        canToggle: entry[Self.toggleKey] as? Bool ?? false
                    )
                }
            }
            return cache?[modelName]
        }
    }

    func save(_ models: [ProviderModelCatalog.VercelAIGatewayModel]) {
        let options = Dictionary(models.map { ($0.id, $0.reasoning) }, uniquingKeysWith: { first, _ in first })
        let saved = options.mapValues { [Self.effortsKey: $0.efforts, Self.toggleKey: $0.canToggle] as [String: Any] }
        lock.withLock {
            cache = options
            defaults.set(saved, forKey: Self.defaultsKey)
        }
    }
}
