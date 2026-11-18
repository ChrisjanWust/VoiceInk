import Foundation

/// Together AI, GMI Cloud, Baseten, Fireworks AI and Vercel AI Gateway: OpenAI-compatible
/// hosts of mostly open models. Requests go through VoiceInkChat's `OpenAICompatibleChatClient`
/// like Groq and Cerebras. This file holds what differs per host: built-in models, models that fail
/// VoiceInk's request, and each model's reasoning control.
///
/// Many of these models reason by default, which takes 5–100 s for a dictation. Where a model
/// can turn that off, Default sends `reasoning_effort: "none"` (or the lowest effort, for
/// models that reject `none`), the same low-latency request Cerebras and Groq Qwen get.
/// Reasoning comes back in `reasoning_content` or `reasoning`, which VoiceInkChat doesn't read,
/// and it strips any `<think>` block left in `content`, as `AIEnhancementOutputFilter` does.
///
/// The tables were measured live on 2026-11-18 with VoiceInk's Default prompt and a 106-word
/// dictation. A model that isn't listed gets Default only: no reasoning parameter and no
/// levels, like an unknown Groq model. That covers models measured that way and ones a host
/// adds later.
enum HostedModelProviders {
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
                // `stream: true`, and VoiceInk doesn't stream.
                || id.range(of: #"^qwen/qwen[0-9.]+-(plus|max|flash)"#, options: .regularExpression) != nil
        case .gmiCloud:
            return gmiCloudUnusableModels.contains(id)
        case .baseten:
            // Baseten's `decision-models/*` are classifiers, not chat models.
            return id.hasPrefix("decision-models/")
        case .fireworks:
            return fireworksUnusableModels.contains(id)
        case .vercelAIGateway:
            return vercelAIGatewayUnusableModels.contains(id)
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
        case .vercelAIGateway: return vercelAIGatewayReasoning[modelName]
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
}
