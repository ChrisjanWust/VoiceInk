import Foundation

// Measured live on 2026-11-18: VoiceInk's Default prompt and a 106-word dictation, sent the way
// AIService sends it. A level is listed when the model returned a clean answer at it, minus levels
// too slow for dictation (over ~45 s, or ~120 s at most, as for the other providers). Models
// without an entry run on the host's own default with no levels.
extension HostedModelProviders {
    // MARK: - Together AI

    static let togetherAIBuiltInModels: [String] = [
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "zai-org/GLM-5.3-Flash",
        "moonshotai/Kimi-K3",
        "nvidia/NVIDIA-Nemotron-3.5-Super-VL",
        "openai/gpt-oss-120b",
    ]

    static let togetherAIUnusableModels: Set<String> = [
        "arize-ai/qwen-2-1.5b-instruct",
        "together/tev1-4b-experimental",
    ]

    static let togetherAIReasoning: [String: ModelReasoning] = [
        "MiniMaxAI/MiniMax-M3.1-Flash": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "Prism-ML/Ternary-Bonsai-27B": ModelReasoning(defaultEffort: nil, levels: [.low]),
        "Qwen/Qwen3.5-9B": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .high, .xhigh]),
        "deepseek-ai/DeepSeek-V4-Pro-0813": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "deepseek-ai/DeepSeek-V4.1-Flash": ModelReasoning(defaultEffort: "none", levels: [.off, .low, .high, .xhigh, .max]),
        "deepseek-ai/DeepSeek-V4.1-Flash-1017": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "elorian-labs/glimpse-1.0": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .xhigh]),
        "meta-models/Muse-Glimmer-30B": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "meta-models/Muse-Spark-Open-816B-A42B": ModelReasoning(defaultEffort: "none", levels: [.off, .low, .medium, .high, .xhigh]),
        "moonshotai/Kimi-K3": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "moonshotai/Kimi-K3.1-Pro": ModelReasoning(defaultEffort: "none", levels: [.off, .low, .medium, .high, .max]),
        "nvidia/NVIDIA-Nemotron-3.5-Super-VL": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "openai/gpt-oss-120b": ModelReasoning(defaultEffort: "low", levels: [.minimal, .low, .medium, .high, .xhigh, .max]),
        "reflection/Beam-1.0-501B-A23B": ModelReasoning(defaultEffort: "minimal", levels: [.minimal, .low, .medium, .high, .xhigh, .max]),
        "ricursive-ai/Ripple-65B-Preview": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .medium, .high, .xhigh, .max]),
        "thinkingmachines/Inkling": ModelReasoning(defaultEffort: "minimal", levels: [.minimal, .low, .medium, .high, .max]),
        "thinkingmachines/Inkling-2-Small": ModelReasoning(defaultEffort: "minimal", levels: [.minimal, .low, .medium, .high, .xhigh, .max]),
        "zai-org/GLM-5.2": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "zai-org/GLM-5.3": ModelReasoning(defaultEffort: "low", levels: [.low, .high, .max]),
        "zai-org/GLM-5.3-Flash": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
    ]

    // MARK: - GMI Cloud

    static let gmiCloudBuiltInModels: [String] = [
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "Qwen/Qwen3.6-35B-A3B",
        "zai-org/GLM-5.3",
        "openai/gpt-6-luna",
    ]

    static let gmiCloudUnusableModels: Set<String> = [
        "google/gemini-3-flash-preview",
        "moonshotai/kimi-k2-thinking",
        "moonshotai/kimi-k2.7-code-highspeed",
        "moonshotai/kimi-k3.1-pro",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "openai/gpt-5.6-luna",
        "qwen/qwen3-235b-a22b-instruct-2507-fp8",
        "qwen/qwen3-235b-a22b-thinking-2507-fp8",
        "qwen/qwen3-coder-480b-a35b-instruct-fp8",
        "qwen/qwen3-next-80b-a3b-thinking",
        "stepfun-ai/step-3.7-flash",
        "stepfun-ai/step-5-preview",
        "tencent/hy4-preview",
        "zai-org/glm-4.7-fp8",
    ]

    static let gmiCloudReasoning: [String: ModelReasoning] = [
        "MiniMaxAI/MiniMax-M2.5": ModelReasoning(defaultEffort: "minimal", levels: [.minimal, .low, .medium, .high, .xhigh, .max]),
        "MiniMaxAI/MiniMax-M2.7": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "MiniMaxAI/MiniMax-M3": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "MiniMaxAI/MiniMax-M3.1-Flash": ModelReasoning(defaultEffort: "minimal", levels: [.minimal, .low, .medium, .high, .xhigh, .max]),
        "Qwen/Qwen3.5-122B-A10B": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .medium, .xhigh]),
        "Qwen/Qwen3.5-27B": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal]),
        "Qwen/Qwen3.5-35B-A3B": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal]),
        "Qwen/Qwen3.5-397B-A17B": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal]),
        "Qwen/Qwen3.6-35B-A3B": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh]),
        "Qwen/Qwen3.6-Plus": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .high]),
        "Qwen/Qwen3.6-Plus-2026-04-02": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal]),
        "Qwen/Qwen3.7-Max": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh]),
        "Qwen/Qwen3.8-27B": ModelReasoning(defaultEffort: "none", levels: [.off]),
        "Qwen/Qwen3.8-Flash": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "Qwen/Qwen3.8-Max": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "Qwen/Qwen3.8-Max-0902": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "XiaomiMiMo/MiMo-V2.6-Flash": ModelReasoning(defaultEffort: "none", levels: [.off, .low, .medium, .high, .max]),
        "XiaomiMiMo/MiMo-V2.6-Pro": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "deepseek-ai/DeepSeek-V3-0324": ModelReasoning(defaultEffort: "low", levels: [.low, .medium, .max]),
        "deepseek-ai/DeepSeek-V3.2": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh]),
        "deepseek-ai/DeepSeek-V4-Flash": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "deepseek-ai/DeepSeek-V4-Flash-0731": ModelReasoning(defaultEffort: nil, levels: [.minimal, .low, .medium, .high, .xhigh, .max]),
        "deepseek-ai/DeepSeek-V4-Pro": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "deepseek-ai/DeepSeek-V4-Pro-0813": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "deepseek-ai/DeepSeek-V4.1-Flash": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "deepseek-ai/deepseek-v4-flash-vision-exp": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "deepseek-ai/deepseek-v4.1-flash-1017": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .high, .xhigh, .max]),
        "gmicloud/Ripple-65B-Preview": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "google/gemini-3.1-flash-lite-preview": ModelReasoning(defaultEffort: nil, levels: [.minimal, .low, .medium, .high, .xhigh, .max]),
        "google/gemini-3.1-pro-preview": ModelReasoning(defaultEffort: "minimal", levels: [.minimal, .low, .medium, .xhigh, .max]),
        "google/gemini-3.5-flash": ModelReasoning(defaultEffort: "low", levels: [.low, .max]),
        // Erratic on GMI: the same request took 1.5–151 s, and High and Max ran over 2 minutes.
        "google/gemini-3.5-flash-lite": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium]),
        "google/gemini-3.6-flash": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "google/gemini-3.7-flash": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "google/gemini-3.8-flash": ModelReasoning(defaultEffort: "minimal", levels: [.minimal, .low, .medium, .high]),
        "google/gemini-4-argon": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium]),
        "google/gemini-4-carbon": ModelReasoning(defaultEffort: "none", levels: [.off]),
        "google/gemini-4-hydrogen": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "google/gemma-4-26b-a4b-it": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .medium, .xhigh]),
        "moonshotai/Kimi-K2.5": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal]),
        "moonshotai/kimi-k2.7-code": ModelReasoning(defaultEffort: nil, levels: [.minimal, .low, .medium, .high, .xhigh, .max]),
        "moonshotai/kimi-k3": ModelReasoning(defaultEffort: nil, levels: [.medium, .max]),
        "openai/gpt-5.5": ModelReasoning(defaultEffort: "none", levels: [.off, .low, .medium, .high, .xhigh]),
        "openai/gpt-5.6-sol": ModelReasoning(defaultEffort: "none", levels: [.off, .low, .medium, .high, .xhigh, .max]),
        "openai/gpt-5.6-terra": ModelReasoning(defaultEffort: "none", levels: [.off, .low, .medium, .high, .xhigh, .max]),
        "openai/gpt-6-astra": ModelReasoning(defaultEffort: "low", levels: [.low, .medium, .high, .xhigh]),
        "openai/gpt-6-luna": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "openai/gpt-6-sol": ModelReasoning(defaultEffort: "low", levels: [.low, .medium, .high, .xhigh, .max]),
        "openai/gpt-6.1-sol": ModelReasoning(defaultEffort: "low", levels: [.low, .medium, .high, .xhigh, .max]),
        "tencent/hy3-preview": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .high, .xhigh, .max]),
        "tencent/hy4": ModelReasoning(defaultEffort: "none", levels: [.off]),
        "xai/grok4.6": ModelReasoning(defaultEffort: "minimal", levels: [.minimal, .low, .medium, .high, .xhigh]),
        "xai/grok4.7": ModelReasoning(defaultEffort: "minimal", levels: [.minimal, .low, .medium, .high, .xhigh]),
        "zai-org/GLM-5-FP8": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "zai-org/GLM-5.1-FP8": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "zai-org/GLM-5.2-FP8": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "zai-org/GLM-5.3": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "zai-org/GLM-5.3-Flash": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
    ]

    // MARK: - Baseten

    static let basetenBuiltInModels: [String] = [
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "zai-org/GLM-5.3",
        "moonshotai/Kimi-K3",
        "openai/gpt-oss-120b",
    ]

    static let basetenUnusableModels: Set<String> = [
    ]

    static let basetenReasoning: [String: ModelReasoning] = [
        "deepseek-ai/DeepSeek-V4-Flash-0731": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "deepseek-ai/DeepSeek-V4-Pro-0813": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "deepseek-ai/DeepSeek-V4.1-Flash": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "deepseek-ai/DeepSeek-V4.1-Flash-1017": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "deepseek-ai/DeepSeek-V4.1-Flash-1017-Fast": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "deepseek-ai/DeepSeek-V4.1-Flash-Fast": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "meta-models/muse-spark-open": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "moonshotai/Kimi-K3": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "moonshotai/Kimi-K3.1-Pro": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "openai/gpt-oss-120b": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .max]),
        "reflection/Beam-1.0": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "zai-org/GLM-5.2-Fast": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "zai-org/GLM-5.3": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "zai-org/GLM-5.3-Fast": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "zai-org/GLM-5.3-Flash": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
    ]

    // MARK: - Fireworks AI

    static let fireworksBuiltInModels: [String] = [
        "accounts/fireworks/models/deepseek-v4p1-flash",
        "accounts/fireworks/models/glm-5p3-flash",
        "accounts/fireworks/models/minimax-m3p1-flash",
        "accounts/fireworks/models/kimi-k3",
        "accounts/fireworks/models/gpt-oss-120b",
    ]

    static let fireworksUnusableModels: Set<String> = [
        "accounts/fireworks/routers/deepseek-v4p1-flash-1017-spot",
        "accounts/fireworks/routers/glm-5p2-fast",
        "accounts/fireworks/routers/kimi-k3-spot",
    ]

    static let fireworksReasoning: [String: ModelReasoning] = [
        "accounts/fireworks/models/deepseek-v4p1-flash": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "accounts/fireworks/models/deepseek-v4p1-flash-1017": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "accounts/fireworks/models/ember-1": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "accounts/fireworks/models/glm-5p3": ModelReasoning(defaultEffort: "minimal", levels: [.minimal, .low, .medium, .high, .xhigh, .max]),
        "accounts/fireworks/models/glm-5p3-flash": ModelReasoning(defaultEffort: "minimal", levels: [.minimal, .low, .medium, .high, .xhigh, .max]),
        "accounts/fireworks/models/gpt-oss-120b": ModelReasoning(defaultEffort: "low", levels: [.minimal, .low, .medium, .high, .max]),
        "accounts/fireworks/models/inkling": ModelReasoning(defaultEffort: "none", levels: [.off, .low, .medium, .high, .xhigh, .max]),
        "accounts/fireworks/models/kimi-k3": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "accounts/fireworks/models/minimax-m3": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "accounts/fireworks/models/minimax-m3p1-flash": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "accounts/fireworks/models/nemotron-3-ultra-nvfp4": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .medium, .high, .xhigh, .max]),
        "accounts/fireworks/models/qwen3p8-2p4t-a95b": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "accounts/fireworks/models/qwen3p8-max": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "accounts/fireworks/routers/deepseek-v4p1-flash-1017-ultrafast": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "accounts/fireworks/routers/deepseek-v4p1-flash-spot": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "accounts/fireworks/routers/glm-5p3-fast": ModelReasoning(defaultEffort: "minimal", levels: [.minimal, .low, .medium, .high, .xhigh, .max]),
        "accounts/fireworks/routers/glm-5p3-spot": ModelReasoning(defaultEffort: "minimal", levels: [.minimal, .low, .medium, .high, .xhigh, .max]),
        "accounts/fireworks/routers/kimi-k3-fast": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "accounts/fireworks/routers/spot-deepseek-v4-flash": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high]),
        "auto": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
        "firerouter/auto": ModelReasoning(defaultEffort: "none", levels: [.off, .minimal, .low, .medium, .high, .xhigh, .max]),
    ]

    // MARK: - Vercel AI Gateway

    static let vercelAIGatewayBuiltInModels: [String] = [
        "deepseek/deepseek-v4.1-flash",
        "google/gemini-3.5-flash-lite",
        "anthropic/claude-haiku-5.5",
        "openai/gpt-6-luna",
        "alibaba/qwen3.8-27b",
        "zai/glm-5.3-flash",
    ]

    /// Catalog `reasoning_options` for the built-in models, used until the live catalog loads.
    static let vercelAIGatewayBuiltInOptions: [String: VercelAIGatewayReasoningOptions] = [
        "deepseek/deepseek-v4.1-flash": VercelAIGatewayReasoningOptions(efforts: ["none", "low", "high", "max"], canToggle: true),
        "google/gemini-3.5-flash-lite": VercelAIGatewayReasoningOptions(efforts: ["minimal", "low", "medium", "high"], canToggle: false),
        "anthropic/claude-haiku-5.5": VercelAIGatewayReasoningOptions(efforts: ["none", "low", "medium", "high", "xhigh", "max"], canToggle: true),
        "openai/gpt-6-luna": VercelAIGatewayReasoningOptions(efforts: ["none", "low", "medium", "high", "xhigh", "max"], canToggle: true),
        "alibaba/qwen3.8-27b": VercelAIGatewayReasoningOptions(efforts: ["none", "low", "medium", "xhigh"], canToggle: false),
        "zai/glm-5.3-flash": VercelAIGatewayReasoningOptions(efforts: ["low", "high", "max"], canToggle: false),
    ]

    /// Model IDs or ID fragments that fail VoiceInk's request or ignore its prompt.
    static let vercelAIGatewayUnusableTokens: [String] = [
        "alibaba/qwen-3-14b", // 400: asks for more output tokens than its context window
        "alibaba/qwen-3-30b",
        "alibaba/qwen3-max-thinking", // 404: listed but not served
        "inference-net/schematron",
        "morph/",
        "tencent/hy-mt2-lite",
        "scaledown/scaledown-extract",
        "stealth/",
        "nvidia/nemotron-nano-9b-v2",
        "tencent/hy4-preview",
    ]

    /// Return 400 for `reasoning_effort: "none"`, with what Default sends instead.
    static let vercelAIGatewayRejectsNone: Set<String> = [
        "interfaze/interfaze-beta",
        "mistral/mistral-large-4",
    ]
    static let vercelAIGatewayDefaultEffortOverrides: [String: String] = [
        "interfaze/interfaze-beta": "low",
    ]

    /// List efforts without `none` in the catalog but took `none` and answered faster.
    static let vercelAIGatewayAcceptsNone: Set<String> = [
        "amazon/nova-2-lite",
        "anthropic/claude-fable-5.2",
        "anthropic/claude-opus-5.5",
        "anthropic/claude-sonnet-5.5",
        "elorian/glimpse",
        "google/gemini-3.5-flash-lite",
        "google/gemini-3.8-flash",
        "google/gemini-4-hydrogen",
        "humansand/ampersand-v0.1",
        "inception/mercury-3",
        "liquid/lfm2.5-24b-a2b",
        "meta/muse-ember",
        "meta/muse-spark-1.1",
        "meta/muse-spark-1.3",
        "minimax/minimax-m2.7",
        "minimax/minimax-m3.1-flash",
        "moonshotai/kimi-k3",
        "moonshotai/kimi-k3.1-pro",
        "openai/gpt-5.5-pro",
        "openai/gpt-6-astra-fast",
        "openai/gpt-6.2-astra",
        "paperinstruments/ultramarine-1-flash",
        "reflection/beam",
        "reflection/beam-free",
        "sakana/fugu-max",
        "sakana/fugu-ultra-v2",
        "spacexai/grok-4.20-multi-agent",
        "spacexai/grok-4.5",
        "spacexai/grok-4.7",
        "stepfun/step-3.7-flash",
        "stepfun/step-5-preview",
        "zai/glm-5.3-flash",
        "zai/glm-5.3-flashx-legal",
    ]

    static let vercelAIGatewayTooSlowLevels: [String: Set<ReasoningLevel>] = [
        "mistral/mistral-large-4": [.high],
        "spacexai/grok-4.3": [.low],
    ]
}
