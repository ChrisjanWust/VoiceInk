import Foundation

// Maps a reasoning effort onto the Converse request fields each Amazon Bedrock model takes.
//
// This file only uses Foundation, so it can move with the Bedrock client into a standalone
// chat library. Efforts are the usual strings: "none", "low", "medium", "high", "xhigh", "max".

/// A JSON value for Converse `additionalModelRequestFields`.
indirect enum BedrockJSONValue: Encodable, Sendable, Equatable {
    case string(String)
    case int(Int)
    case object([String: BedrockJSONValue])

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .int(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        }
    }
}

/// The reasoning fields a Converse request sends for an effort.
struct BedrockReasoningFields: Sendable, Equatable {
    let additionalModelRequestFields: [String: BedrockJSONValue]
    /// `inferenceConfig.maxTokens`, when thinking needs more room than the model's default
    /// output limit. `nil` keeps the model's default.
    let maxTokens: Int?
}

/// What one model accepts: its efforts, Bedrock's default, and how each effort is sent.
struct BedrockReasoningOptions: Sendable, Equatable {
    enum Encoding: Sendable, Equatable {
        /// A string enum at `path`, listed in the model's catalog schema: `reasoning_config`
        /// (OpenAI, xAI, Moonshot K3.1) or `output_config.effort` (Claude 5.5, Fable 5.1+).
        case schemaEnum(path: [String])
        /// Claude 4.6 to 5: `thinking.type: adaptive` with `output_config.effort`, or
        /// `thinking.type: disabled` for "none".
        case anthropicAdaptive
        /// Claude 4.5: `thinking.type: enabled` with `budget_tokens`.
        case anthropicBudget
        /// Nova 2: `reasoningConfig.type: enabled` with `maxReasoningEffort`.
        case novaEffort
        /// A top-level `reasoning_effort` string (gpt-oss, DeepSeek, GLM, Nemotron 3).
        case reasoningEffort
    }

    let efforts: [String]
    /// What the model does with no reasoning fields, when known, for display.
    let defaultEffort: String?
    let encoding: Encoding
    let maxTokensDefault: Int?
    let maxTokensMaximum: Int?

    /// Claude 4.5's thinking budget for each effort.
    static let anthropicBudgets = ["low": 1_024, "medium": 4_096, "high": 16_384]

    /// Room for thinking, which can fill a small default output limit (Claude 4.7 and later
    /// default to 4,096 tokens) before the answer starts.
    static let thinkingOutputLimit = 32_000

    /// The fields to send, or `nil` when the model doesn't offer this effort.
    func fields(for effort: String) -> BedrockReasoningFields? {
        guard efforts.contains(effort) else { return nil }
        let off = effort == "none"
        let request: [String: BedrockJSONValue]
        switch encoding {
        case .schemaEnum(let path):
            request = Self.nested(path, .string(effort))
        case .anthropicAdaptive:
            request = off
                ? ["thinking": .object(["type": .string("disabled")])]
                : [
                    "thinking": .object(["type": .string("adaptive")]),
                    "output_config": .object(["effort": .string(effort)]),
                ]
        case .anthropicBudget:
            guard let budget = Self.anthropicBudgets[effort] else { return nil }
            request = ["thinking": .object(["type": .string("enabled"), "budget_tokens": .int(budget)])]
        case .novaEffort:
            request = ["reasoningConfig": .object(["type": .string("enabled"), "maxReasoningEffort": .string(effort)])]
        case .reasoningEffort:
            request = ["reasoning_effort": .string(effort)]
        }

        var maxTokens: Int?
        if !off, let maxTokensDefault, maxTokensDefault < Self.thinkingOutputLimit {
            maxTokens = min(Self.thinkingOutputLimit, maxTokensMaximum ?? Self.thinkingOutputLimit)
        }
        return BedrockReasoningFields(additionalModelRequestFields: request, maxTokens: maxTokens)
    }

    private static func nested(_ path: [String], _ value: BedrockJSONValue) -> [String: BedrockJSONValue] {
        guard let first = path.first else { return [:] }
        if path.count == 1 { return [first: value] }
        return [first: .object(nested(Array(path.dropFirst()), value))]
    }
}

enum BedrockReasoningFieldMap {
    /// The reasoning options for a model, or `nil` when it has no reasoning control VoiceInk
    /// can send.
    ///
    /// Models whose catalog schema lists a `reasoning_config` or `output_config.effort` enum
    /// use it. Others use `nativeFamilies`, because their schema's `reasoningConfig` is not
    /// what the request accepts.
    ///
    /// - Parameters:
    ///   - modelID: a foundation model ID, or an inference profile ID such as `global.…`.
    ///   - schema: the catalog's `converse.additionalRequestFieldsSchema`, when known.
    static func options(
        modelID: String,
        schema: String?,
        maxTokensDefault: Int? = nil,
        maxTokensMaximum: Int? = nil
    ) -> BedrockReasoningOptions? {
        let foundationID = foundationModelID(for: modelID)
        let tooSlow = tooSlowEfforts[foundationID] ?? []

        let efforts: [String]
        let defaultEffort: String?
        let encoding: BedrockReasoningOptions.Encoding
        if let schemaEnum = SchemaEnum(schema: schema) {
            efforts = schemaEnum.values
            defaultEffort = schemaEnum.defaultValue
            encoding = .schemaEnum(path: schemaEnum.path)
        } else if let family = nativeFamilies[foundationID] {
            (efforts, defaultEffort, encoding) = family
        } else {
            return nil
        }

        let offered = efforts.filter { !tooSlow.contains($0) }
        guard !offered.isEmpty else { return nil }
        return BedrockReasoningOptions(
            efforts: offered,
            defaultEffort: defaultEffort,
            encoding: encoding,
            maxTokensDefault: maxTokensDefault,
            maxTokensMaximum: maxTokensMaximum
        )
    }

    /// `global.xai.grok-4.7` -> `xai.grok-4.7`; foundation model IDs are returned unchanged.
    static func foundationModelID(for modelID: String) -> String {
        let trimmed = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        let profilePrefixes: Set<String> = ["global", "us", "us-gov", "eu", "apac", "au", "ca", "jp"]
        guard let dot = trimmed.firstIndex(of: "."), profilePrefixes.contains(trimmed[..<dot].lowercased()) else {
            return trimmed
        }
        return String(trimmed[trimmed.index(after: dot)...])
    }

    // MARK: - Native families

    // Models whose catalog schema doesn't describe their reasoning field, with the native
    // field each takes. Every effort here returned clean text in live tests on 2026-11-18
    // (us-east-2, VoiceInk's Default prompt, 3 runs each). Efforts that were rejected, changed
    // nothing, leaked reasoning into the text or were too slow are left out.
    //
    // No working control, so not listed: Gemma 3 and Voxtral Small don't reason; MiniMax M2.5
    // reasons the same with every field tried; Qwen3 "high" ran past 150 s and Qwen3 VL wrote
    // its reasoning into the text; Kimi K2.5 "high" took a 66 s median and GLM 5 "high" a 47 s
    // median; DeepSeek V3.2 and Nemotron Nano 3 added a preamble at "high", and Nano 3 leaked
    // reasoning into the text at "medium".
    static let nativeFamilies: [String: (efforts: [String], defaultEffort: String?, encoding: BedrockReasoningOptions.Encoding)] = {
        var families: [String: (efforts: [String], defaultEffort: String?, encoding: BedrockReasoningOptions.Encoding)] = [:]
        // Claude 4.6 rejects "xhigh"; Fable 5 rejects `thinking.type: disabled`.
        let adaptive: [String: [String]] = [
            "anthropic.claude-sonnet-4-6": ["none", "low", "medium", "high", "max"],
            "anthropic.claude-opus-4-6-v1": ["none", "low", "medium", "high", "max"],
            "anthropic.claude-opus-4-7": ["none", "low", "medium", "high", "xhigh", "max"],
            "anthropic.claude-opus-4-8": ["none", "low", "medium", "high", "xhigh", "max"],
            "anthropic.claude-fable-5": ["low", "medium", "high", "xhigh", "max"],
            "anthropic.claude-sonnet-5": ["none", "low", "medium", "high", "xhigh", "max"],
            "anthropic.claude-opus-5": ["none", "low", "medium", "high", "xhigh", "max"],
        ]
        for (model, efforts) in adaptive {
            families[model] = (efforts, nil, .anthropicAdaptive)
        }
        // Reject adaptive thinking, and don't think by default.
        for model in ["anthropic.claude-haiku-4-5-20251001-v1:0", "anthropic.claude-opus-4-5-20251101-v1:0"] {
            families[model] = (["low", "medium", "high"], "none", .anthropicBudget)
        }
        // Off by default. "medium" took 31-65 s and "high" 114 s.
        families["amazon.nova-2-lite-v1:0"] = (["low"], "none", .novaEffort)
        // gpt-oss reasons at OpenAI's default, "medium", unless told otherwise.
        for model in ["openai.gpt-oss-120b-1:0", "openai.gpt-oss-20b-1:0", "openai.gpt-oss-safeguard-120b", "openai.gpt-oss-safeguard-20b"] {
            families[model] = (["low", "medium", "high"], "medium", .reasoningEffort)
        }
        // Only "high" turns reasoning on; "low" and "medium" answered exactly like Default.
        for model in ["deepseek.v3-v1:0", "zai.glm-4.7", "zai.glm-4.7-flash", "nvidia.nemotron-super-3-120b"] {
            families[model] = (["high"], "none", .reasoningEffort)
        }
        return families
    }()

    // Efforts too slow for dictation in live tests (median above ~45 s or max above ~120 s,
    // the rule in `ReasoningConfig.tooSlowLevels`): Claude Opus 5.5 "max" (65 s median),
    // Opus 4.8 "max" (54 s) and Sonnet 5 "max" (114 s).
    static let tooSlowEfforts: [String: Set<String>] = [
        "anthropic.claude-opus-5-5": ["max"],
        "anthropic.claude-opus-4-8": ["max"],
        "anthropic.claude-sonnet-5": ["max"],
    ]

    // MARK: - Schema enums

    private struct SchemaEnum {
        let path: [String]
        let values: [String]
        let defaultValue: String?

        init?(schema: String?) {
            guard let schema, let data = schema.data(using: .utf8),
                let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            else {
                return nil
            }
            if let effort = Self.stringEnum(root["reasoning_config"]) {
                path = ["reasoning_config"]
                (values, defaultValue) = effort
            } else if let outputConfig = root["output_config"] as? [String: Any],
                let effort = Self.stringEnum(outputConfig["effort"])
            {
                path = ["output_config", "effort"]
                (values, defaultValue) = effort
            } else {
                return nil
            }
        }

        private static func stringEnum(_ node: Any?) -> (values: [String], defaultValue: String?)? {
            guard let node = node as? [String: Any], node["type"] as? String == "enum",
                let values = node["enum"] as? [String], !values.isEmpty
            else {
                return nil
            }
            return (values, node["default"] as? String)
        }
    }
}
