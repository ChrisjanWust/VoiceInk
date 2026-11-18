import Foundation

// Maps a reasoning effort onto the Converse request fields each Amazon Bedrock model takes.
//
// This file only uses Foundation, so it can move with the Bedrock client into a standalone
// chat library: callers pass a model ID and its catalog data, and get back JSON fields.

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

/// A reasoning effort, in increasing order. `off` turns thinking off where the model allows it.
enum BedrockReasoningEffort: String, CaseIterable, Sendable {
    case off
    case minimal
    case low
    case medium
    case high
    case xhigh
    case max

    /// The effort string most APIs use (`none` for `off`).
    var wireValue: String {
        self == .off ? "none" : rawValue
    }

    init?(wireValue: String) {
        let value = wireValue.lowercased()
        guard let effort = Self.allCases.first(where: { $0.wireValue == value }) else { return nil }
        self = effort
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
        /// `thinking.type: disabled` for off.
        case anthropicAdaptive
        /// Claude 4.5: `thinking.type: enabled` with a `budget_tokens` per effort.
        case anthropicBudget([BedrockReasoningEffort: Int])
        /// Nova 2: `reasoningConfig.type: enabled` with `maxReasoningEffort`.
        case novaEffort
        /// A top-level OpenAI-style `reasoning_effort` string (gpt-oss, DeepSeek, GLM, Kimi
        /// K2.5, Nemotron 3).
        case reasoningEffort
    }

    let efforts: [BedrockReasoningEffort]
    /// What the model does with no reasoning fields, when known, for display.
    let defaultEffort: BedrockReasoningEffort?
    let encoding: Encoding
    let maxTokensDefault: Int?
    let maxTokensMaximum: Int?

    /// The fields to send, or `nil` when the model doesn't offer this effort.
    func fields(for effort: BedrockReasoningEffort) -> BedrockReasoningFields? {
        guard efforts.contains(effort) else { return nil }
        let request: [String: BedrockJSONValue]
        switch encoding {
        case .schemaEnum(let path):
            request = Self.nested(path, .string(effort.wireValue))
        case .anthropicAdaptive:
            request = effort == .off
                ? ["thinking": .object(["type": .string("disabled")])]
                : [
                    "thinking": .object(["type": .string("adaptive")]),
                    "output_config": .object(["effort": .string(effort.rawValue)]),
                ]
        case .anthropicBudget(let budgets):
            if effort == .off {
                request = ["thinking": .object(["type": .string("disabled")])]
            } else {
                guard let budget = budgets[effort] else { return nil }
                request = ["thinking": .object(["type": .string("enabled"), "budget_tokens": .int(budget)])]
            }
        case .novaEffort:
            request = effort == .off
                ? ["reasoningConfig": .object(["type": .string("disabled")])]
                : ["reasoningConfig": .object(["type": .string("enabled"), "maxReasoningEffort": .string(effort.rawValue)])]
        case .reasoningEffort:
            request = ["reasoning_effort": .string(effort.wireValue)]
        }
        return BedrockReasoningFields(additionalModelRequestFields: request, maxTokens: outputLimit(for: effort))
    }

    /// Room for thinking, which can fill a small default output limit (Claude and gpt-oss
    /// default to 4,096 tokens) before the answer starts.
    static let thinkingOutputLimit = 32_000

    private func outputLimit(for effort: BedrockReasoningEffort) -> Int? {
        guard effort != .off, let maxTokensDefault, maxTokensDefault < Self.thinkingOutputLimit else { return nil }
        return min(Self.thinkingOutputLimit, maxTokensMaximum ?? Self.thinkingOutputLimit)
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
    /// - Parameters:
    ///   - modelID: a foundation model ID, or an inference profile ID such as `global.…`.
    ///   - schema: the catalog's `converse.additionalRequestFieldsSchema`, when known.
    ///     Models that list a `reasoning_config` or `output_config.effort` enum use it.
    ///     Others use the native fields in `nativeFamilies`, because their schema's
    ///     `reasoningConfig` is not what the request accepts.
    static func options(
        modelID: String,
        schema: String?,
        maxTokensDefault: Int? = nil,
        maxTokensMaximum: Int? = nil
    ) -> BedrockReasoningOptions? {
        let foundationID = foundationModelID(for: modelID)
        let tooSlow = tooSlowEfforts[foundationID] ?? []

        if let schemaEnum = SchemaEnum(schema: schema) {
            let efforts = BedrockReasoningEffort.allCases.filter {
                schemaEnum.values.contains($0.wireValue) && !tooSlow.contains($0)
            }
            guard !efforts.isEmpty else { return nil }
            return BedrockReasoningOptions(
                efforts: efforts,
                defaultEffort: schemaEnum.defaultValue.flatMap(BedrockReasoningEffort.init(wireValue:)),
                encoding: .schemaEnum(path: schemaEnum.path),
                maxTokensDefault: maxTokensDefault,
                maxTokensMaximum: maxTokensMaximum
            )
        }

        guard let family = nativeFamilies[foundationID] else { return nil }
        let efforts = family.efforts.filter { !tooSlow.contains($0) }
        guard !efforts.isEmpty else { return nil }
        return BedrockReasoningOptions(
            efforts: efforts,
            defaultEffort: family.defaultEffort,
            encoding: family.encoding,
            maxTokensDefault: maxTokensDefault,
            maxTokensMaximum: maxTokensMaximum
        )
    }

    /// `global.xai.grok-4.7` -> `xai.grok-4.7`; foundation model IDs are returned unchanged.
    static func foundationModelID(for modelID: String) -> String {
        let trimmed = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        let profilePrefixes: Set<String> = ["global", "us", "us-gov", "eu", "apac", "au", "ca", "jp"]
        guard let dot = trimmed.firstIndex(of: "."),
            profilePrefixes.contains(trimmed[..<dot].lowercased())
        else {
            return trimmed
        }
        return String(trimmed[trimmed.index(after: dot)...])
    }

    // MARK: - Native families

    struct Family {
        let efforts: [BedrockReasoningEffort]
        let defaultEffort: BedrockReasoningEffort?
        let encoding: BedrockReasoningOptions.Encoding
    }

    // Models whose catalog schema doesn't describe their reasoning field, with the native
    // field each family takes. Every effort here returned clean text in live tests on
    // 2026-11-18 (us-east-2, VoiceInk's Default prompt, 3 runs each); efforts that were
    // rejected, changed nothing, leaked reasoning into the text or were too slow are left out.
    static let nativeFamilies: [String: Family] = {
        // `thinking.type: adaptive` with `output_config.effort`. Claude 4.6 rejects xhigh.
        let adaptive: [String: [BedrockReasoningEffort]] = [
            "anthropic.claude-sonnet-4-6": [.off, .low, .medium, .high, .max],
            "anthropic.claude-opus-4-6-v1": [.off, .low, .medium, .high, .max],
            "anthropic.claude-opus-4-7": [.off, .low, .medium, .high, .xhigh, .max],
            "anthropic.claude-opus-4-8": [.off, .low, .medium, .high, .xhigh, .max],
            // Rejects `thinking.type: disabled`.
            "anthropic.claude-fable-5": [.low, .medium, .high, .xhigh, .max],
            "anthropic.claude-sonnet-5": [.off, .low, .medium, .high, .xhigh, .max],
            "anthropic.claude-opus-5": [.off, .low, .medium, .high, .xhigh, .max],
        ]
        // Reject adaptive thinking and take a thinking budget instead. They don't think by default.
        let budgets: [BedrockReasoningEffort: Int] = [.low: 1_024, .medium: 4_096, .high: 16_384]
        let budgetModels = ["anthropic.claude-haiku-4-5-20251001-v1:0", "anthropic.claude-opus-4-5-20251101-v1:0"]
        // gpt-oss reasons at OpenAI's default, medium, unless told otherwise.
        let gptOSS = ["openai.gpt-oss-120b-1:0", "openai.gpt-oss-20b-1:0", "openai.gpt-oss-safeguard-120b", "openai.gpt-oss-safeguard-20b"]
        // Only `high` turns reasoning on; `low` and `medium` answered exactly like Default.
        let highOnly = ["deepseek.v3.2", "deepseek.v3-v1:0", "zai.glm-5", "zai.glm-4.7", "zai.glm-4.7-flash", "nvidia.nemotron-super-3-120b"]

        var families: [String: Family] = [:]
        for (model, efforts) in adaptive {
            families[model] = Family(efforts: efforts, defaultEffort: nil, encoding: .anthropicAdaptive)
        }
        for model in budgetModels {
            families[model] = Family(efforts: [.low, .medium, .high], defaultEffort: .off, encoding: .anthropicBudget(budgets))
        }
        // `reasoningConfig.type: enabled` with `maxReasoningEffort`. Off by default.
        families["amazon.nova-2-lite-v1:0"] = Family(efforts: [.low, .medium], defaultEffort: .off, encoding: .novaEffort)
        for model in gptOSS {
            families[model] = Family(efforts: [.low, .medium, .high], defaultEffort: .medium, encoding: .reasoningEffort)
        }
        for model in highOnly {
            families[model] = Family(efforts: [.high], defaultEffort: .off, encoding: .reasoningEffort)
        }
        // No working control: Gemma 3 and Voxtral Small don't reason; MiniMax M2.5 reasons the
        // same with every field tried; Qwen3 `high` ran past 150 s and Qwen3 VL wrote its
        // reasoning into the text; Kimi K2.5 `high` took a 66 s median; Nemotron Nano 3 leaked
        // reasoning into the text at medium and added a preamble at high.
        return families
    }()

    // Efforts too slow for dictation in live measurements (median above ~45 s or max above
    // ~120 s, the rule in `ReasoningConfig.tooSlowLevels`), keyed by foundation model ID:
    // Claude Opus 5.5 max (65 s median), Opus 4.8 max (54 s), Sonnet 5 max (114 s) and
    // Nova 2 Lite high (114 s).
    static let tooSlowEfforts: [String: Set<BedrockReasoningEffort>] = [
        "anthropic.claude-opus-5-5": [.max],
        "anthropic.claude-opus-4-8": [.max],
        "anthropic.claude-sonnet-5": [.max],
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
            return (values.map { $0.lowercased() }, (node["default"] as? String)?.lowercased())
        }
    }
}
