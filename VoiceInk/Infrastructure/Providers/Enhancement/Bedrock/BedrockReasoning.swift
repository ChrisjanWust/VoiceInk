import Foundation

/// A JSON value for Converse `additionalModelRequestFields`.
indirect enum BedrockJSONValue: Encodable, Sendable, Equatable {
    case string(String)
    case object([String: BedrockJSONValue])

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        }
    }
}

/// The reasoning fields a Converse request sends for a picked level.
struct BedrockReasoningFields: Sendable, Equatable {
    let additionalModelRequestFields: [String: BedrockJSONValue]
    /// `inferenceConfig.maxTokens`, when thinking needs more room than the model's default
    /// output limit. `nil` keeps the model's default.
    let maxTokens: Int?
}

/// How a Bedrock model takes a reasoning effort, read from its
/// `converse.additionalRequestFieldsSchema`: a string enum at `path`, either
/// `reasoning_config` (OpenAI, xAI, Moonshot K3.1) or `output_config.effort` (Claude 5.5,
/// Fable 5.1 and later).
///
/// Schemas that only list `reasoningConfig` aren't mapped. Live tests on 2026-11-18 showed
/// it isn't the request format: older Claude models and Nova 2 reject it, and Qwen, DeepSeek,
/// Gemma and MiniMax accept it without any change in behavior.
struct BedrockReasoningControl: Equatable {
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

/// Maps VoiceInk's reasoning levels onto each Bedrock model's own reasoning field.
///
/// Default sends no reasoning fields, so each model keeps its Bedrock default. Only levels
/// the model's schema lists are offered, and custom model IDs (with no catalog entry) stay on
/// Default.
enum BedrockReasoning {
    static func control(for model: BedrockModelInfo?) -> BedrockReasoningControl? {
        BedrockReasoningControl(schema: model?.additionalRequestFieldsSchema)
    }

    static func levels(for model: BedrockModelInfo?) -> [ReasoningLevel] {
        guard let model, let control = control(for: model) else { return [] }
        let offered = Set(control.values.compactMap(ReasoningLevel.init(effortValue:)))
            .subtracting(tooSlowLevels[model.modelID] ?? [])
        return ReasoningLevel.allCases.filter(offered.contains)
    }

    // Levels too slow for dictation in live measurements on 2026-11-18 (median above ~45 s or
    // max above ~120 s, the rule in `ReasoningConfig.tooSlowLevels`), keyed by foundation
    // model ID so every inference profile for the model is covered.
    static let tooSlowLevels: [String: Set<ReasoningLevel>] = [
        "anthropic.claude-opus-5-5": [.max]
    ]

    /// What Bedrock's default does for this model, for the picker's "Default (…)" label.
    static func defaultLevel(for model: BedrockModelInfo?) -> ReasoningLevel? {
        control(for: model)?.defaultValue.flatMap(ReasoningLevel.init(effortValue:))
    }

    /// The fields to send for a level, or `nil` for Default and levels the model doesn't list.
    static func fields(for level: ReasoningLevel?, model: BedrockModelInfo?) -> BedrockReasoningFields? {
        guard let level, let value = level.effortValue, let control = control(for: model),
            levels(for: model).contains(level)
        else {
            return nil
        }
        return BedrockReasoningFields(
            additionalModelRequestFields: nested(control.path, .string(value)),
            maxTokens: outputLimit(for: level, model: model)
        )
    }

    /// Room for extra-high and max thinking, which can fill a small default output limit
    /// (Claude's 4,096 tokens) before the answer starts. Models with no listed default have no
    /// small limit to raise.
    static let thinkingOutputLimit = 32_000

    private static func outputLimit(for level: ReasoningLevel, model: BedrockModelInfo?) -> Int? {
        guard level == .xhigh || level == .max,
            let defaultLimit = model?.maxTokensDefault, defaultLimit < thinkingOutputLimit
        else {
            return nil
        }
        return min(thinkingOutputLimit, model?.maxTokensMaximum ?? thinkingOutputLimit)
    }

    private static func nested(_ path: [String], _ value: BedrockJSONValue) -> [String: BedrockJSONValue] {
        guard let first = path.first else { return [:] }
        if path.count == 1 { return [first: value] }
        return [first: .object(nested(Array(path.dropFirst()), value))]
    }
}
