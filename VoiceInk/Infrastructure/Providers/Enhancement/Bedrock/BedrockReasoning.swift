import Foundation

/// A JSON value for Converse `additionalModelRequestFields`.
indirect enum BedrockJSONValue: Encodable, Sendable, Equatable {
    case string(String)
    case int(Int)
    case bool(Bool)
    case object([String: BedrockJSONValue])

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .int(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        }
    }
}

/// The reasoning fields a Converse request sends for a picked level.
struct BedrockReasoningFields: Sendable, Equatable {
    let additionalModelRequestFields: [String: BedrockJSONValue]
    /// `inferenceConfig.maxTokens`, when a thinking budget needs more room than the model's
    /// default output limit. `nil` keeps the model's default.
    let maxTokens: Int?
}

/// How a Bedrock model's reasoning is controlled, read from its
/// `converse.additionalRequestFieldsSchema`.
enum BedrockReasoningControl: Equatable {
    /// A string enum at `path`: `reasoning_config` (OpenAI, xAI, Moonshot K3, GLM 5.3) or
    /// `output_config.effort` (Claude 5.5, Fable 5.1 and later).
    case effort(path: [String], values: [String], defaultValue: String?)
    /// `reasoningConfig.enabled`, optionally with named `budgetTokens` presets (older Claude
    /// models list `low`/`medium`/`high`). Off sends `enabled: false`.
    case toggle(defaultEnabled: Bool, budgetPresets: [String: Int])
    /// Named `reasoningConfig.budgetTokens` presets with no on/off switch (Nova 2).
    case budget(presets: [String: Int])

    init?(schema: String?) {
        guard let schema, let data = schema.data(using: .utf8),
            let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else {
            return nil
        }

        if let effort = Self.stringEnum(root["reasoning_config"]) {
            self = .effort(path: ["reasoning_config"], values: effort.values, defaultValue: effort.defaultValue)
        } else if let outputConfig = root["output_config"] as? [String: Any],
            let effort = Self.stringEnum(outputConfig["effort"])
        {
            self = .effort(path: ["output_config", "effort"], values: effort.values, defaultValue: effort.defaultValue)
        } else if let reasoningConfig = root["reasoningConfig"] as? [String: Any] {
            let presets = Self.budgetPresets(reasoningConfig["budgetTokens"])
            if let enabled = reasoningConfig["enabled"] as? [String: Any], enabled["type"] as? String == "boolean" {
                self = .toggle(defaultEnabled: enabled["default"] as? Bool ?? false, budgetPresets: presets)
            } else if !presets.isEmpty {
                self = .budget(presets: presets)
            } else {
                return nil
            }
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

    /// `{"type": "enum", "enum": {"low": 1024, ...}}`. An integer `budgetTokens` has no presets.
    private static func budgetPresets(_ node: Any?) -> [String: Int] {
        guard let node = node as? [String: Any], node["type"] as? String == "enum",
            let presets = node["enum"] as? [String: Any]
        else {
            return [:]
        }
        return presets.compactMapValues { ($0 as? NSNumber)?.intValue }
    }
}

/// Maps VoiceInk's reasoning levels onto each Bedrock model's own reasoning fields.
///
/// Default sends no reasoning fields, so each model keeps its Bedrock default. Only levels
/// the model's schema lists are offered, and custom model IDs (with no catalog entry) stay on
/// Default.
enum BedrockReasoning {
    private static let budgetLevels: [ReasoningLevel] = [.low, .medium, .high]

    static func control(for model: BedrockModelInfo?) -> BedrockReasoningControl? {
        BedrockReasoningControl(schema: model?.additionalRequestFieldsSchema)
    }

    static func levels(for model: BedrockModelInfo?) -> [ReasoningLevel] {
        switch control(for: model) {
        case .effort(_, let values, _):
            let offered = Set(values.compactMap(ReasoningLevel.init(effortValue:)))
            return ReasoningLevel.allCases.filter(offered.contains)
        case .toggle(_, let presets):
            let budgets = budgetLevels.filter { presets[$0.rawValue] != nil }
            return [.off] + (budgets.isEmpty ? [.enabled] : budgets)
        case .budget(let presets):
            return budgetLevels.filter { presets[$0.rawValue] != nil }
        case nil:
            return []
        }
    }

    /// What Bedrock's default does for this model, for the picker's "Default (…)" label.
    static func defaultLevel(for model: BedrockModelInfo?) -> ReasoningLevel? {
        switch control(for: model) {
        case .effort(_, _, let defaultValue):
            return defaultValue.flatMap(ReasoningLevel.init(effortValue:))
        case .toggle, .budget, nil:
            // The schema's `enabled` default isn't reliable: Claude Sonnet 5 and MiniMax M2.5
            // list `false` but still returned reasoning at Default in live tests.
            return nil
        }
    }

    /// The fields to send for a level, or `nil` for Default and levels the model doesn't list.
    static func fields(for level: ReasoningLevel?, model: BedrockModelInfo?) -> BedrockReasoningFields? {
        guard let level, levels(for: model).contains(level) else { return nil }

        switch control(for: model) {
        case .effort(let path, _, _):
            guard let value = level.effortValue else { return nil }
            return BedrockReasoningFields(additionalModelRequestFields: nested(path, .string(value)), maxTokens: nil)
        case .toggle(_, let presets):
            var config: [String: BedrockJSONValue] = ["enabled": .bool(level != .off)]
            var maxTokens: Int?
            if let budget = presets[level.rawValue] {
                config["budgetTokens"] = .int(budget)
                maxTokens = outputLimit(forBudget: budget, model: model)
            }
            return BedrockReasoningFields(additionalModelRequestFields: ["reasoningConfig": .object(config)], maxTokens: maxTokens)
        case .budget(let presets):
            guard let budget = presets[level.rawValue] else { return nil }
            return BedrockReasoningFields(
                additionalModelRequestFields: ["reasoningConfig": .object(["budgetTokens": .int(budget)])],
                maxTokens: outputLimit(forBudget: budget, model: model)
            )
        case nil:
            return nil
        }
    }

    /// Thinking tokens count toward the output limit, so a budget above the model's default
    /// limit needs a larger one, with room left for the answer.
    private static func outputLimit(forBudget budget: Int, model: BedrockModelInfo?) -> Int? {
        guard let model, let defaultLimit = model.maxTokensDefault, budget >= defaultLimit else { return nil }
        let wanted = budget + defaultLimit
        return model.maxTokensMaximum.map { min(wanted, $0) } ?? wanted
    }

    private static func nested(_ path: [String], _ value: BedrockJSONValue) -> [String: BedrockJSONValue] {
        guard let first = path.first else { return [:] }
        if path.count == 1 { return [first: value] }
        return [first: .object(nested(Array(path.dropFirst()), value))]
    }
}
