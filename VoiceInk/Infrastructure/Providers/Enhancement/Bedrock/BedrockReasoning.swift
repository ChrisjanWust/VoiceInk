import Foundation

/// Connects VoiceInk's reasoning levels to `BedrockReasoningFieldMap`, which holds the
/// per-model Bedrock mapping.
///
/// Default sends no reasoning fields, so each model keeps its Bedrock default. Models in the
/// live catalog use their schema or their family's native fields; typed IDs get native fields
/// when their foundation model is in the family table.
enum BedrockReasoning {
    static func options(for model: BedrockModelInfo?, modelName: String) -> BedrockReasoningOptions? {
        BedrockReasoningFieldMap.options(
            modelID: model?.invocationID ?? modelName,
            schema: model?.additionalRequestFieldsSchema,
            maxTokensDefault: model?.maxTokensDefault,
            maxTokensMaximum: model?.maxTokensMaximum
        )
    }

    static func levels(for model: BedrockModelInfo?, modelName: String) -> [ReasoningLevel] {
        (options(for: model, modelName: modelName)?.efforts ?? []).compactMap(ReasoningLevel.init(bedrockEffort:))
    }

    /// What Bedrock's default does for this model, for the picker's "Default (…)" label.
    static func defaultLevel(for model: BedrockModelInfo?, modelName: String) -> ReasoningLevel? {
        options(for: model, modelName: modelName)?.defaultEffort.flatMap(ReasoningLevel.init(bedrockEffort:))
    }

    /// The fields to send for a level, or `nil` for Default and levels the model doesn't offer.
    static func fields(for level: ReasoningLevel?, model: BedrockModelInfo?, modelName: String) -> BedrockReasoningFields? {
        guard let effort = level?.bedrockEffort else { return nil }
        return options(for: model, modelName: modelName)?.fields(for: effort)
    }
}

extension ReasoningLevel {
    init?(bedrockEffort: BedrockReasoningEffort) {
        switch bedrockEffort {
        case .off: self = .off
        case .minimal: self = .minimal
        case .low: self = .low
        case .medium: self = .medium
        case .high: self = .high
        case .xhigh: self = .xhigh
        case .max: self = .max
        }
    }

    var bedrockEffort: BedrockReasoningEffort? {
        switch self {
        case .off: return .off
        case .minimal: return .minimal
        case .low: return .low
        case .medium: return .medium
        case .high: return .high
        case .xhigh: return .xhigh
        case .max: return .max
        case .enabled: return nil
        }
    }
}
