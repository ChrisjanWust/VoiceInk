import Foundation

/// Connects VoiceInk's reasoning levels to `BedrockReasoningFieldMap`, which holds the
/// per-model Bedrock mapping.
///
/// Default sends no reasoning fields, so each model keeps its Bedrock default. Typed model
/// IDs with no catalog entry still get their family's native fields.
enum BedrockReasoning {
    static func levels(for model: BedrockModelInfo?, modelName: String) -> [ReasoningLevel] {
        guard let efforts = options(for: model, modelName: modelName)?.efforts else { return [] }
        return ReasoningLevel.allCases.filter { $0.effortValue.map(efforts.contains) ?? false }
    }

    /// What Bedrock's default does for this model, for the picker's "Default (…)" label.
    static func defaultLevel(for model: BedrockModelInfo?, modelName: String) -> ReasoningLevel? {
        options(for: model, modelName: modelName)?.defaultEffort.flatMap(ReasoningLevel.init(effortValue:))
    }

    /// The fields to send for a level, or `nil` for Default and levels the model doesn't offer.
    static func fields(for level: ReasoningLevel?, model: BedrockModelInfo?, modelName: String) -> BedrockReasoningFields? {
        guard let effort = level?.effortValue else { return nil }
        return options(for: model, modelName: modelName)?.fields(for: effort)
    }

    private static func options(for model: BedrockModelInfo?, modelName: String) -> BedrockReasoningOptions? {
        BedrockReasoningFieldMap.options(
            modelID: model?.invocationID ?? modelName,
            schema: model?.additionalRequestFieldsSchema,
            maxTokensDefault: model?.maxTokensDefault,
            maxTokensMaximum: model?.maxTokensMaximum
        )
    }
}
