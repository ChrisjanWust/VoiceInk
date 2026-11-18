import Foundation

/// Connects VoiceInk's reasoning levels to `BedrockReasoningFieldMap`, which holds the
/// per-model Bedrock mapping.
///
/// Default sends no reasoning fields, so each model keeps its Bedrock default. Models with no
/// catalog entry, such as typed IDs before the live list loads, stay on Default.
enum BedrockReasoning {
    static func levels(for model: BedrockModelInfo?) -> [ReasoningLevel] {
        guard let efforts = options(for: model)?.efforts else { return [] }
        return ReasoningLevel.allCases.filter { $0.effortValue.map(efforts.contains) ?? false }
    }

    /// What Bedrock's default does for this model, for the picker's "Default (…)" label.
    static func defaultLevel(for model: BedrockModelInfo?) -> ReasoningLevel? {
        options(for: model)?.defaultEffort.flatMap(ReasoningLevel.init(effortValue:))
    }

    /// The fields to send for a level, or `nil` for Default and levels the model doesn't offer.
    static func fields(for level: ReasoningLevel?, model: BedrockModelInfo?) -> BedrockReasoningFields? {
        guard let effort = level?.effortValue else { return nil }
        return options(for: model)?.fields(for: effort)
    }

    private static func options(for model: BedrockModelInfo?) -> BedrockReasoningOptions? {
        guard let model else { return nil }
        return BedrockReasoningFieldMap.options(
            foundationModelID: model.modelID,
            schema: model.additionalRequestFieldsSchema,
            maxTokensDefault: model.maxTokensDefault,
            maxTokensMaximum: model.maxTokensMaximum
        )
    }
}
