import Foundation

/// The model ID a request actually uses, after each provider's retired-model mapping.
/// Each migration only changes its own provider's IDs.
enum EnhancementModelMigration {
    static func resolvedModel(_ modelName: String, for provider: AIProvider) -> String {
        GeminiModelMigration.resolvedModel(
            CerebrasModelMigration.resolvedModel(modelName, for: provider),
            for: provider
        )
    }
}
