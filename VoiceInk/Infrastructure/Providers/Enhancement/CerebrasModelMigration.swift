import Foundation

/// Cerebras removes retired models from Shared Inference entirely: requests for them
/// return HTTP 404 (`model_archived` or `model_not_found`). Earlier VoiceInk releases
/// offered the IDs below for Cerebras, so they can still be saved as the selected model
/// or in a mode. Since custom model IDs are allowed for Cerebras, a saved ID is sent
/// as-is, so retired IDs are swapped for the current default instead.
enum CerebrasModelMigration {
    // See https://inference-docs.cerebras.ai/support/deprecation
    static let retiredModels: Set<String> = [
        "gemma-4-31b",
        "zai-glm-4.7",
        "zai-glm-4.6",
        "qwen-3-32b",
        "qwen-3-235b-a22b",
        "qwen-3-235b-a22b-instruct-2507",
        "qwen-3-235b-a22b-thinking-2507",
        "qwen-3-coder-480b",
        "llama-3.3-70b",
        "llama3.1-8b",
        "llama-4-scout-17b-16e-instruct",
        "llama-4-maverick-17b-128e-instruct",
    ]

    /// Returns the model to request: the current Cerebras default for a retired
    /// Cerebras model, otherwise `modelName` unchanged.
    static func resolvedModel(_ modelName: String, for provider: AIProvider) -> String {
        guard provider == .cerebras, isRetired(modelName) else { return modelName }
        return AIProvider.cerebras.defaultModel
    }

    /// Replaces a retired Cerebras model saved as the provider's selection.
    static func migrateSavedSelection(in defaults: UserDefaults = .standard) {
        let provider = AIProvider.cerebras
        let selectedModelKey = "\(provider.rawValue)SelectedModel"
        let customModelIDKey = "\(provider.rawValue)CustomModelID"

        if let savedModel = defaults.string(forKey: selectedModelKey), isRetired(savedModel) {
            defaults.set(provider.defaultModel, forKey: selectedModelKey)
        }
        if let savedModel = defaults.string(forKey: customModelIDKey), isRetired(savedModel) {
            defaults.removeObject(forKey: customModelIDKey)
        }
    }

    private static func isRetired(_ modelName: String) -> Bool {
        retiredModels.contains(modelName.trimmingCharacters(in: .whitespacesAndNewlines))
    }
}
