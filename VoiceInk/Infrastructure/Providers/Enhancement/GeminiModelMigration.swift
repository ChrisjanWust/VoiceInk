import Foundation

/// Only Gemini 3 and later accept a system instruction on `/v1/interactions`; older models
/// return HTTP 400, and enhancement always sends one. Earlier VoiceInk releases offered older
/// Gemini models, so they can still be saved as the selected model or in a mode. Since custom
/// model IDs are allowed for Gemini, a saved ID is sent as-is, so these are swapped for the
/// current default instead.
enum GeminiModelMigration {
    static let replacementModel = "gemini-3.8-flash"
    private static let minimumMajorVersion = 3

    /// Returns the model to request: the replacement for a Gemini model older than version 3,
    /// otherwise `modelName` unchanged.
    static func resolvedModel(_ modelName: String, for provider: AIProvider) -> String {
        guard provider == .gemini, isUnsupported(modelName) else { return modelName }
        return replacementModel
    }

    /// Replaces a Gemini model older than version 3 saved as the provider's selection.
    static func migrateSavedSelection(in defaults: UserDefaults = .standard) {
        let provider = AIProvider.gemini
        let selectedModelKey = "\(provider.rawValue)SelectedModel"
        let customModelIDKey = "\(provider.rawValue)CustomModelID"

        if let savedModel = defaults.string(forKey: selectedModelKey), isUnsupported(savedModel) {
            defaults.set(replacementModel, forKey: selectedModelKey)
        }
        if let savedModel = defaults.string(forKey: customModelIDKey), isUnsupported(savedModel) {
            defaults.removeObject(forKey: customModelIDKey)
        }
    }

    /// Matches `gemini-<major>…` IDs whose major version is below 3.
    static func isUnsupported(_ modelName: String) -> Bool {
        let id = modelName.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard let range = id.range(of: #"^gemini-\d+"#, options: .regularExpression),
            let major = Int(id[range].dropFirst("gemini-".count))
        else {
            return false
        }
        return major < minimumMajorVersion
    }
}
