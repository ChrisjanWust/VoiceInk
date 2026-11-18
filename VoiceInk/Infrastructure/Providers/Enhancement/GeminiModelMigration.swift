import Foundation

/// VoiceInk only offers Gemini 3 and later. On Gemini's own API, only Gemini 3 and later accept
/// a system instruction on `/v1/interactions`; older models return HTTP 400, and enhancement
/// always sends one. Routers such as OpenRouter still list older Gemini models too. Earlier
/// VoiceInk releases offered older Gemini models, so they can still be saved as the selected
/// model, in a mode or as the Auto-Learn reviewer. A saved ID is sent as-is, so these are
/// swapped for a current Gemini model instead.
enum GeminiModelMigration {
    static let replacementModel = "gemini-3.8-flash"
    /// OpenRouter's ID for `replacementModel` (on its live list on 2026-11-18).
    static let openRouterReplacementModel = "google/gemini-3.8-flash"
    private static let minimumMajorVersion = 3

    /// Returns the model to request: the provider's replacement for a Gemini model older
    /// than version 3, otherwise `modelName` unchanged.
    static func resolvedModel(_ modelName: String, for provider: AIProvider) -> String {
        guard isUnsupported(modelName) else { return modelName }
        switch provider {
        case .gemini:
            return replacementModel
        case .openRouter:
            return openRouterReplacementModel
        default:
            return modelName
        }
    }

    /// Replaces a Gemini model older than version 3 saved as the Gemini or OpenRouter
    /// selection, or as the Auto-Learn reviewer model. Modes are repaired in
    /// `AIEnhancementService.repairModePromptSelections`.
    static func migrateSavedSelection(in defaults: UserDefaults = .standard) {
        for provider in [AIProvider.gemini, .openRouter] {
            let selectedModelKey = "\(provider.rawValue)SelectedModel"
            let customModelIDKey = "\(provider.rawValue)CustomModelID"

            if let savedModel = defaults.string(forKey: selectedModelKey), isUnsupported(savedModel) {
                defaults.set(resolvedModel(savedModel, for: provider), forKey: selectedModelKey)
            }
            if let savedModel = defaults.string(forKey: customModelIDKey), isUnsupported(savedModel) {
                defaults.removeObject(forKey: customModelIDKey)
            }
        }

        if let providerName = defaults.string(forKey: AutoLearnSettings.reviewerProviderKey),
            let provider = AIProvider(rawValue: providerName),
            let savedModel = defaults.string(forKey: AutoLearnSettings.reviewerModelKey)
        {
            let currentModel = resolvedModel(savedModel, for: provider)
            if currentModel != savedModel {
                defaults.set(currentModel, forKey: AutoLearnSettings.reviewerModelKey)
            }
        }
    }

    /// Matches Gemini IDs whose major version is below 3: `gemini-<major>…`, also with the
    /// owner prefix routers use (`google/gemini-<major>…`). Shared by every provider that
    /// lists Gemini models.
    static func isUnsupported(_ modelName: String) -> Bool {
        var id = modelName.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if let slash = id.lastIndex(of: "/") {
            id = String(id[id.index(after: slash)...])
        }
        guard let range = id.range(of: #"^gemini-\d+"#, options: .regularExpression),
            let major = Int(id[range].dropFirst("gemini-".count))
        else {
            return false
        }
        return major < minimumMajorVersion
    }
}
