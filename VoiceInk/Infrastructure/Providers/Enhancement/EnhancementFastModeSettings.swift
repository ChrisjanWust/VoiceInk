import Foundation

/// Per provider + model Fast mode choices, saved in UserDefaults next to the reasoning levels.
/// A model that isn't listed has Fast mode off, which sends exactly VoiceInk's standard request.
enum EnhancementFastModeSettings {
    static let modelsKey = "EnhancementFastModeModels"

    /// Whether the user turned Fast mode on for this model. Keyed like reasoning levels, by the
    /// model a request actually uses, so a choice made for a retired model ID carries over.
    static func isEnabled(
        for provider: AIProvider,
        model: String,
        defaults: UserDefaults = .standard
    ) -> Bool {
        let model = model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !model.isEmpty, let models = defaults.stringArray(forKey: modelsKey) else {
            return false
        }
        return models.contains(EnhancementReasoningSettings.storageKey(provider: provider, model: model))
    }

    static func setEnabled(
        _ enabled: Bool,
        for provider: AIProvider,
        model: String,
        defaults: UserDefaults = .standard
    ) {
        let model = model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !model.isEmpty else { return }

        var models = defaults.stringArray(forKey: modelsKey) ?? []
        let key = EnhancementReasoningSettings.storageKey(provider: provider, model: model)
        guard models.contains(key) != enabled else { return }

        if enabled {
            models.append(key)
        } else {
            models.removeAll { $0 == key }
        }

        if models.isEmpty {
            defaults.removeObject(forKey: modelsKey)
        } else {
            defaults.set(models, forKey: modelsKey)
        }
        NotificationCenter.default.post(name: .AppSettingsDidChange, object: nil)
    }
}
