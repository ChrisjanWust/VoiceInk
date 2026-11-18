import Foundation

/// A reasoning level the user can pick for an enhancement model. `nil` means "Default",
/// which keeps VoiceInk's built-in low-latency request for that model.
enum ReasoningLevel: String, CaseIterable, Codable, Sendable, Identifiable {
    /// No reasoning: `reasoning_effort: "none"` or the provider's equivalent.
    case off
    case minimal
    case low
    case medium
    case high
    case xhigh
    case max
    /// Turns thinking on where the provider only has an on/off switch (Ollama `think: true`).
    case enabled

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .off: return String(localized: "Off")
        case .minimal: return String(localized: "Minimal")
        case .low: return String(localized: "Low")
        case .medium: return String(localized: "Medium")
        case .high: return String(localized: "High")
        case .xhigh: return String(localized: "Extra High")
        case .max: return String(localized: "Max")
        case .enabled: return String(localized: "On")
        }
    }

    /// The OpenAI-style effort string (`reasoning_effort`, OpenRouter `reasoning.effort`).
    var effortValue: String? {
        switch self {
        case .off: return "none"
        case .minimal: return "minimal"
        case .low: return "low"
        case .medium: return "medium"
        case .high: return "high"
        case .xhigh: return "xhigh"
        case .max: return "max"
        case .enabled: return nil
        }
    }

    init?(effortValue: String) {
        guard let level = Self.allCases.first(where: { $0.effortValue == effortValue.lowercased() }) else {
            return nil
        }
        self = level
    }
}

/// Per provider + model reasoning choices, saved in UserDefaults next to the other AI settings.
enum EnhancementReasoningSettings {
    static let levelsKey = "EnhancementReasoningLevels"

    /// Keyed by the model a request actually uses, so a choice made for a retired
    /// model ID (still saved in a mode or the Auto-Learn reviewer) applies to its
    /// replacement.
    static func storageKey(provider: AIProvider, model: String) -> String {
        let model = model.trimmingCharacters(in: .whitespacesAndNewlines)
        return "\(provider.rawValue)|\(EnhancementModelMigration.resolvedModel(model, for: provider))"
    }

    /// The level the user saved for this model, or `nil` for Default.
    static func savedLevel(
        for provider: AIProvider,
        model: String,
        defaults: UserDefaults = .standard
    ) -> ReasoningLevel? {
        let model = model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !model.isEmpty,
            let levels = defaults.dictionary(forKey: levelsKey) as? [String: String],
            let rawValue = levels[storageKey(provider: provider, model: model)]
        else {
            return nil
        }
        return ReasoningLevel(rawValue: rawValue)
    }

    /// Saves a level for this model. Passing `nil` returns the model to Default.
    static func setLevel(
        _ level: ReasoningLevel?,
        for provider: AIProvider,
        model: String,
        defaults: UserDefaults = .standard
    ) {
        let model = model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !model.isEmpty else { return }

        var levels = defaults.dictionary(forKey: levelsKey) as? [String: String] ?? [:]
        let key = storageKey(provider: provider, model: model)
        guard levels[key] != level?.rawValue else { return }

        if let level {
            levels[key] = level.rawValue
        } else {
            levels.removeValue(forKey: key)
        }

        if levels.isEmpty {
            defaults.removeObject(forKey: levelsKey)
        } else {
            defaults.set(levels, forKey: levelsKey)
        }
        NotificationCenter.default.post(name: .AppSettingsDidChange, object: nil)
    }
}
