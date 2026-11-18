import SwiftUI

/// Picks the reasoning level for one enhancement model. Shown below the model picker,
/// and hidden for models that only run on Default.
struct EnhancementReasoningPicker: View {
    let provider: AIProvider
    let modelName: String
    var modelInfo = ReasoningModelInfo()
    @Binding var level: ReasoningLevel?

    /// The model requests use, after retired-model migration.
    private var requestModel: String {
        EnhancementModelMigration.resolvedModel(
            modelName.trimmingCharacters(in: .whitespacesAndNewlines),
            for: provider
        )
    }

    private var levels: [ReasoningLevel] {
        ReasoningConfig.supportedLevels(
            for: provider,
            modelName: requestModel,
            modelInfo: modelInfo
        )
    }

    /// A saved level this model no longer accepts is sent as Default, so show Default.
    private var selection: Binding<ReasoningLevel?> {
        Binding(
            get: { levels.contains(where: { $0 == level }) ? level : nil },
            set: { level = $0 }
        )
    }

    private var defaultLabel: String {
        guard
            let defaultLevel = ReasoningConfig.defaultLevel(
                for: provider,
                modelName: requestModel,
                modelInfo: modelInfo
            )
        else {
            return String(localized: "Default")
        }
        return String(format: String(localized: "Default (%@)"), defaultLevel.displayName)
    }

    /// How long a request at the selected level may take, for levels slower than Default.
    private var slowLevelCaption: String? {
        guard let level = selection.wrappedValue, ReasoningConfig.timeoutFloor(for: level) != nil else {
            return nil
        }
        let timeout = ReasoningConfig.requestTimeout(
            EnhancementRequestSettings.timeout,
            retryOnTimeout: EnhancementRequestSettings.retryOnTimeout,
            level: level
        )
        // A timeout retry makes up to three attempts.
        let seconds = Int(timeout.seconds) * (timeout.retryOnTimeout ? 3 : 1)
        return String(
            format: String(localized: "Higher reasoning is slower. Enhancement may take up to %d seconds."),
            seconds
        )
    }

    var body: some View {
        if !modelName.isEmpty, !levels.isEmpty {
            Picker(selection: selection) {
                Text(defaultLabel).tag(ReasoningLevel?.none)
                ForEach(levels) { level in
                    Text(level.displayName).tag(Optional(level))
                }
            } label: {
                HStack(spacing: 6) {
                    Text("Reasoning")
                    InfoTip(
                        "How much the model thinks before it answers. Higher levels can help with complex prompts but are slower and cost more. Default keeps VoiceInk's low-latency setting. This applies to this model wherever it's used."
                    )
                }
            }

            if let slowLevelCaption {
                Text(slowLevelCaption)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

/// A reasoning picker that saves the choice as soon as it changes.
struct SavedEnhancementReasoningPicker: View {
    let provider: AIProvider
    let modelName: String
    var modelInfo = ReasoningModelInfo()

    // UserDefaults dictionaries aren't observable, so redraw after each save.
    @State private var saveCount = 0

    var body: some View {
        let _ = saveCount
        EnhancementReasoningPicker(
            provider: provider,
            modelName: modelName,
            modelInfo: modelInfo,
            level: Binding(
                get: { EnhancementReasoningSettings.savedLevel(for: provider, model: modelName) },
                set: { newLevel in
                    EnhancementReasoningSettings.setLevel(newLevel, for: provider, model: modelName)
                    saveCount += 1
                }
            )
        )
    }
}
