import SwiftUI

/// Turns Fast mode on or off for one enhancement model. Shown next to the Reasoning picker,
/// and hidden for models without a fast option.
struct EnhancementFastModeToggle: View {
    let provider: AIProvider
    let modelName: String
    @Binding var isOn: Bool

    /// The model requests use, after retired-model migration.
    private var requestModel: String {
        EnhancementModelMigration.resolvedModel(
            modelName.trimmingCharacters(in: .whitespacesAndNewlines),
            for: provider
        )
    }

    private var isSupported: Bool {
        FastModeConfig.isSupported(for: provider, modelName: requestModel)
    }

    var body: some View {
        if !modelName.isEmpty, isSupported {
            Toggle(isOn: $isOn) {
                HStack(spacing: 6) {
                    Text("Fast mode")
                    InfoTip(
                        "Sends this model's requests through the provider's faster processing tier, which costs more. If the provider is at capacity, it may answer at standard speed and price. This applies to this model wherever it's used."
                    )
                }
            }

            if isOn {
                Text(FastModeConfig.costCaption(for: provider))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

/// A Fast mode toggle that saves the choice as soon as it changes.
struct SavedEnhancementFastModeToggle: View {
    let provider: AIProvider
    let modelName: String

    // UserDefaults arrays aren't observable, so redraw after each save.
    @State private var saveCount = 0

    var body: some View {
        let _ = saveCount
        EnhancementFastModeToggle(
            provider: provider,
            modelName: modelName,
            isOn: Binding(
                get: { EnhancementFastModeSettings.isEnabled(for: provider, model: modelName) },
                set: { enabled in
                    EnhancementFastModeSettings.setEnabled(enabled, for: provider, model: modelName)
                    saveCount += 1
                }
            )
        )
    }
}
