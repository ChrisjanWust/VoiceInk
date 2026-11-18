import SwiftUI

/// AWS Region setting for Amazon Bedrock, shown above the API key field.
///
/// Valid edits are saved right away. The API key itself stays in the Keychain, like every other provider's key.
struct BedrockRegionField: View {
    var isDisabled = false

    @State private var draft = BedrockRegion.current

    private var isDraftValid: Bool {
        BedrockRegion.isValid(draft)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("AWS Region")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.primary)

                Spacer(minLength: 8)

                TextField("us-east-1", text: $draft)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 12, design: .monospaced))
                    .frame(width: 140)
                    .disabled(isDisabled)

                Menu {
                    ForEach(BedrockRegion.commonRegions, id: \.self) { region in
                        Button(region) {
                            draft = region
                        }
                    }
                } label: {
                    Image(systemName: "chevron.up.chevron.down")
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .fixedSize()
                .disabled(isDisabled)
                .help("Choose an AWS Region")
            }

            if isDraftValid {
                Text(
                    "Requests go to \(BedrockRegion.runtimeEndpoint(for: draft)). Short-term API keys only work in the Region where they were created."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            } else {
                Text("Enter an AWS Region code, such as us-east-1 or eu-central-1.")
                    .font(.caption)
                    .foregroundStyle(AppTheme.Status.error)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .onAppear {
            draft = BedrockRegion.current
        }
        .onChange(of: draft) { _, newValue in
            guard BedrockRegion.isValid(newValue) else { return }
            let region = BedrockRegion.normalized(newValue)
            guard region != BedrockRegion.current else { return }
            BedrockRegion.current = region
            NotificationCenter.default.post(name: .AppSettingsDidChange, object: nil)
        }
    }
}

/// Model used to verify a Bedrock key.
///
/// Bedrock keys inherit IAM permissions that can be limited to specific models, so verification runs against
/// the model the person chooses here. The caller saves the choice as Bedrock's selected model once the key verifies.
struct BedrockVerificationModelPicker: View {
    @ObservedObject var aiService: AIService
    @Binding var model: String
    var isDisabled = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            EnhancementModelPicker(
                title: "Model",
                provider: .bedrock,
                models: aiService.availableModels(for: .bedrock),
                savedCustomModelID: aiService.customModelID(for: .bedrock),
                draftModel: $model
            )
            .disabled(isDisabled)

            Text("Verification sends a short request to this model, so choose one your API key is allowed to use.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
