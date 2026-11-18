import Foundation
import LLMkit
import SwiftUI

class OllamaService: ObservableObject {
    static let defaultBaseURL = "http://localhost:11434"

    // MARK: - Published Properties
    @Published var baseURL: String {
        didSet {
            UserDefaults.standard.set(baseURL, forKey: "ollamaBaseURL")
        }
    }

    @Published var selectedModel: String {
        didSet {
            UserDefaults.standard.set(selectedModel, forKey: "ollamaSelectedModel")
        }
    }

    @Published var availableModels: [OllamaModel] = []
    /// Whether each installed model lists the "thinking" capability. Models are
    /// missing when the server doesn't report capabilities (older Ollama versions).
    @Published var thinkingSupport: [String: Bool] = [:]
    /// Named thinking levels per model, from `/api/show`'s `thinking.values` (for example
    /// `["low", "medium", "high"]` for gpt-oss). Empty for an on/off model, missing when unknown.
    @Published var thinkingLevels: [String: [String]] = [:]
    @Published var isConnected: Bool = false
    @Published var isLoadingModels: Bool = false

    private let defaultTemperature: Double = 0.3

    init() {
        self.baseURL = UserDefaults.standard.string(forKey: "ollamaBaseURL") ?? Self.defaultBaseURL
        self.selectedModel = UserDefaults.standard.string(forKey: "ollamaSelectedModel") ?? "llama2"
    }

    private var baseURLValue: URL? {
        URL(string: baseURL)
    }

    @MainActor
    func checkConnection() async {
        guard let url = baseURLValue else {
            isConnected = false
            return
        }
        isConnected = await OllamaClient.checkConnection(baseURL: url)
    }

    @MainActor
    func refreshModels() async {
        _ = await refreshConnectionAndModels()
    }

    @MainActor
    func refreshConnectionAndModels() async -> Result<[OllamaModel], Error> {
        isLoadingModels = true
        defer { isLoadingModels = false }

        guard let url = baseURLValue else {
            isConnected = false
            availableModels = []
            return .failure(LocalAIError.invalidURL)
        }

        do {
            let models = try await OllamaClient.fetchModels(baseURL: url)
            isConnected = true
            availableModels = models
            thinkingSupport = await fetchThinkingSupport(baseURL: url)
            thinkingLevels = await fetchThinkingLevels(
                baseURL: url,
                models: thinkingSupport.filter(\.value).map(\.key)
            )

            if !models.contains(where: { $0.name == selectedModel }) && !models.isEmpty {
                selectedModel = models[0].name
            }

            return .success(models)
        } catch {
            isConnected = false
            availableModels = []
            thinkingSupport = [:]
            thinkingLevels = [:]
            return .failure(error)
        }
    }

    // LLMkit's OllamaModel doesn't decode `capabilities`, so read them from /api/tags here.
    private func fetchThinkingSupport(baseURL: URL) async -> [String: Bool] {
        var request = URLRequest(url: baseURL.appendingPathComponent("api/tags"))
        request.timeoutInterval = 10

        guard let (data, response) = try? await URLSession.shared.data(for: request),
            let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode),
            let decoded = try? JSONDecoder().decode(OllamaCapabilitiesResponse.self, from: data)
        else {
            return [:]
        }

        var support: [String: Bool] = [:]
        for model in decoded.models {
            if let capabilities = model.capabilities {
                support[model.name] = capabilities.contains("thinking")
            }
        }
        return support
    }

    // `/api/tags` doesn't include `thinking.values`, so ask `/api/show` for each thinking model.
    private func fetchThinkingLevels(baseURL: URL, models: [String]) async -> [String: [String]] {
        await withTaskGroup(of: (String, [String]?).self) { group in
            for model in models {
                group.addTask {
                    var request = URLRequest(url: baseURL.appendingPathComponent("api/show"))
                    request.httpMethod = "POST"
                    request.timeoutInterval = 10
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    request.httpBody = try? JSONSerialization.data(withJSONObject: ["model": model])

                    guard let (data, response) = try? await URLSession.shared.data(for: request),
                        let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode),
                        let decoded = try? JSONDecoder().decode(OllamaShowResponse.self, from: data)
                    else {
                        return (model, nil)
                    }
                    // Booleans in `values` are an on/off switch; strings are named levels.
                    return (model, decoded.thinking?.values.compactMap(\.level) ?? [])
                }
            }

            var levels: [String: [String]] = [:]
            for await (model, modelLevels) in group {
                if let modelLevels {
                    levels[model] = modelLevels
                }
            }
            return levels
        }
    }

    func enhance(
        _ text: String,
        withSystemPrompt systemPrompt: String? = nil,
        model: String? = nil,
        think: ReasoningConfig.OllamaThink = .enabled(false),
        timeout: TimeInterval = 30,
        retryOnTimeout: Bool = false
    ) async throws -> String {
        guard let systemPrompt = systemPrompt else {
            throw LocalAIError.invalidRequest
        }

        guard let url = baseURLValue else {
            throw LocalAIError.invalidURL
        }

        let trimmedModel = model?.trimmingCharacters(in: .whitespacesAndNewlines)
        let requestModel = (trimmedModel?.isEmpty == false ? trimmedModel : nil) ?? selectedModel

        do {
            switch think {
            case .enabled(let enabled):
                return try await OllamaClient.generate(
                    baseURL: url,
                    model: requestModel,
                    prompt: text,
                    systemPrompt: systemPrompt,
                    options: OllamaGenerationOptions(temperature: defaultTemperature),
                    think: enabled,
                    timeout: timeout,
                    retryOnTimeout: retryOnTimeout
                )
            case .level(let level):
                return try await OllamaThinkLevelClient.generate(
                    baseURL: url,
                    model: requestModel,
                    prompt: text,
                    systemPrompt: systemPrompt,
                    temperature: defaultTemperature,
                    thinkLevel: level,
                    timeout: timeout,
                    retryOnTimeout: retryOnTimeout
                )
            }
        } catch let error as LLMKitError {
            throw mapLLMKitError(error)
        }
    }

    private func mapLLMKitError(_ error: LLMKitError) -> LocalAIError {
        switch error {
        case .invalidURL:
            return .invalidURL
        case .httpError(let statusCode, _):
            if statusCode == 404 { return .modelNotFound }
            if statusCode == 408 { return .timeout }
            if statusCode == 500 { return .serverError }
            return .invalidResponse
        case .networkError:
            return .serviceUnavailable
        case .noResultReturned, .decodingError:
            return .invalidResponse
        case .encodingError:
            return .invalidRequest
        case .missingAPIKey:
            return .invalidResponse
        case .timeout:
            return .timeout
        case .unsupportedModel:
            return .modelNotFound
        }
    }
}

private struct OllamaCapabilitiesResponse: Decodable {
    struct Model: Decodable {
        let name: String
        let capabilities: [String]?
    }

    let models: [Model]
}

private struct OllamaShowResponse: Decodable {
    struct Thinking: Decodable {
        let values: [Value]
    }

    /// A `thinking.values` entry: `true`/`false`, or a level name such as `"high"`.
    enum Value: Decodable {
        case flag(Bool)
        case level(String)

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let flag = try? container.decode(Bool.self) {
                self = .flag(flag)
            } else {
                self = .level(try container.decode(String.self))
            }
        }

        var level: String? {
            if case .level(let name) = self { return name }
            return nil
        }
    }

    let thinking: Thinking?
}

// MARK: - Error Types
enum LocalAIError: Error, LocalizedError {
    case invalidURL
    case serviceUnavailable
    case invalidResponse
    case modelNotFound
    case serverError
    case invalidRequest
    case timeout

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return String(localized: "Invalid Ollama server URL")
        case .serviceUnavailable:
            return String(localized: "Ollama service is not available")
        case .invalidResponse:
            return String(localized: "Invalid response from Ollama server")
        case .modelNotFound:
            return String(localized: "Selected model not found")
        case .serverError:
            return String(localized: "Ollama server error")
        case .invalidRequest:
            return String(localized: "System prompt is required")
        case .timeout:
            return String(localized: "Ollama request timed out")
        }
    }
}
