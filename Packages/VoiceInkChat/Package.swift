// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "VoiceInkChat",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "VoiceInkChat", targets: ["VoiceInkChat"]),
        // A developer tool for live checks. The app doesn't link it.
        .executable(name: "voiceink-chat-smoke", targets: ["voiceink-chat-smoke"]),
    ],
    targets: [
        .target(name: "VoiceInkChat"),
        .executableTarget(name: "voiceink-chat-smoke", dependencies: ["VoiceInkChat"]),
        .testTarget(name: "VoiceInkChatTests", dependencies: ["VoiceInkChat"]),
    ],
    swiftLanguageModes: [.v6]
)
