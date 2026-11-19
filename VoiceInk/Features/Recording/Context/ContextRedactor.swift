import Foundation

/// Replaces SSNs and API keys, tokens, passwords and private keys in captured context
/// (window OCR, clipboard, selected text) before any of it can reach an AI provider.
enum ContextRedactor {
    static let ssnPlaceholder = "[REDACTED SSN]"
    static let secretPlaceholder = "[REDACTED SECRET]"

    static func redact(_ text: String) -> String {
        rules.reduce(text) { text, rule in
            rule.regex.stringByReplacingMatches(
                in: text,
                range: NSRange(text.startIndex..., in: text),
                withTemplate: rule.template
            )
        }
    }

    private struct Rule {
        let regex: NSRegularExpression
        let template: String

        init(_ pattern: String, _ template: String) {
            // The patterns are constants, so a bad one is a programming error.
            regex = try! NSRegularExpression(pattern: pattern)
            self.template = template
        }
    }

    private static let rules: [Rule] = [
        // PEM private keys. Without an END line (cut off on screen), redact the base64 body that follows.
        Rule(
            #"-{2,}\s*BEGIN[A-Z ]*PRIVATE KEY\s*-{2,}(?:[\s\S]*?-{2,}\s*END[A-Z ]*PRIVATE KEY\s*-{2,}|[A-Za-z0-9+/=\\\s]*)"#,
            secretPlaceholder
        ),
        // Password in a connection URL: scheme://user:password@host
        Rule(#"\b([A-Za-z][A-Za-z0-9+.-]*://[^\s:/@]+:)[^\s@/]+@"#, "$1\(secretPlaceholder)@"),
        // Keys recognizable by their prefix alone.
        Rule(
            #"\b(?:sk-[A-Za-z0-9_-]{20,}|(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}|whsec_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|glpat-[A-Za-z0-9_-]{20,}|xox[abposr]-[A-Za-z0-9-]{10,}|AIza[A-Za-z0-9_-]{30,}|(?:AKIA|ASIA)[A-Z0-9]{16}|gsk_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{30,}|xai-[A-Za-z0-9]{20,}|pplx-[A-Za-z0-9]{20,}|r8_[A-Za-z0-9]{20,}|npm_[A-Za-z0-9]{30,}|ops_[A-Za-z0-9_-]{20,}|sntry[su]_[A-Za-z0-9_=+/-]{20,}|apify_api_[A-Za-z0-9]{20,}|tgp_v1_[A-Za-z0-9_-]{20,}|SG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}|eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})"#,
            secretPlaceholder
        ),
        Rule(#"(?i)\b(Bearer\s+)[A-Za-z0-9._~+/-]{16,}=*"#, "$1\(secretPlaceholder)"),
        // Values assigned to a secret-sounding name, as in .env files, config and JSON:
        // OPENAI_API_KEY=…, "clientSecret": "…", password: …
        // The value must contain a digit, which keeps code like `token = getToken()` intact.
        Rule(
            #"(?i)(?<![A-Za-z0-9_.-])([A-Za-z0-9_.-]{0,64}(?:api[_-]?key|[_-]key|secret|token|passw(?:or)?d|pwd|credentials?)(?:[_.-][A-Za-z0-9]+)*["']?\s*[:=]\s*["']?)(?!\[REDACTED)(?=[^\s"'`,;]*\d)[^\s"'`,;]{8,}"#,
            "$1\(secretPlaceholder)"
        ),
        // SSNs after a label, with any or no separators: "SSN: 123456789"
        Rule(
            #"(?i)(\b(?:SSN|social security(?:\s+(?:number|no\.?))?)\s*[:#]?\s*)\d{3}[- ]?\d{2}[- ]?\d{4}(?!\d)"#,
            "$1\(ssnPlaceholder)"
        ),
        // Dashed SSNs anywhere, skipping numbers the SSA never issues.
        Rule(#"(?<![\w-])(?!000|666|9)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\w-])"#, ssnPlaceholder),
    ]
}
