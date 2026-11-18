// Ported from teunlao/swift-ai-sdk, Sources/EventSourceParser/Parser.swift and Types.swift
// (commit 35c334a4, Apache-2.0; see LICENSE-APACHE-2.0 in this package).
// Modified for VoiceInkChat: turned into a value type that returns events instead of calling
// callbacks, dropped comment and parse-error reporting, and kept the WHATWG line handling and
// the UTF-8 split handling as they were.

import Foundation

/// One server-sent event.
public struct ServerSentEvent: Sendable, Equatable {
    public var id: String?
    /// The `event:` field, or `nil` for the default `message` type.
    public var event: String?
    /// Every `data:` line of the event, joined with newlines.
    public var data: String

    public init(id: String? = nil, event: String? = nil, data: String) {
        self.id = id
        self.event = event
        self.data = data
    }
}

/// An incremental `text/event-stream` parser, following the WHATWG rules: LF, CR and CRLF line
/// endings (including a CRLF split across chunks), a leading BOM, `:` comments, and multi-line
/// `data`. Chunks may split UTF-8 characters anywhere.
public struct ServerSentEventParser: Sendable {
    private var incompleteLine = ""
    private var isFirstChunk = true
    private var id: String?
    private var data = ""
    private var eventType = ""
    private var previousChunkEndedWithCR = false
    /// Bytes left from a chunk that ended in the middle of a UTF-8 character.
    private var pendingBytes = Data()
    private var events: [ServerSentEvent] = []

    public init() {}

    /// Feeds raw bytes and returns the events they complete.
    public mutating func feed(_ chunk: Data) -> [ServerSentEvent] {
        var bytes: Data
        if pendingBytes.isEmpty {
            bytes = chunk
        } else {
            bytes = pendingBytes
            bytes.append(chunk)
            pendingBytes.removeAll(keepingCapacity: true)
        }

        if let text = String(data: bytes, encoding: .utf8) {
            return feed(text)
        }
        let splitIndex = Self.utf8SafeSplitIndex(bytes)
        if splitIndex > 0, let text = String(data: bytes.prefix(splitIndex), encoding: .utf8) {
            pendingBytes = Data(bytes.suffix(from: bytes.startIndex + splitIndex))
            return feed(text)
        }
        pendingBytes = bytes
        return []
    }

    /// Feeds text and returns the events it completes.
    public mutating func feed(_ newChunk: String) -> [ServerSentEvent] {
        var chunk = newChunk
        if isFirstChunk, chunk.unicodeScalars.first?.value == 0xFEFF {
            chunk = String(chunk.unicodeScalars.dropFirst())
        }
        if chunk.isEmpty { return takeEvents() }

        if previousChunkEndedWithCR {
            // A CR at the end of the last chunk ends a line. If this chunk starts with LF, the
            // two are one CRLF.
            parseLine(incompleteLine)
            incompleteLine.removeAll(keepingCapacity: true)
            if chunk.unicodeScalars.first?.value == 10 {
                chunk = String(chunk.unicodeScalars.dropFirst())
            }
            previousChunkEndedWithCR = false
        }

        let (complete, incomplete, trailingCR) = Self.splitLines(chunk)
        for line in complete {
            if incompleteLine.isEmpty {
                parseLine(line)
            } else {
                parseLine(incompleteLine + line)
                incompleteLine.removeAll(keepingCapacity: true)
            }
        }
        incompleteLine += incomplete
        previousChunkEndedWithCR = trailingCR
        isFirstChunk = false
        return takeEvents()
    }

    /// Ends the stream. A final event without a trailing blank line is still delivered, which
    /// matches how providers close their streams in practice.
    public mutating func finish() -> [ServerSentEvent] {
        if !incompleteLine.isEmpty || previousChunkEndedWithCR {
            parseLine(incompleteLine)
            incompleteLine = ""
            previousChunkEndedWithCR = false
        }
        dispatchEvent()
        return takeEvents()
    }

    private mutating func takeEvents() -> [ServerSentEvent] {
        defer { events.removeAll(keepingCapacity: true) }
        return events
    }

    private mutating func parseLine(_ line: String) {
        if line.isEmpty {
            dispatchEvent()
            return
        }
        if line.hasPrefix(":") {
            return  // A comment, such as OpenRouter's ": OPENROUTER PROCESSING" keep-alive.
        }
        let field: String
        var value: String
        if let colon = line.firstIndex(of: ":") {
            field = String(line[..<colon])
            value = String(line[line.index(after: colon)...])
            if value.hasPrefix(" ") { value.removeFirst() }
        } else {
            field = line
            value = ""
        }
        switch field {
        case "event": eventType = value
        case "data": data += value + "\n"
        case "id": if !value.contains("\u{0000}") { id = value }
        default: break  // `retry` and unknown fields don't matter to a one-shot request.
        }
    }

    private mutating func dispatchEvent() {
        if !data.isEmpty {
            var payload = data
            if payload.last == "\n" { payload.removeLast() }
            events.append(ServerSentEvent(id: id, event: eventType.isEmpty ? nil : eventType, data: payload))
        }
        // Per the spec the last event ID persists, but nothing here reconnects, so reset it.
        id = nil
        data = ""
        eventType = ""
    }

    /// Splits on LF, CR and CRLF. A trailing CR is reported so a CRLF split across two chunks
    /// isn't read as two line endings.
    private static func splitLines(_ chunk: String) -> (complete: [String], incomplete: String, trailingCR: Bool) {
        var lines: [String] = []
        var current = String.UnicodeScalarView()
        let scalars = Array(chunk.unicodeScalars)
        var index = 0
        while index < scalars.count {
            let value = scalars[index].value
            if value == 13 {
                if index + 1 == scalars.count {
                    // Keep the line open: the next chunk may start with the LF of this CRLF.
                    return (lines, String(current), true)
                }
                lines.append(String(current))
                current = String.UnicodeScalarView()
                index += scalars[index + 1].value == 10 ? 2 : 1
            } else if value == 10 {
                lines.append(String(current))
                current = String.UnicodeScalarView()
                index += 1
            } else {
                current.append(scalars[index])
                index += 1
            }
        }
        return (lines, String(current), false)
    }

    /// The largest prefix of `data` that ends on a whole UTF-8 character.
    private static func utf8SafeSplitIndex(_ data: Data) -> Int {
        let bytes = [UInt8](data)
        let count = bytes.count
        guard count > 0 else { return 0 }
        if bytes[count - 1] < 0x80 { return count }
        var index = count - 1
        let limit = max(0, count - 4)
        while index > limit && bytes[index] & 0xC0 == 0x80 { index -= 1 }
        let lead = bytes[index]
        let expectedLength: Int
        if lead & 0x80 == 0 {
            expectedLength = 1
        } else if lead & 0xE0 == 0xC0 {
            expectedLength = 2
        } else if lead & 0xF0 == 0xE0 {
            expectedLength = 3
        } else if lead & 0xF8 == 0xF0 {
            expectedLength = 4
        } else {
            return count
        }
        return count - index >= expectedLength ? count : index
    }
}
