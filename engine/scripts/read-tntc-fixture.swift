// Reads the TNTC track written by the Rust engine and checks it with CryptoKit.
//
// Compiled with the engine's own chunk reader:
//
//   swiftc -parse-as-library -O -o /tmp/read-tntc-fixture \
//     engine/Sources/TintaEngine/Protocol.swift \
//     engine/Sources/TintaEngine/ChunkStore.swift \
//     engine/scripts/read-tntc-fixture.swift
//   /tmp/read-tntc-fixture <dir>
//
// The fixture comes from `crates/tinta-core/tests/tntc_fixture.rs`
// (`writes_rust_chunks`): `audio/remote` with two chunks, start 0 (700 samples)
// and start 2500 (300 samples), the key 0x00..0x1f, and repeating sample
// patterns that `expected` reproduces.

import CryptoKit
import Foundation

@main
struct ReadTntcFixture {
    enum Failure: Error, CustomStringConvertible {
        case message(String)
        var description: String {
            switch self {
            case .message(let text): return text
            }
        }
    }

    static let pattern: [Float] = [0, 0.25, -0.25, 0.5, -0.5, 1.0, -1.0]

    static func expected(_ index: Int) -> Float {
        pattern[index % pattern.count]
    }

    static func main() throws {
        guard CommandLine.arguments.count == 2 else {
            FileHandle.standardError.write(Data("usage: read-tntc-fixture <dir>\n".utf8))
            exit(2)
        }
        let directory = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
        let key = SymmetricKey(data: Data((0..<32).map { UInt8($0) }))

        let chunks = try ChunkReader.chunks(directory: directory, key: key, track: .remote)
        guard chunks.count == 2 else {
            throw Failure.message("expected 2 chunks, read \(chunks.count)")
        }
        guard chunks[0].start == 0, chunks[0].samples.count == 700 else {
            throw Failure.message(
                "chunk 0 is start \(chunks[0].start), \(chunks[0].samples.count) samples")
        }
        guard chunks[1].start == 2500, chunks[1].samples.count == 300 else {
            throw Failure.message(
                "chunk 1 is start \(chunks[1].start), \(chunks[1].samples.count) samples")
        }

        let track = try ChunkReader.track(directory: directory, key: key, track: .remote)
        guard track.count == 2800 else {
            throw Failure.message("expected 2800 samples on the meeting clock, got \(track.count)")
        }
        for index in 0..<700 where abs(track[index] - expected(index)) > 1e-4 {
            throw Failure.message(
                "chunk 0 sample \(index): \(track[index]) != \(expected(index))")
        }
        for index in 700..<2500 where track[index] != 0 {
            throw Failure.message("gap sample \(index) is \(track[index]), not silence")
        }
        for index in 2500..<2800 where abs(track[index] - expected(index - 2500)) > 1e-4 {
            throw Failure.message(
                "chunk 1 sample \(index - 2500): \(track[index]) != \(expected(index - 2500))")
        }

        // The microphone track was never written.
        let mic = try ChunkReader.chunks(directory: directory, key: key, track: .mic)
        guard mic.isEmpty else {
            throw Failure.message("the mic track should be empty, got \(mic.count) chunks")
        }
        print("read \(track.count) samples from the Rust fixture")
    }
}
