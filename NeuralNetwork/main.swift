//
//  main.swift
//  NeuralNetwork
//
//  Created by Milo Ullman on 3/8/26.
//


import Foundation

// MARK: - Saving Parameters with Data()
@discardableResult
func writeParameters(_ params: [Double], to filename: String) -> Bool {
    let data = params.withUnsafeBufferPointer { Data(buffer: $0) }
    let url = URL(filePath: getPath(filename: filename))
    do {
        try data.write(to: url, options: .atomic)
        print("wrote \(params.count) parameters to \(filename)")
        return true
    } catch {
        print("could not write \(filename): \(error)")
        return false
    }
}

func readParameters(from filename: String) -> [Double]? {
    let url = URL(filePath: getPath(filename: filename))

    guard let data = try? Data(contentsOf: url) else {
        print("no parameter file at \(filename)")
        return nil
    }

    guard data.count % MemoryLayout<Double>.stride == 0 else {
        print("\(filename) is \(data.count) bytes, not a multiple of \(MemoryLayout<Double>.stride)")
        return nil
    }

    let count = data.count / MemoryLayout<Double>.stride
    let params = data.withUnsafeBytes { raw in
        (0..<count).map { i in
            raw.loadUnaligned(fromByteOffset: i * MemoryLayout<Double>.stride, as: Double.self)
        }
    }
    print("read \(params.count) parameters from \(filename)")
    return params
}

// MARK: - Saving Scores JSON

var scores: [Double] = []

let path2 = "/Users/miloullman/Desktop/NeuralNetwork/NeuralNetwork/Scores.swift"
let url2 = URL(fileURLWithPath: path2)

func writeScores(_ scores: [Double], title: String) {
    let swift = "let scores\(title): [Double] = \(scores) \n \n"
    
    if let fileHandle = try? FileHandle(forWritingTo: url2) {
        defer { try? fileHandle.close() } // Always close the file stream
        let _ = try? fileHandle.seekToEnd()
        if let data = swift.data(using: .utf8) {
            try? fileHandle.write(contentsOf: data)
        }
    }
}

var tim = "/Users/tim/Documents/neural_network/NeuralNetwork/"
var milo = "/Users/miloullman/Desktop/NeuralNetwork/NeuralNetwork/"

func getPath(filename: String) -> String {
    return tim + filename
}


// MARK: - Files

// Files for saving parameters between training runs
let parametersBefore = "Parameters-before.bin"
let parametersAfter = "Parameters-after.bin"


// 36000 iterations, 0.03-0.05 noise, 28 character context, 756 256 180 27 network layers
let iter36k = "iter36k.bin"

// Building off previous network, 9k more iterations at 0.07 noise
let iter45k = "iter42k.bin"

// 18000 iterations 486, 256, 180, 96, 27 Network, higher noise
let iter18k5L = "iter18k5L.bin"

// 18000 iterations 486, 256, 180, 96, 27 Network, higher noise
let iter30k5L = "iter30k5L.bin"


// MARK: - EXECUTE
var network = NeuralNetwork()

let images = fetchCGImagesFromLocalPath(folderPath: "/Users/tim/Documents/neural_network/NeuralNetwork/pumpkin")

let sample = images[3]


let datum = ImageDatum(cgimage: sample)

print(datum.unflattened.count)
print(datum.unflattened.first!.count)

ImageDatum.exportCGImageToJpeg(img: datum.getCGImage()!, name: "67skibidkidtoiletl", overrideName: "file:///Users/tim/Documents/neural_network/NeuralNetwork/67676.jpg")
