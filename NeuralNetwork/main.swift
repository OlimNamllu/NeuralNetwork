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

let trainingImages = fetchCGImagesFromLocalPath(folderPath: "/Users/miloullman/Desktop/NeuralNetwork/NeuralNetwork/pumpkin")

func runAndSave(sample: Int, to file: String) {
    let datum = ImageDatum(cgimage: trainingImages[sample])
    var output = network.run(datum.getTraningDatum().0).0.map { UInt8(($0 + 1) * 128) }
    
    var rgb: [ImageDatum.RGBPixel] = []
    for i in 0..<(output.count / 3) {
        let step = i * 3
        var pixel = ImageDatum.RGBPixel(r: 0, g: 0, b: 0)
        pixel.r = output[step + 0]
        pixel.g = output[step + 1]
        pixel.b = output[step + 2]
        
        rgb.append(pixel)
    }
    let result = ImageDatum(from: rgb)
    ImageDatum.exportCGImageToJpeg(img: result.getCGImage()!, name: "saving", overrideName: "file:///Users/miloullman/Desktop/NeuralNetwork/NeuralNetwork/\(file).jpg")
}

func save(sample: Int, bw: Bool, file: String) {
    let datum = ImageDatum(cgimage: trainingImages[sample])
    
    let bwDatum = ImageDatum(from: datum.bwToRGB())
    
    ImageDatum.exportCGImageToJpeg(img: (bw ? bwDatum.getCGImage()! : datum.getCGImage()!), name: "saving", overrideName: "file:///Users/miloullman/Desktop/NeuralNetwork/NeuralNetwork/\(file).jpg")
    
}

// MARK: - EXECUTE
var network = NeuralNetwork()
print("param count: \(network.paramCount)")

let testIndicies: [Int] = Array(220..<225)

for i in 0..<testIndicies.count {
    save(sample: testIndicies[i], bw: true, file: "original\(i)")
}

for i in 0..<testIndicies.count {
    runAndSave(sample: testIndicies[i], to: "pretrain\(i)")
}

// TRAINING
print("Training...")
network.backprop(iterations: 750, learnRate: 0.001, batchSize: 4)
network.backprop(iterations: 1000, learnRate: 0.0001, batchSize: 4)
print("Training Complete")


// POST TRAINING

for i in 0..<testIndicies.count {
    runAndSave(sample: testIndicies[i], to: "posttrain\(i)")
}
