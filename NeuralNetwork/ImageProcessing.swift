//
//  ImageProcessing.swift
//  NeuralNetwork
//
//  Created by Milo Ullman on 17/10/26.
//
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers



func fetchCGImagesFromLocalPath(folderPath: String) -> [CGImage] {
    // 1. Convert the string path to a standard file URL
    // Resolving symlinks and expanding tilde (~) makes the path robust
    let standardPath = (folderPath as NSString).expandingTildeInPath
    let folderURL = URL(fileURLWithPath: standardPath, isDirectory: true)
    
    var cgImages: [CGImage] = []
    let fileManager = FileManager.default
    
    do {
        // 2. Scan the physical directory contents on your Mac
        let fileURLs = try fileManager.contentsOfDirectory(at: folderURL, includingPropertiesForKeys: nil)
        
        // 3. Case-insensitive filter for JPG/JPEG extensions
        let jpgURLs = fileURLs.filter {
            let ext = $0.pathExtension.lowercased()
            return ext == "jpg" || ext == "jpeg"
        }
        
        if jpgURLs.isEmpty {
            print("Warning: No JPG images found at path: \(standardPath)")
            return []
        }
        
        // 4. Create CGImage directly from the local file URLs
        for url in jpgURLs {
            if let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil),
               let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) {
                cgImages.append(cgImage)
            } else {
                print("Failed to decode image at: \(url.lastPathComponent)")
            }
        }
        
    } catch {
        print("Error reading local directory: \(error.localizedDescription)")
    }
    
    // Returns the array of raw CGImage objects
    return cgImages
}



struct ImageDatum {
    struct RGBPixel {
        var r: UInt8
        var g: UInt8
        var b: UInt8
        var a: UInt8 = 255
    }
    
    struct BWPixel {
        var c: UInt8
    }
    
    var RGB: [RGBPixel]
    var BW: [BWPixel]
    
    var width: Int = 52
    var height: Int = 64
    
    var unflattened: [[RGBPixel]] {
        var matrix: [[RGBPixel]] = []
        
        for rowStart in stride(from: 0, to: RGB.count, by: width) {
            let rowEnd = min(rowStart + width, RGB.count)
            let row = Array(RGB[rowStart..<rowEnd])
            matrix.append(row)
        }
        return matrix
    }
    
    func printOut() -> [[(Int, Int, Int)]] {
        var output: [[(Int, Int, Int)]] = []
        
        for row in unflattened {
            var inter: [(Int, Int, Int)] = []
            for col in row {
                inter.append((Int(col.r), Int(col.g), Int(col.b)))
            }
            
            output.append(inter)
        }
        return output
    }
    
    
    init(cgimage: CGImage) {
        guard let dataProvider = cgimage.dataProvider,
              let data = dataProvider.data,
              let pointer = CFDataGetBytePtr(data) else {
            fatalError("Couldn't access image data")
        }
        
        self.RGB = []
        self.BW = []
        
        let width = cgimage.width
        let height = cgimage.height
        let bytesPerRow = cgimage.bytesPerRow
        let bytesPerPixel = cgimage.bitsPerPixel / cgimage.bitsPerComponent
        
        self.width = width
        self.height = height
        
        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * bytesPerRow) + (x * bytesPerPixel)
                let red = pointer[offset]
                let green = pointer[offset + 1]
                let blue = pointer[offset + 2]
                self.RGB.append(RGBPixel(r: red, g: green, b: blue))
            }
        }
        
        for pixel in self.RGB {
            let grayscale = (pixel.r / 3) + (pixel.g / 3) + (pixel.b / 3)
            self.BW.append(BWPixel(c: grayscale))
        }
    }
    
    func getCGImage() -> CGImage? {
        
        let data = NSData(bytes: RGB, length: 8 * 4 * height * width)
        
        guard let provider = CGDataProvider(data: data) else { return nil }
        
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 8 * 4,
            bytesPerRow: 4 * width,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: [CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                         .byteOrder32Big],
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )
    }
    
    func bwToRGB() -> [RGBPixel] {
        return self.BW.map { element in
            RGBPixel(r: element.c, g: element.c, b: element.c, a: 255)
        }
    }
    
    
    @discardableResult
    static func exportCGImageToJpeg(img: CGImage, name: String, overrideName: String? = nil) -> Bool {
        
        var url = URL(string: getPath(filename: name))
        
        if overrideName != nil {
            url = URL(string: overrideName!)
        }
        
        guard let destUrl = url else { return false }
        
        // 1. Create the destination for the JPEG file
        guard let destination = CGImageDestinationCreateWithURL(
            destUrl as CFURL,
            UTType.jpeg.identifier as CFString,
            1, // Count of images in the file
            nil
        ) else {
            return false
        }
        
        // 2. Set the compression quality (0.0 to 1.0)
        let options: [CFString: Any] = [
            kCGImageDestinationLossyCompressionQuality: 1
        ]
        
        // 3. Add the image and finalize the destination
        CGImageDestinationAddImage(destination, img, options as CFDictionary)
        return CGImageDestinationFinalize(destination)
    }
    
}
