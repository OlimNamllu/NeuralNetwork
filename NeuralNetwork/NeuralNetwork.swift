import Foundation
import Accelerate

// MARK: - Network

struct CapturedNeurons {
    var inputNeurons: [Double]
    var outputNeurons: [Double]
    var hiddenLayers: [[Double]]
    
    init() {
        self.inputNeurons = []
        self.outputNeurons = []
        self.hiddenLayers = []
    }
    
    var description: String {
        let inputs = "\nInputs: \(inputNeurons)"
        let outputs = "\nOutputs: \(outputNeurons)"
        
        var hidden = "\nHidden: "
        for (index, layer) in hiddenLayers.enumerated() {
            hidden.append("\nLayer \(index): \(layer)")
        }
        
        return inputs + hidden + outputs
    }
}


class NeuralNetwork {
//    static let layers = [textContextSize * vocab.count, 324, vocab.count]
    static let layers = [textContextSize * vocab.count, 324, 144, vocab.count]
    
    struct HiddenLayer: Equatable {
        var biases: [Double]
        
        //This is the weight to the next layer not previous
        var weights: [[Double]]
        
        init(layerCount: Int, nextLayerCount: Int) {
            biases = (0..<layerCount).map { _ in
                Double.random(in: -0.3...0.3)
            }
            
            weights = (0..<layerCount).map { _ in
                (0..<nextLayerCount).map { _ in
                    Double.random(in: -0.3...0.3)
                }
            }
        }
    }
    
    var hiddenLayers: [HiddenLayer] = {
        var newLayers = layers
        newLayers.removeFirst()
        
        var outputSize = newLayers.removeLast()
        
        return newLayers.enumerated().map { (index, layerSize) in
            if index < newLayers.count - 1 {
                return HiddenLayer(layerCount: layerSize, nextLayerCount: newLayers[index + 1])
            } else {
                return HiddenLayer(layerCount: layerSize, nextLayerCount: outputSize)
            }
        }
    }()
    
    var inputWeights: [[Double]] = (0..<layers[0]).map { _ in
        (0..<layers[1]).map { _ in Double.random(in: -0.3...0.3) } }
    
    var outputBiases: [Double] = (0..<layers.last!).map { _ in Double.random(in: -0.1...0.1) }
    
    struct Metrics {
        var mse: Double
        var accuracy: Double
        var top3: Double
        
        func printOut() -> String {
            return "MSE: \(mse), ACCURACY: \(accuracy), TOP3: \(top3)"
        }
    }
    
    func saveParameters(_ params: [Double]) {
        let expectedCount = {
            Self.layers.enumerated().map { (index, size) in
                if index < Self.layers.count - 1 {
                    return size + size * Self.layers[index + 1]
                } else {
                    return size
                }
            }.reduce(0, +) - Self.layers[0]
        }()
        
        guard params.count == expectedCount else { return }
        
        var currentIndex = 0
        
        // 1. Reconstruct inputWeights (layers[0] x layers[1])
        for i in 0..<Self.layers[0] {
            let nextIndex = currentIndex + Self.layers[1]
            inputWeights[i] = Array(params[currentIndex..<nextIndex])
            currentIndex = nextIndex
            
        }
        
        print("after inputs: \(currentIndex)")
                
        for (index, layer) in hiddenLayers.enumerated() {
            print("rep: \(currentIndex)")

            let biasesStartIndex = currentIndex
            let biasesEndIndex = currentIndex + layer.biases.count
            
            for i in biasesStartIndex..<biasesEndIndex {
                hiddenLayers[index].biases[i - biasesStartIndex] = params[i]
                currentIndex += 1
            }
            print("poop: \(currentIndex)")

            let weightsStartIndex = currentIndex
            let weightsEndIndex = currentIndex + layer.weights.count * layer.weights.first!.count
            
            for i in weightsStartIndex ..< weightsEndIndex {
                let relativeIndex = i - weightsStartIndex
                
                var rowNumDouble = (Double(relativeIndex) / Double(layer.weights[0].count))
                rowNumDouble.round(.down)
                let rowNum = Int(rowNumDouble)
                
                let columnNum = relativeIndex % layer.weights.first!.count
                
                hiddenLayers[index].weights[rowNum][columnNum] = params[i]
                
                currentIndex += 1
            }
        }
        
        print("after middle: \(currentIndex)")

        
        // 4. Reconstruct outputBiases (layers[2])
        let outputBiasesEnd = currentIndex + Self.layers.last!
        outputBiases = Array(params[currentIndex..<outputBiasesEnd])
        
        
        print(outputBiasesEnd, expectedCount)
    }
    
    func list() -> [Double] {
        var list = [Double]()
        
        for array in inputWeights {
            list.append(contentsOf: array)
        }
        
        for layer in hiddenLayers {
            list.append(contentsOf: layer.biases)
            list.append(contentsOf: layer.weights.flatMap(\.self))
        }
        
        list.append(contentsOf: outputBiases)
        
        return list
    }
    
    enum NetworkErrors: Error {
           case invalidInputs
    }
       
    //    func mapRange(_ x: Double) -> Double {
    //        return (2.0 / 9.0) * x - 1.0
    //    }
       
    enum RandomNumberType {
        case human
        case computer
        case logicError
    }
    
    static func cost(output: [Double], answer: [Double]) -> Double {
        guard output.count == Self.layers.last else { return Double.infinity }
        guard answer.count == Self.layers.last else { return Double.infinity }
        
        var sum: Double = 0
        
        for (output, answer) in zip(output, answer) {
            sum += pow(output - answer, 2)
        }
        
        
        return sum
    }
    
    func tanhDerivative(_ postActivation: Double) -> Double {
        return 1 - pow(postActivation, 2)
    }
    
    func flattenArray2D(_ input: [[Double]]) -> (Int, Int, [Double]) {
        return (input.count, input.first!.count, input.flatMap() {$0})
    }
    
    func flattenArray3D(_ input: [[[Double]]]) -> (Int, Int, Int, [Double]) {
        return (input.count, input.first!.count, input.first!.first!.count, input.flatMap() {$0.flatMap(){$0}})
    }
    
    // MARK: - Backpropagation
    func backprop(iterations: Int, learnRate: Double, batchSize: Int) {
        
        for i in 0..<iterations {
            if i % 20 == 0 {
                print("Iteration: \(i), \(self.evaluate(on: validationStarts))")
            }
            // define partials
            var inputWeightPartials: [[Double]] = Array(repeating: Array(repeating: 0, count: Self.layers[1]), count: Self.layers[0])
            var outputBiasPartials: [Double] = Array(repeating: 0, count: Self.layers.last!)
            var middleBiasPartials: [[Double]] = Array(repeating: 0, count: Self.layers.count - 2).enumerated().map { layerNum in
                Array(repeating: 0, count: Self.layers[layerNum.0 + 1])
            }
            var middleWeightPartials: [[[Double]]] = Array(repeating: 0, count: Self.layers.count - 2).enumerated().map { layerNum in
                Array(repeating: Array(repeating: 0, count: Self.layers[layerNum.0 + 2]), count: Self.layers[layerNum.0 + 1])
            }
            
            for _ in 0..<batchSize {
                // Reference point: PREACTIVATION
                var referenceMatrix: [[Double]] = Array(repeating: 0, count: Self.layers.count-1).enumerated().map { layerNum in
                    Array(repeating: 0, count: Self.layers[layerNum.0 + 1])
                }
                
                let seed = sampleSeed
                let inputs = encode(startingAt: sampleSeed)
                let target = makeTarget(startingAt: seed)
                
                let datum: ([Double], [Double]) = (inputs, target)
                
                let (_, captured) = self.run(datum.0)
                
                
                // add changes to partials
                for (index, outputNeuron) in captured.outputNeurons.enumerated() {
                    outputBiasPartials[index] += 2 * (outputNeuron - datum.1[index]) * tanhDerivative(outputNeuron)
                    
                    referenceMatrix[referenceMatrix.count - 1][index] = (outputNeuron - datum.1[index]) * tanhDerivative(outputNeuron)
                }
                
                for (index, middleLayer) in captured.hiddenLayers.reversed().enumerated() {
                    let actualIndex = captured.hiddenLayers.count - index - 1
                    for (syndex, middleNeuron) in middleLayer.enumerated() {
                        //SET REFERENCE PARTIALS
                        for (lindex) in 0..<(index==0 ? captured.outputNeurons : captured.hiddenLayers[actualIndex + 1]).count {
                            referenceMatrix[actualIndex][syndex] += tanhDerivative(middleNeuron) * hiddenLayers[actualIndex].weights[syndex][lindex] * referenceMatrix[actualIndex + 1][lindex]
                            
                            middleWeightPartials[actualIndex][syndex][lindex] += captured.hiddenLayers[actualIndex][syndex] * referenceMatrix[actualIndex+1][lindex]
                        }
                        
                        middleBiasPartials[actualIndex][syndex] += referenceMatrix[actualIndex][syndex]
                    }
                }
                
                for (index, _) in captured.inputNeurons.enumerated() {
                    for (syndex, _) in captured.hiddenLayers[0].enumerated() {
                        inputWeightPartials[index][syndex] += captured.inputNeurons[index] * referenceMatrix[0][syndex]
                    }
                }
                
            }
            
            let factor = learnRate / Double(batchSize)
            
            for j in 0..<inputWeightPartials.count {
                vDSP.multiply(factor, inputWeightPartials[j], result: &inputWeightPartials[j])
                for (index, k) in inputWeightPartials[j].enumerated() {
                    inputWeights[j][index] -= k
                }
            }
            
            vDSP.multiply(factor, outputBiasPartials, result: &outputBiasPartials)
            for (index, k) in outputBiasPartials.enumerated() {
                outputBiases[index] -= k
            }
            
            for j in 0..<middleBiasPartials.count {
                vDSP.multiply(factor, middleBiasPartials[j], result: &middleBiasPartials[j])
                for (index, k) in middleBiasPartials[j].enumerated() {
                    hiddenLayers[j].biases[index] -= k
                }
            }
            
            for j in 0..<middleWeightPartials.count {
                for k in 0..<middleWeightPartials[j].count {
                    vDSP.multiply(factor, middleWeightPartials[j][k], result: &middleWeightPartials[j][k])
                    for (index, l) in middleWeightPartials[j][k].enumerated() {
                        hiddenLayers[j].weights[k][index] -= l
                    }
                }
            }
        }
    }
     
    func prediction(from context: String) -> String {
        let inputs = generateOneHotVectors(for: context)
        let (outputs, _) = run(inputs)
        return softmax(outputs)
    }
    
    func generate(from context: String, characters: Int) -> String {
        var result = context
        for _ in 0..<characters {
            result.append(prediction(from: result))
        }
        return result
    }
    
    func argmax(_ outputs: [Double]) -> String {
        var max: (value: Double, index: Int) = (-Double.infinity, 0)
        for (i, o) in outputs.enumerated() {
            if o > max.value { max.value = o; max.index = i }
        }
        return String(vocab[max.index])
    }
    
    func softmax(_ outputs: [Double], temperature: Double = 0.08) -> String {
        let scaled = outputs.map { $0 / temperature }
        let m = scaled.max()!
        let exps = scaled.map { exp(($0 - m)) }
        let sum = exps.reduce(0, +)
        
        var r = Double.random(in: 0..<sum)
        for (k, e) in exps.enumerated() {
            r -= e
            if r <= 0 { return String(vocab[k]) }
        }
        return String(vocab[exps.count - 1])
    }
       
    func run(_ inputs: [Double]) -> ([Double], CapturedNeurons) {
        guard inputs.count == Self.layers[0] else {
            fatalError("Input count mismatch")
        }
        
        let mapped = inputs
        
        var neurons = CapturedNeurons()
        neurons.inputNeurons = mapped
        

        neurons.hiddenLayers = (0..<hiddenLayers.count).map { layerIndex in
            Array(repeating: 0.0, count: Self.layers[layerIndex + 1])
        }

        neurons.outputNeurons = Array(repeating: 0.0, count: Self.layers.last!)
        
        let outputLayer = Self.layers.count - 1

        var currentValues = mapped
        for layer in 1...outputLayer {
            for neuron in 0..<Self.layers[layer] {
                var sum: Double = 0

                for (index, input) in currentValues.enumerated() {
//                    if input == 0 { continue }

                    var scaled: Double = 0

                    if layer == 1 {
                        scaled = input * inputWeights[index][neuron]
                    } else {
                        scaled = input * hiddenLayers[layer - 2].weights[index][neuron]
                    }

                    sum += scaled
                }

                if layer == outputLayer {
                    sum += outputBiases[neuron]
                    neurons.outputNeurons[neuron] = tanh(sum)
                } else {
                    sum += hiddenLayers[layer - 1].biases[neuron]
                    neurons.hiddenLayers[layer - 1][neuron] = tanh(sum)
                }

            }

            if layer == outputLayer {
                currentValues = neurons.outputNeurons
            } else {
                currentValues = neurons.hiddenLayers[layer - 1]
            }
        }
        
        return (currentValues, neurons)
    }
    
    func evaluate(on starts: [Int]) -> Metrics {
        var squaredError: Double = 0
        var correct = 0
        var inTop3 = 0
        
        for start in starts {
            let input = encode(startingAt: start)
            let target = makeTarget(startingAt: start)
            let answer = sampleData[start + textContextSize]
            let output = self.run(input).0
            
            for k in 0..<27 {
                let e = output[k] - target[k]
                squaredError += e * e
            }
            
            // argmax
            var best = 0
            for k in 1..<27 where output[k] > output[best] { best = k }
            if best == answer { correct += 1 }
            
            // is the true answer among the three highest outputs?
            var betterCount = 0
            for k in 0..<27 where output[k] > output[answer] { betterCount += 1 }
            if betterCount < 3 { inTop3 += 1 }
        }
        
        let n = Double(starts.count)
        
        return Metrics(
            mse: squaredError / (n * 27),
            accuracy: Double(correct) / n,
            top3: Double(inTop3) / n
        )
    }
}

