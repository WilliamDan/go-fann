package main

import (
	"fmt"

	"github.com/white-pony/go-fann"
)

func main() {
	const numLayers = 3
	const desiredError = 0.000001
	const maxEpochs = 500000
	const epochsBetweenReports = 1000

	//testData := fann.ReadTrainFromFile("../../datasets/xor.data")
	ann := fann.CreateStandard(numLayers, []uint32{2, 3, 1})
	ann.SetActivationFunctionHidden(fann.SIGMOID_SYMMETRIC)
	ann.SetActivationFunctionOutput(fann.SIGMOID_SYMMETRIC)
	//ann.TrainOnFile("../../datasets/xor.data", maxEpochs, epochsBetweenReports, desiredError)

	testData := fann.CreateTrain(2, 1)
	testData.PushExample([]float64{1, 0.5}, []float64{0.5}, 5)
	fmt.Println(testData.Length())
	testData.PushExample([]float64{0.2, 0.5}, []float64{0.1}, 5)
	fmt.Println(testData.Length())
	testData.PushExample([]float64{0.2, 0.1}, []float64{0.02}, 5)
	fmt.Println(testData.Length())
	testData.PushExample([]float64{0.2, 0.8}, []float64{0.16}, 5)
	fmt.Println(testData.Length())
	fmt.Println(testData.GetInput(0))
	testData.PushExample([]float64{0.7, 0.8}, []float64{0.56}, 5)
	fmt.Println(testData.Length())
	fmt.Println(testData.GetInput(0))
	testData.PushExample([]float64{0.3, 0.3}, []float64{0.09}, 5)
	fmt.Println(testData.Length())
	fmt.Println(testData.GetInput(0))
	testData.PushExample([]float64{0.4, 0.3}, []float64{0.12}, 5)
	fmt.Println(testData.Length())
	fmt.Println(testData.GetInput(0))
	testData.PushExample([]float64{0.4, 0.3}, []float64{0.12}, 5)
	fmt.Println(testData.Length())
	fmt.Println(testData.GetInput(0))
	fmt.Println("---------------")
	ann.TrainOnData(testData, maxEpochs, epochsBetweenReports, desiredError)
	tData := fann.CreateTrain(2, 1)
	tData.PushExample([]float64{1, 0.5}, []float64{-0}, 20)
	tData.PushExample([]float64{0.5, 0.5}, []float64{-0}, 20)
	tData.PushExample([]float64{0.3, 0.5}, []float64{-0}, 20)
	var i uint32
	for i = 0; i < testData.Length(); i++ {
		fmt.Println(testData.GetInput(i))
		fmt.Println(ann.Run(testData.GetInput(i)))
	}

	ann.Save("xor_float.net")

	ann.Destroy()
}
