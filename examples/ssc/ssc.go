package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"

	"github.com/white-pony/go-fann"
)

func prepareData() {
	nIn := 100
	//	sTrainData := "ssc.data"
	file, err := os.Open("ssc.csv")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	defer file.Close()
	aOut := make([]string, 5)
	reader := csv.NewReader(file)
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			fmt.Println("Error:", err)
			return
		}
		aOut = append(aOut, record[1:]...)

		fmt.Println(aOut) // record has the type []string
		if len(aOut) > nIn {
			break
		}
	}
}
func addExample([]int, []int) {
	//在这里加入一个用例，output
}
func main() {
	const num_layers = 3
	const num_neurons_hidden = 96
	const desired_error = 0.001
	prepareData()
	fmt.Println("Creating network.")
	train_data := fann.CreateTrain(10, 1)
	var aInput []float64
	var aOut []float64
	aInput = make([]float64, 10)
	for i := 0; i < 10; i++ {
		aInput[i] = 0.5
	}
	aOut = make([]float64, 1)
	train_data = fann.PushExample(train_data, aInput, aOut, 10, 1, 20)
	fmt.Println(train_data.Length())
	for i := 0; i < 10; i++ {
		aInput[i] = 0.6
	}
	train_data = fann.PushExample(train_data, aInput, aOut, 10, 1, 20)
	fmt.Println(train_data.GetInput(1))
	ann := fann.CreateStandard(num_layers, []uint32{train_data.GetNumInput(), num_neurons_hidden, train_data.GetNumOutput()})

	fmt.Println("Training network.")

	ann.SetTrainingAlgorithm(fann.TRAIN_INCREMENTAL)
	ann.SetLearningMomentum(0.4)

	ann.TrainOnData(train_data, 3000, 10, desired_error)

	fmt.Println("Testing network")

	test_data := fann.ReadTrainFromFile("../../datasets/robot.test")

	ann.ResetMSE()

	var i uint32
	for i = 0; i < test_data.Length(); i++ {
		ann.Test(test_data.GetInput(i), test_data.GetOutput(i))
	}

	fmt.Printf("MSE error on test data: %f\n", ann.GetMSE())

	fmt.Println("Saving network.")
	ann.Save("robot_float.net")
	fmt.Println("Cleaning up.")

	train_data.Destroy()
	test_data.Destroy()
	ann.Destroy()

}
