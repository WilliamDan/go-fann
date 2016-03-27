package main

import (
	"encoding/csv"
	"fmt"
	"io"
	//"log"
	"os"

	//	"github.com/white-pony/go-fann"
)

func prepareData() {
	//read data from csv and make them to the train data file
	sFile := "ssc.csv"
	file, _ := os.Open(sFile)
	r := csv.NewReader(file)

	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			//log.Fatal(err)
			//fmt.Println(err)
		}

		fmt.Println(record)
	}

}
func main() {

	const num_layers = 4
	const num_neurons_hidden = 96
	const desired_error = 0.001

	fmt.Println("Creating network.")

	train_data := fann.ReadTrainFromFile("../../datasets/robot.train")

	ann := fann.CreateStandard(num_layers,
		[]uint32{train_data.GetNumInput(), num_neurons_hidden, num_neurons_hidden, train_data.GetNumOutput()})

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
