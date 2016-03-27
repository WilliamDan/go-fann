package fann

/*
#cgo LDFLAGS: -lfann -lm
#include <fann.h>

static void cpFannTypeArray(fann_type* src, fann_type* dst, unsigned int n) {
	unsigned int i = 0;
	for(; i < n; i++)
		dst[i] = src[i];
}

static void get_train_input(struct fann_train_data* td, fann_type* dst, unsigned int pos, unsigned int ln) {
	cpFannTypeArray(td->input[pos], dst, ln);
}

static void get_train_output(struct fann_train_data* td, fann_type* dst, unsigned int pos, unsigned int ln) {
	cpFannTypeArray(td->output[pos], dst, ln);
}

struct fann_train_data *push_train_data(
unsigned int max_data,
struct fann_train_data *data, double *din, double *dout
) {
	 unsigned int num_input = data->num_input;
	 unsigned int num_output = data->num_output;;
  unsigned int i, j;
  fann_type *data_input, *data_output;

 fann_init_error_data((struct fann_error *) data);
 if(data->num_data == 0){//如果是空的，那么就始化一下
	data->input = (double **) calloc(max_data, sizeof(double *));
	if(data->input == NULL) {
	    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
	    fann_destroy_train(data);
	    return NULL;
	  }
	  data->output = (double **) calloc(max_data, sizeof(double *));
	  if(data->output == NULL) {
	    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
	    fann_destroy_train(data);
	    return NULL;
	  }
	}


	if(data->num_data < max_data)data->num_data++;
	else{
		for(i= 0 ; i< max_data -1; i++){
			data->input[i] = data->input[ data->num_data - max_data + i+1];
			data->output[i] = data->output[ data->num_data - max_data + i+1];
		}
	}
  data_input = (double *) calloc(num_input, sizeof(double));
  if(data_input == NULL) {
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    fann_destroy_train(data);
    return NULL;
  }

  data_output = (double *) calloc(num_output, sizeof(double));
  if(data_output == NULL) {
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    fann_destroy_train(data);
    return NULL;
  }
  i = data->num_data - 1;
  data->input[i] = (double *) calloc(num_input , sizeof(double));
  data->output[i] = (double *) calloc(num_output , sizeof(double));
  data->input[i] = data_input;
 	for(j = 0; j != num_input; j++) {
      data->input[i][j] = din[j];
    }

    data->output[i] = data_output;

    for(j = 0; j != num_output; j++) {
      data->output[i][j] = dout[j];
    }

  return data;
}
*/
import "C"
import "unsafe"

/*
func PushExample(td *TrainData, din []float64, dout []float64,
	nMax int) *TrainData {
	td.object = C.push_train_data(C.uint(nMax),
		td.object,
		(*C.double)(unsafe.Pointer(&din[0])),
		(*C.double)(unsafe.Pointer(&dout[0])),
	)
	return td
}*/
func CreateTrain(nInput int, nOutput int) *TrainData {
	var td TrainData

	td.object = C.fann_create_train(0, C.uint(nInput), C.uint(nOutput))

	return &td
}
func ReadTrainFromFile(filename string) *TrainData {
	var td TrainData
	cfn := C.CString(filename)
	defer C.free(unsafe.Pointer(cfn))

	td.object = C.fann_read_train_from_file(cfn)

	return &td
}

/*
func MakeTrainFromArray() *TrainData {
	var td TrainData

	cfn := C.CString(filename)
	defer C.free(unsafe.Pointer(cfn))

	td.object = C.fann_read_train_from_file(cfn)

	return &td
}*/
func (td *TrainData) Destroy() {
	C.fann_destroy_train(td.object)
}
func (td *TrainData) PushExample(din []float64, dout []float64,
	nMax int) {
	td.object = C.push_train_data(C.uint(nMax),
		td.object,
		(*C.double)(unsafe.Pointer(&din[0])),
		(*C.double)(unsafe.Pointer(&dout[0])),
	)
}

func (td *TrainData) GetInput(i uint32) []FannType {
	num := td.GetNumInput()
	input := make([]FannType, num)
	C.get_train_input(td.object, (*C.fann_type)(&input[0]), C.uint(i), C.uint(num))
	return input
}

func (td *TrainData) GetOutput(i uint32) []FannType {
	num := td.GetNumOutput()
	output := make([]FannType, num)
	C.get_train_output(td.object, (*C.fann_type)(&output[0]), C.uint(i), C.uint(num))
	return output
}

func (td *TrainData) Shuffle() {
	C.fann_shuffle_train_data(td.object)
}

func (ann *Ann) ScaleTrain(td *TrainData) {
	C.fann_scale_train(ann.object, td.object)
}

func (ann *Ann) DescaleTrain(td *TrainData) {
	C.fann_descale_train(ann.object, td.object)
}

func (td *TrainData) Length() uint32 {
	return uint32(C.fann_length_train_data(td.object))
}

func MergeTrainData(td1 *TrainData, td2 *TrainData) *TrainData {
	var td TrainData
	td.object = C.fann_merge_train_data(td1.object, td2.object)
	return &td
}

func (td *TrainData) Duplicate() *TrainData {
	var td_dup TrainData
	td_dup.object = C.fann_duplicate_train_data(td.object)
	return &td_dup
}

func (td *TrainData) Subset(pos uint32, length uint32) *TrainData {
	var td_sub TrainData
	td_sub.object = C.fann_subset_train_data(td.object, C.uint(pos), C.uint(length))
	return &td_sub
}

func (td *TrainData) GetNumInput() uint32 {
	return uint32(C.fann_num_input_train_data(td.object))
}

func (td *TrainData) GetNumOutput() uint32 {
	return uint32(C.fann_num_output_train_data(td.object))
}

func (td *TrainData) SaveTrain(filename string) {
	cfn := C.CString(filename)
	defer C.free(unsafe.Pointer(cfn))
	C.fann_save_train(td.object, cfn)
}

func (td *TrainData) SaveTrainToFixed(filename string, decimal_point uint32) {
	cfn := C.CString(filename)
	defer C.free(unsafe.Pointer(cfn))

	C.fann_save_train_to_fixed(td.object, cfn, C.uint(decimal_point))
}

func (td *TrainData) ScaleInputTrainData(new_min FannType, new_max FannType) {
	C.fann_scale_input_train_data(td.object, C.fann_type(new_min), C.fann_type(new_max))
}

func (td *TrainData) ScaleOutputTrainData(new_min FannType, new_max FannType) {
	C.fann_scale_output_train_data(td.object, C.fann_type(new_min), C.fann_type(new_max))
}

func (td *TrainData) ScaleTrainData(new_min FannType, new_max FannType) {
	C.fann_scale_train_data(td.object, C.fann_type(new_min), C.fann_type(new_max))
}
