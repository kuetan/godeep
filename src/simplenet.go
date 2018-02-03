package main

import (
	"math/rand"
	"math"
	"./setup"
	"fmt"
	"github.com/gonum/matrix/mat64"
)


func setup(input_size int , hidden_size int,output_size int) (w1 ,b1,w2,b2 *mat64.Dense) {
	w1_init := make([]float64, input_size * hidden_size)
	for i := range w1_init {
		w1_init[i] = rand.NormFloat64()
		
	}
	// fmt.Println(w1_init)
	w1 = mat64.NewDense(input_size, hidden_size, w1_init)
	b1 = mat64.NewDense(input_size, hidden_size, nil) 
	w2_init := make([]float64, hidden_size * output_size )
	for i := range w2_init {
		w2_init[i] = rand.NormFloat64()
		
	}
	w2 = mat64.NewDense(hidden_size, output_size, w2_init)
	b2 = mat64.NewDense(hidden_size, output_size, nil) 
	return w1,b1,w2,b2
}

func Sigmoid(x float64) (y float64) {
	return 1.0 / (1.0 + math.Exp(-1*x))
}

func predict(x ,w1,b1,w2,b2 *mat64.Dense) (y *mat64.Dense) {
	var a1 *mat64.Dense
	a1.Mul(x, w1) 
	a1.Add(a1,b1)
	// for i := 0; i < len(a1); i++ {
	// 	z1 := append(a1, Sigmoid(a1[i]))
	// }
	//z1 := math.Exp(a1)
	z1 := a1
	var a2 *mat64.Dense
	a2.Mul(z1, w2) 
	a2.Add(a2,b2)
	//a2 += b2
	// for i := 0; i < len(a2); i++ {
	// 	y := append(a2, Sigmoid(a2[i]))
	// }
	
	y = a2
	//y = math.exp(a2)
	return y
}

func cross_entropy_error( *mat64.Dense) *mat64.Dense {
	delta :=1e-7
	return -Sum(MulElem(t,log(y+delta)))
}

func loss(x,t,w1,b1,w2,b2 mat64.Dense) *mat64.Dense  {
	y := predict(x,t,w1,b1,w2,b2)
	return cross_entropy_error(y,t)
}

// func accuracy(x,t mat64.Dense) {
// 	y:=predict(x)
// }



func main() {
	train, test, err := read.Load("./dataset/mnist/")
	train = train
	if err == nil { 
		fmt.Println(test.Images[0]) 
	}
	// fmt.Print(rand.NormFloat64())
	// fmt.Print(rand.NormFloat64())
	// fmt.Print(rand.NormFloat64())
}
