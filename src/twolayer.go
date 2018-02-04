package main

import (
	"math/rand"
	"math"
	"./setup"
	"fmt"
	"github.com/gonum/matrix/mat64"
//	"reflect"
)
type AffineLayer interface {
	init(w,b *mat64.Dense)
	forward(x *mat64.Dense)
	backward(dout *mat64.Dense)
}

type AffineLayer_param struct  {
	w *mat64.Dense
	b *mat64.Dense
	x *mat64.Dense
	dw *mat64.Dense
	db *mat64.Dense
}

func (p *AffineLayer_param) init(w,b *mat64.Dense)  {
	p.w = w
	p.b = b
	p.x = nil
	p.dw = nil
	p.db = nil
}

func (p *AffineLayer_param) forward(x *mat64.Dense)  *mat64.Dense {
	p.x = x
	var out *mat64.Dense
	out.Mul(x,p.w)
	out.Add(out,p.b)
	return out
}

func (p *AffineLayer_param) backward(dout *mat64.Dense)  *mat64.Dense {
	var dx *mat64.Dense
	dx.Mul(dout,p.w.T())
	p.dw.Mul(p.x.T(),dout)
	r ,c :=dout.Dims()
	data := make([]float64, c)
	for i := range data {
		data[i] = mat64.Sum(dout.ColView(i))
	}
	p.db = mat64.NewDense(r, 1, data)
	return dx
}

type ReluLayer interface {
	init()
	forward(x *mat64.Dense)
	backward(dout *mat64.Dense)
}

type ReluLayer_param struct  {
	mask *mat64.Dense
}

func (p *ReluLayer_param) init()  {
	p.mask = nil
}

func (p *ReluLayer_param) forward(x *mat64.Dense) *mat64.Dense {
	relu_for := func(i,j int, v float64) float64 {
		if (v < 0) {v = 0}
		return v
	}
	relu_back := func(i,j int, v float64) float64 {
		if (v < 0) { v = 0
		} else {v = 1}
		return v
	}
	x.Apply(relu_for,x)
	p.mask.Apply(relu_back,x)
	return x
}

func (p *ReluLayer_param) backward(dout *mat64.Dense) *mat64.Dense {
	dout.MulElem(dout,p.mask)
	return dout
}

type SoftmaxWithLossLayer interface {
	init()
	forward(x *mat64.Dense,t *mat64.Dense)
	backward(dout int)
}

type SoftmaxWithLossLayer_param struct  {
	loss *mat64.Dense
	y *mat64.Dense
	t *mat64.Dense
}

func (p *SoftmaxWithLossLayer_param) init()  {
	p.loss = nil
	p.y = nil
	p.t = nil
}

func cross_entropy_error(t,y *mat64.Dense) *mat64.Dense {
	log := func(i,j int, v float64) float64 {
		delta :=1e-7
		res :=math.Log(v + delta)
		return res
	}
	y.Apply(log,y)
	t.MulElem(t,y)
	r ,c :=t.Dims()
	sum_data := make([]float64, c)
	for i := range sum_data {
		sum_data[i] = mat64.Sum(t.ColView(i))
	}
	sum := mat64.NewDense(r,1,sum_data)
	return sum
}

func softmax(x *mat64.Dense) *mat64.Dense {
	var exp_x *mat64.Dense
	exp_x.Exp(x)
	r ,c :=x.Dims()
	sum_data := make([]float64, c)
	for i := range sum_data {
		sum_data[i] = mat64.Sum(exp_x.ColView(i))
	}
	sum := mat64.NewDense(r,1,sum_data)
	return sum
}

func (p *SoftmaxWithLossLayer_param) forward(x *mat64.Dense,t *mat64.Dense)  *mat64.Dense {
	p.t = t
	p.y = softmax(x)
	p.loss = cross_entropy_error(p.y,p.t)
	return p.loss
}

func (p *SoftmaxWithLossLayer_param) backward(dout int)  *mat64.Dense {
	var dx *mat64.Dense
	r,c :=p.t.Dims()
	batch_size := r
	c = c
	div := func(i,j int, v float64) float64 {
		return v / float64(batch_size)
	}
	p.t.Apply(div,p.t)
	p.y.Sub(p.y,p.t)
	dx = p.y
	return dx
}

type TwoLayerNet interface {
	init(input_size,hidden_size,output_size int,weight_init_std float64)
//	predict(x *mat64.Dense)
	loss(x,t *mat64.Dense)
	gradient(x,t *mat64.Dense) *Grads
}

	// loss(x,t *mat64,Dense)
	// accuracy(x,t *mat64,Dense)

type TwoLayerNet_params struct {
	w1 *mat64.Dense
	b1 *mat64.Dense
	w2 *mat64.Dense
	b2 *mat64.Dense
	Affine1 *AffineLayer_param
	Affine2 *AffineLayer_param
	Relu1 *ReluLayer_param
	SoftmaxWithLoss *SoftmaxWithLossLayer_param
}

// type TwoLayerNet_layers struct {
// 	Affine1 *AffineLayer_param
// 	Affine2 *AffineLayer_param
// 	Relu1 *ReluLayer_param
// 	SoftmaxWithLoss *SoftmaxWithLossLayer_param
// }

type Grads struct {
	w1 *mat64.Dense
	b1 *mat64.Dense
	w2 *mat64.Dense
	b2 *mat64.Dense
}

func (p *TwoLayerNet_params) init(input_size,hidden_size,output_size int,weight_init_std float64) {
	w1_init := make([]float64, input_size * hidden_size)
	for i := range w1_init {
		w1_init[i] = rand.NormFloat64()
	}
	p.w1 = mat64.NewDense(input_size, hidden_size, w1_init)
	p.b1 = mat64.NewDense(hidden_size,1 , nil)
	w2_init := make([]float64, hidden_size * output_size )
	for i := range w2_init {
		w2_init[i] = rand.NormFloat64()

	}
	p.w2 = mat64.NewDense(hidden_size, output_size, w2_init)
	p.b2 = mat64.NewDense(output_size,1, nil)
	p.Affine1 = &AffineLayer_param{}
	p.Affine1.init(p.w1,p.b1)
	p.Relu1 =&ReluLayer_param{}
	p.Relu1.init()
	p.Affine2 = &AffineLayer_param{}
	p.Affine2.init(p.w2,p.b2)
	p.SoftmaxWithLoss = &SoftmaxWithLossLayer_param{}
	p.SoftmaxWithLoss.init()
}

//func (p *TwoLayerNet_params) predict(x *mat64.Dense) *mat64.Dense {
// 	x = p.Affine1.forward(x)
// 	x = p.Relu1.forward(x)
// 	x = p.Affine2.forward(x)
// 	return x
//}

func (p *TwoLayerNet_params) loss(x,t *mat64.Dense) *mat64.Dense {
	x = p.Affine1.forward(x)
	x = p.Relu1.forward(x)
	y := p.Affine2.forward(x)
	return p.SoftmaxWithLoss.forward(y,t)
}

func (p *TwoLayerNet_params) gradient(x ,t *mat64.Dense) *Grads {
	x = p.Affine1.forward(x)
	x = p.Relu1.forward(x)
	y := p.Affine2.forward(x)
	p.SoftmaxWithLoss.forward(y,t)
	dout_first := 1
	p.SoftmaxWithLoss.init()
	dout := p.SoftmaxWithLoss.backward(dout_first)
	p.Affine2.init(p.w2,p.b2)
	dout = p.Affine2.backward(dout)
	p.Relu1.init()
	dout = p.Relu1.backward(dout)
	p.Affine1.init(p.w1,p.b1)
	dout = p.Affine1.backward(dout)
	grads := &Grads{}
	grads.w1 = p.Affine1.dw
	grads.b1 = p.Affine1.db
	grads.w2 = p.Affine2.dw
	grads.b2 = p.Affine2.db
	return grads
}


func main() {
	train, test, err := read.Load("./dataset/mnist/")
	err = err
	test = test
	//fmt.Println(train.Images[0])
	tmp :=train.Labels[0]
	fmt.Println(train.Labels[0])
	if (tmp == 5 ) {
		fmt.Println(tmp)
	}
	network := &TwoLayerNet_params{}
	iters_num := float64(1000)
	train_size := float64(len(train.Labels))
	batch_size := float64(100)
	learning_rate := 0.1
	tests := mat64.NewDense(1, 1, train.Labels)
	iter_per_epoch := math.Max(train_size/ batch_size,1.0)
	for i := 0; i < int(iters_num/batch_size); i++  {
		batch_size_int := int(batch_size)
		x_batch := train.Images[i*batch_size_int:(i+1)*batch_size_int]
		t_batch := train.Labels[i*batch_size_int:(i+1)*batch_size_int]

		grad := network.gradient(x_batch,t_batch)
		network.w1 -= learning_rate * grad.w1
		network.b1 -= learning_rate * grad.b1
		network.w2 -= learning_rate * grad.w2
		network.b2 -= learning_rate * grad.b2

		loss := network.loss(x_batch,t_batch)
	}
}
